import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from glm_proxy import make_app


async def start(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def fake_upstream(seen, delay=0.0, paths=None):
    held = {"now": 0, "most": 0}

    async def anything(request):
        paths.append(f"{request.method} {request.path}")
        return web.json_response({"leaked": True})

    async def handler(request):
        held["now"] += 1
        held["most"] = max(held["most"], held["now"])
        seen.append(request.headers.get("Authorization"))
        await asyncio.sleep(delay)
        held["now"] -= 1
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    async def models(request):
        return web.json_response({"data": [{"id": "glm"}]})

    app.router.add_get("/v1/models", models)
    if paths is not None:
        app.router.add_route("*", "/{tail:.*}", anything)
    server = TestServer(app)
    await server.start_server()
    return server, held


async def test_replaces_the_callers_key_with_the_real_one():
    seen = []
    upstream, _ = await fake_upstream(seen)
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real", 300))
    response = await client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer unused"})
    assert response.status == 200
    assert seen == ["Bearer sk-real"]
    await client.close()
    await upstream.close()


async def test_never_sends_more_requests_at_once_than_the_cap():
    seen = []
    upstream, held = await fake_upstream(seen, delay=0.05)
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real", 300))
    await asyncio.gather(*(client.post("/v1/chat/completions", json={}) for _ in range(6)))
    assert len(seen) == 6
    assert held["most"] == 2
    await client.close()
    await upstream.close()


async def test_answers_502_naming_the_upstream_when_it_cannot_be_reached():
    client = await start(make_app("http://127.0.0.1:9", 2, "sk-real", 300))
    response = await client.post("/v1/chat/completions", json={})
    assert response.status == 502
    body = await response.json()
    assert "could not reach the GLM endpoint at http://127.0.0.1:9" in body["error"]["message"]
    assert "sk-real" not in body["error"]["message"]
    await client.close()


def test_refuses_an_empty_key():
    try:
        make_app("http://x", 2, "", 300)
    except ValueError as error:
        assert "GLM_API_KEY" in str(error)
    else:
        raise AssertionError("an empty key was accepted")


async def test_relays_the_model_list():
    upstream, _ = await fake_upstream([])
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real", 300))
    response = await client.get("/v1/models")
    assert response.status == 200
    assert (await response.json())["data"] == [{"id": "glm"}]
    await client.close()
    await upstream.close()


async def test_answers_404_for_any_other_path_without_reaching_the_endpoint():
    paths = []
    upstream, _ = await fake_upstream([], paths=paths)
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real", 300))
    assert (await client.get("/v1/key/info")).status == 404
    assert (await client.post("/v1/embeddings", json={})).status == 404
    assert (await client.get("/v1/chat/completions")).status in (404, 405)
    assert (await client.post("/key/generate", json={})).status == 404
    assert paths == []
    await client.close()
    await upstream.close()


async def test_answers_502_when_the_endpoint_hangs_longer_than_the_read_timeout():
    async def hang(request):
        await asyncio.sleep(5)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", hang)
    upstream = TestServer(app)
    await upstream.start_server()
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 1, "sk-real", 0.2))
    response = await client.post("/v1/chat/completions", json={})
    assert response.status == 502
    assert "could not reach the GLM endpoint" in (await response.json())["error"]["message"]
    await client.close()
    await upstream.close()
