import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from glm_proxy import make_app


async def start(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def fake_upstream(seen, delay=0.0):
    held = {"now": 0, "most": 0}

    async def handler(request):
        held["now"] += 1
        held["most"] = max(held["most"], held["now"])
        seen.append(request.headers.get("Authorization"))
        await asyncio.sleep(delay)
        held["now"] -= 1
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    return server, held


async def test_replaces_the_callers_key_with_the_real_one():
    seen = []
    upstream, _ = await fake_upstream(seen)
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real"))
    response = await client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer unused"})
    assert response.status == 200
    assert seen == ["Bearer sk-real"]
    await client.close()
    await upstream.close()


async def test_never_sends_more_requests_at_once_than_the_cap():
    seen = []
    upstream, held = await fake_upstream(seen, delay=0.05)
    client = await start(make_app(str(upstream.make_url("")).rstrip("/"), 2, "sk-real"))
    await asyncio.gather(*(client.post("/v1/chat/completions", json={}) for _ in range(6)))
    assert len(seen) == 6
    assert held["most"] == 2
    await client.close()
    await upstream.close()


async def test_answers_502_naming_the_upstream_when_it_cannot_be_reached():
    client = await start(make_app("http://127.0.0.1:9", 2, "sk-real"))
    response = await client.post("/v1/chat/completions", json={})
    assert response.status == 502
    body = await response.json()
    assert "could not reach the GLM endpoint at http://127.0.0.1:9" in body["error"]["message"]
    assert "sk-real" not in body["error"]["message"]
    await client.close()


def test_refuses_an_empty_key():
    try:
        make_app("http://x", 2, "")
    except ValueError as error:
        assert "GLM_API_KEY" in str(error)
    else:
        raise AssertionError("an empty key was accepted")
