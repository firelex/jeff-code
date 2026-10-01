import asyncio
import contextlib

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from sparkgate_proxy import make_app


class FakeGate:
    """Stands in for sparkgate.aslot: counts slots taken and how many are held at once."""

    def __init__(self):
        self.taken = 0
        self.held = 0

    @contextlib.asynccontextmanager
    async def aslot(self):
        self.taken += 1
        self.held += 1
        try:
            yield
        finally:
            self.held -= 1


async def start(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def fake_spark(gate, state):
    async def chat(request):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        state["slot_held_during_request"] = gate.held >= 1
        state["body"] = await request.json()
        state["auth"] = request.headers.get("Authorization")
        await asyncio.sleep(state.get("delay", 0))
        state["in_flight"] -= 1
        if state["body"].get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for chunk in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
                await response.write(chunk)
            await response.write_eof()
            return response
        if state["body"].get("bad"):
            return web.json_response({"error": {"message": "bad request"}}, status=400)
        return web.json_response({"choices": [{"message": {"content": "hi"}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    async def models(request):
        return web.json_response({"data": [{"id": "qwen3.8-flash-next"}]})

    app.router.add_get("/v1/models", models)
    return app


async def setup(cap=2, delay=0):
    gate = FakeGate()
    state = {"in_flight": 0, "max_in_flight": 0, "delay": delay}
    spark = await start(fake_spark(gate, state))
    proxy = await start(make_app(str(spark.make_url("")).rstrip("/"), cap, gate.aslot))
    return gate, state, spark, proxy


async def test_forwards_requests_and_takes_one_slot_each():
    gate, state, spark, proxy = await setup()
    try:
        response = await proxy.post("/v1/chat/completions", json={"model": "m"}, headers={"Authorization": "Bearer x"})
        assert response.status == 200
        assert await response.json() == {"choices": [{"message": {"content": "hi"}}]}
        assert state["body"] == {"model": "m"}
        assert state["auth"] == "Bearer x"
        assert gate.taken == 1
        assert state["slot_held_during_request"] is True
        models = await proxy.get("/v1/models")
        assert (await models.json())["data"][0]["id"] == "qwen3.8-flash-next"
        assert gate.taken == 2
    finally:
        await proxy.close()
        await spark.close()


async def test_streams_the_answer_through():
    gate, state, spark, proxy = await setup()
    try:
        response = await proxy.post("/v1/chat/completions", json={"stream": True})
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert await response.read() == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
        assert gate.held == 0
    finally:
        await proxy.close()
        await spark.close()


async def test_never_has_more_requests_in_flight_than_its_cap():
    gate, state, spark, proxy = await setup(cap=2, delay=0.05)
    try:
        responses = await asyncio.gather(*(proxy.post("/v1/chat/completions", json={}) for _ in range(5)))
        assert [r.status for r in responses] == [200] * 5
        assert state["max_in_flight"] == 2
        assert gate.taken == 5
    finally:
        await proxy.close()
        await spark.close()


async def test_passes_upstream_errors_through_unchanged():
    gate, state, spark, proxy = await setup()
    try:
        response = await proxy.post("/v1/chat/completions", json={"bad": True})
        assert response.status == 400
        assert await response.json() == {"error": {"message": "bad request"}}
    finally:
        await proxy.close()
        await spark.close()


async def test_reports_an_unreachable_spark_as_502_naming_it():
    gate = FakeGate()
    proxy = await start(make_app("http://127.0.0.1:9", 2, gate.aslot))
    try:
        response = await proxy.post("/v1/chat/completions", json={})
        assert response.status == 502
        message = (await response.json())["error"]["message"]
        assert "could not reach the Spark at http://127.0.0.1:9" in message
        assert gate.held == 0
    finally:
        await proxy.close()
