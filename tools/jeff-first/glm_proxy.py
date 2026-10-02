"""A pass-through server in front of the GLM endpoint that adds the real API key and limits requests at once.

pi runs inside each task's Docker container, so any key it holds can be printed by the model (GLM once ran `env`
and its key ended up in the traces). The containers therefore send "unused" as their key; this server, running on
datigator, replaces the Authorization header with the real key, read from the environment variable GLM_API_KEY.
At most --cap requests are sent at once; the rest wait here.

If the endpoint cannot be reached, the request fails with status 502 and a message naming the endpoint (never the
key). There is no fallback to another server.

Usage (on datigator, started by the owner, because commands that handle the key are blocked for Claude Code):
    . ~/jeff-pi-run/.glm-key && GLM_API_KEY="$JEFF_RUN_API_KEY" uv run python glm_proxy.py \
        --upstream https://litellm.scissero.com --host 0.0.0.0 --port 8898 --cap 10
"""

import argparse
import asyncio
import contextlib
import os

import aiohttp
from aiohttp import web

DROP_REQUEST_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding", "authorization"}
DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection"}
SESSION = web.AppKey("session", aiohttp.ClientSession)


def make_app(upstream: str, cap: int, key: str) -> web.Application:
    if not key:
        raise ValueError("the GLM key is empty: set GLM_API_KEY")
    if cap < 1:
        raise ValueError(f"the cap must be at least 1, not {cap}")
    slots = asyncio.Semaphore(cap)
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def client_session(app: web.Application):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
            app[SESSION] = session
            yield

    app.cleanup_ctx.append(client_session)

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST_HEADERS}
        headers["Authorization"] = f"Bearer {key}"
        async with slots:
            try:
                upstream_response = await app[SESSION].request(
                    request.method, f"{upstream}{request.rel_url}", headers=headers, data=body
                )
            except aiohttp.ClientConnectionError as error:
                message = f"GLM proxy: could not reach the GLM endpoint at {upstream}: {type(error).__name__}"
                return web.json_response({"error": {"message": message}}, status=502)
            async with upstream_response:
                response = web.StreamResponse(
                    status=upstream_response.status,
                    headers={
                        k: v for k, v in upstream_response.headers.items() if k.lower() not in DROP_RESPONSE_HEADERS
                    },
                )
                await response.prepare(request)
                async for chunk in upstream_response.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response

    app.router.add_route("*", "/v1/{tail:.*}", forward)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", required=True, help="the GLM endpoint without /v1, e.g. https://litellm.scissero.com")
    parser.add_argument("--host", required=True, help="address to listen on")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cap", required=True, type=int, help="most requests sent at once")
    args = parser.parse_args()
    key = os.environ.get("GLM_API_KEY", "")
    app = make_app(args.upstream.rstrip("/"), args.cap, key)
    print(f"GLM proxy: {args.host}:{args.port} -> {args.upstream}, cap {args.cap}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
