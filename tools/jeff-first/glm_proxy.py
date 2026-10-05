"""A pass-through server in front of the GLM endpoint that adds the real API key and limits requests at once.

Jeff-Code runs inside each task's Docker container, so any key it holds can be printed by the model (GLM once ran `env`
and its key ended up in the traces). The containers therefore send "unused" as their key; this server, running on
datigator, replaces the Authorization header with the real key, read from the environment variable GLM_API_KEY.
At most --cap requests are sent at once; the rest wait here.

Only the two calls Jeff-Code and the teacher make are relayed: POST /v1/chat/completions and GET /v1/models. Every other
path answers 404 and never reaches the endpoint, so a container cannot use the real key for anything else (for example
to read the key's own details). A request whose answer stalls for more than 300 seconds fails with status 502, so a
hung GLM request cannot hold one of the --cap slots forever.

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

from aiohttp import web

from proxy_common import DROP_REQUEST_HEADERS, add_client_session, relay

READ_TIMEOUT_SECONDS = 300


def make_app(upstream: str, cap: int, key: str, read_timeout_seconds: float) -> web.Application:
    if not key:
        raise ValueError("the GLM key is empty: set GLM_API_KEY")
    if cap < 1:
        raise ValueError(f"the cap must be at least 1, not {cap}")
    slots = asyncio.Semaphore(cap)
    app = web.Application(client_max_size=64 * 1024 * 1024)

    add_client_session(app, sock_read=read_timeout_seconds)

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        # Drop the shared headers plus authorization (which GLM will replace with the real key).
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST_HEADERS and k.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {key}"
        async with slots:

            def error_message(error):
                return f"GLM proxy: could not reach the GLM endpoint at {upstream}: {type(error).__name__}"

            return await relay(request, app, f"{upstream}{request.rel_url}", headers, body, error_message)

    app.router.add_post("/v1/chat/completions", forward)
    app.router.add_get("/v1/models", forward)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", required=True, help="the GLM endpoint without /v1, e.g. https://litellm.scissero.com")
    parser.add_argument("--host", required=True, help="address to listen on")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cap", required=True, type=int, help="most requests sent at once")
    args = parser.parse_args()
    key = os.environ.get("GLM_API_KEY", "")
    app = make_app(args.upstream.rstrip("/"), args.cap, key, READ_TIMEOUT_SECONDS)
    print(f"GLM proxy: {args.host}:{args.port} -> {args.upstream}, cap {args.cap}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
