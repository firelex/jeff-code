"""A front for several jeff-serve instances (one per GPU) at one address: jeff-serve answers one request at a time and
turns others away with 529 ("busy"), so one instance next to a busy Qwen server could not keep up with tonight's 40
Jeff-arm streams on the B200 (sessions ended after 300 s of busy answers). Every instance loads the same base and
adapters, so any idle one may answer any request.

Each request goes to an idle instance (one request at a time per instance, round robin among the idle ones) and its
answer, status included, is passed back unchanged. When every instance is busy the client gets 529 at once, exactly
as from a single busy jeff-serve, and retries. A request to an instance that cannot be reached fails with 502 naming
the instance; nothing is retried here (the client retries server errors). GET /pool reports the instances and counts.

Usage: python jeff_pool.py HOST PORT BACKEND_URL...   (run with jeff-serve's Python: starlette, uvicorn, httpx)
"""

import asyncio
import itertools
import sys

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

HOST, PORT, BACKENDS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
if not BACKENDS:
    raise SystemExit(__doc__)
idle = set(BACKENDS)
order = itertools.cycle(BACKENDS)
counts = {b: {"answered": 0, "failed": 0} for b in BACKENDS}
stats = {"busy": 0}
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
lock = asyncio.Lock()


async def take() -> str | None:
    async with lock:
        for _ in range(len(BACKENDS)):
            b = next(order)
            if b in idle:
                idle.discard(b)
                return b
        return None


async def forward(request: Request) -> Response:
    backend = await take()
    if backend is None:
        stats["busy"] += 1
        return Response(status_code=529)
    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "authorization")}
        try:
            answer = await client.request(request.method, backend + request.url.path, content=body, headers=headers)
        except httpx.HTTPError as error:
            counts[backend]["failed"] += 1
            return JSONResponse({"detail": f"jeff_pool: {backend} unreachable: {error!r}"}, status_code=502)
        counts[backend]["answered"] += 1
        return Response(answer.content, status_code=answer.status_code, media_type=answer.headers.get("content-type"))
    finally:
        idle.add(backend)


async def pool(request: Request) -> Response:
    return JSONResponse({"backends": counts, "idle": sorted(idle), **stats})


app = Starlette(
    routes=[
        Route("/pool", pool, methods=["GET"]),
        Route("/{path:path}", forward, methods=["GET", "POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
