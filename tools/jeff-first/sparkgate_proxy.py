"""A pass-through server in front of the shared Spark model server that respects sparkgate.

sparkgate (~/jeff-finetunes/sparkgate.py on datigator) is a Python library: every job that sends requests to the
Spark holds one machine-wide slot per request, so all jobs together stay under the limit in ~/.spark-slots/limit.
pi runs in Node inside Docker containers and cannot take those slots itself. This server takes them on its behalf:
each request first waits for one of this job's own slots (the cap in ~/.spark-slots/cap-<job>), then for a
machine-wide sparkgate slot, then goes to the Spark, and the answer is streamed back. The slot is held until the
answer has been fully sent.

If the Spark cannot be reached, the request fails with status 502 and a message naming the Spark. There is no
fallback to another server.

Usage (on datigator):
    uv run python sparkgate_proxy.py --upstream http://spark-fa14:8888 --host 0.0.0.0 --port 8899 \
        --job jeffpi --sparkgate-dir ~/jeff-finetunes
"""

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import AsyncContextManager

from aiohttp import web

from proxy_common import DROP_REQUEST_HEADERS, add_client_session, relay


def make_app(upstream: str, cap: int, take_slot: Callable[[], AsyncContextManager[None]]) -> web.Application:
    if cap < 1:
        raise ValueError(f"the job's cap must be at least 1, not {cap}")
    own_slots = asyncio.Semaphore(cap)
    app = web.Application(client_max_size=64 * 1024 * 1024)

    add_client_session(app, sock_read=None)

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST_HEADERS}
        # Wait for this job's own slot first, so a queued request never holds a machine-wide slot.
        async with own_slots, take_slot():

            def error_message(error):
                return f"sparkgate proxy: could not reach the Spark at {upstream}: {error}"

            return await relay(request, app, f"{upstream}{request.rel_url}", headers, body, error_message)

    app.router.add_route("*", "/v1/{tail:.*}", forward)
    return app


def read_cap(job: str) -> int:
    path = Path.home() / ".spark-slots" / f"cap-{job}"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: write this job's request cap into it, for example 2")
    return int(path.read_text().strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", required=True, help="the Spark model server, for example http://spark-fa14:8888")
    parser.add_argument("--host", required=True, help="address to listen on")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--job", required=True, help="job name; its cap is read from ~/.spark-slots/cap-<job>")
    parser.add_argument("--sparkgate-dir", required=True, type=Path, help="folder that holds sparkgate.py")
    args = parser.parse_args()

    sparkgate_dir = args.sparkgate_dir.expanduser()
    if not (sparkgate_dir / "sparkgate.py").exists():
        raise FileNotFoundError(f"{sparkgate_dir / 'sparkgate.py'} is missing")
    sys.path.insert(0, str(sparkgate_dir))
    from sparkgate import aslot  # noqa: PLC0415 - its folder is only known at run time

    cap = read_cap(args.job)
    print(f"sparkgate proxy: {args.host}:{args.port} -> {args.upstream}, job {args.job}, cap {cap}", flush=True)
    web.run_app(make_app(args.upstream.rstrip("/"), cap, aslot), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
