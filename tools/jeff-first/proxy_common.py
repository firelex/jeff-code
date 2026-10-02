"""Shared relay logic for proxy servers (sparkgate_proxy and glm_proxy)."""

from collections.abc import Callable

import aiohttp
from aiohttp import web

# Hop-by-hop and encoding headers are set again by each side of the proxy.
DROP_REQUEST_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}
DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection"}
SESSION = web.AppKey("session", aiohttp.ClientSession)


def add_client_session(app: web.Application) -> None:
    """Register the client session cleanup context with the app."""

    async def client_session(app: web.Application):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
            app[SESSION] = session
            yield

    app.cleanup_ctx.append(client_session)


async def relay(
    request: web.Request,
    app: web.Application,
    url: str,
    headers: dict[str, str],
    body: bytes,
    unreachable_message: Callable[[aiohttp.ClientConnectionError], str],
) -> web.StreamResponse:
    """Forward a request to an upstream server and stream the response back.

    On ClientConnectionError, calls unreachable_message(error) to get the error message and returns 502 JSON.
    """
    try:
        upstream_response = await app[SESSION].request(request.method, url, headers=headers, data=body)
    except aiohttp.ClientConnectionError as error:
        message = unreachable_message(error)
        return web.json_response({"error": {"message": message}}, status=502)
    async with upstream_response:
        response = web.StreamResponse(
            status=upstream_response.status,
            headers={k: v for k, v in upstream_response.headers.items() if k.lower() not in DROP_RESPONSE_HEADERS},
        )
        await response.prepare(request)
        async for chunk in upstream_response.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        return response
