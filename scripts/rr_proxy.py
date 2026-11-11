"""Least-busy HTTP proxy in front of several OpenAI-compatible servers on one node.

vLLM's own data parallelism shards an MoE model's experts across the DP ranks, so the Gemma judge fills a
3-GPU node as three independent servers instead (``serve.slurm`` with ``DPT_SERVE_REPLICAS=3``). This
proxy gives them one endpoint: each request goes to the backend with the fewest requests in flight, and a
backend that refuses connections is skipped for a minute. Every request is forwarded at once (no connection
limit), so the queue is vLLM's own; a client that gives up cancels its request upstream too (2026-10-24:
with aiohttp's 100-connection default, verification requests queued in the proxy past their client timeout,
were sent again, and the judge kept working on the abandoned ones).

    python scripts/rr_proxy.py --port 8000 --backend http://127.0.0.1:8001 --backend http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import time

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, TCPConnector, web

PASS_HEADERS = ("content-type", "authorization", "accept")
SESSION = web.AppKey("session", ClientSession)
RUN_KW = {"handler_cancellation": True}       # web.run_app / AppRunner: a closed client cancels its handler


def make_app(backends: list[str], *, down_s: float = 60.0) -> web.Application:
    busy = {b: 0 for b in backends}
    down_until = {b: 0.0 for b in backends}

    def pick() -> str:
        now = time.monotonic()
        up = [b for b in backends if down_until[b] <= now] or backends
        return min(up, key=lambda b: busy[b])

    async def handle(request: web.Request) -> web.Response:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() in PASS_HEADERS}
        tried: set[str] = set()
        while True:
            b = pick()
            if b in tried:
                return web.Response(status=502, text="no backend accepted the connection")
            tried.add(b)
            busy[b] += 1
            try:
                async with request.app[SESSION].request(request.method, b + request.rel_url.path_qs,
                                                        data=body, headers=headers) as r:
                    data = await r.read()
                    return web.Response(status=r.status, body=data,
                                        headers={"Content-Type": r.headers.get("Content-Type", "application/json")})
            except ClientConnectorError:
                down_until[b] = time.monotonic() + down_s
            finally:
                busy[b] -= 1

    async def start(app: web.Application) -> None:
        # No read timeout of our own: vLLM answers a request only when all its samples are done, and a pass@64
        # request queued behind others took over an hour (campaign 3532660, 2026-10-27: sock_read=3600 failed
        # them all at once). The client's timeout governs; a client that gives up cancels the request upstream.
        app[SESSION] = ClientSession(connector=TCPConnector(limit=0),
                                     timeout=ClientTimeout(total=None, sock_connect=30, sock_read=None))

    async def stop(app: web.Application) -> None:
        await app[SESSION].close()

    app = web.Application(client_max_size=256 * 1024 ** 2)
    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backend", action="append", required=True, help="http://host:port of one server")
    args = ap.parse_args(argv)
    web.run_app(make_app([b.rstrip("/") for b in args.backend]), host=args.host, port=args.port,
                access_log=None, print=None, **RUN_KW)


if __name__ == "__main__":
    main()
