"""scripts/rr_proxy.py: one endpoint in front of independent judge replicas (serve.slurm DPT_SERVE_REPLICAS)."""

from __future__ import annotations

import asyncio
import importlib.util
import socket
from pathlib import Path

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _proxy_module():
    spec = importlib.util.spec_from_file_location("rr_proxy", REPO / "scripts" / "rr_proxy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve(app: web.Application, port: int, **kw) -> web.AppRunner:
    runner = web.AppRunner(app, **kw)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def _backend(name: str, delay: float) -> web.Application:
    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        await asyncio.sleep(delay)
        return web.json_response({"backend": name, "echo": body["prompt"]})

    app = web.Application()
    app.router.add_post("/v1/completions", handle)
    return app


def test_proxy_spreads_requests_and_skips_a_dead_backend():
    mod = _proxy_module()

    async def run():
        ports = [_free_port() for _ in range(3)]
        runners = [await _serve(_backend(n, 0.2), p) for n, p in zip("ab", ports)]
        dead = f"http://127.0.0.1:{_free_port()}"          # nothing listens there
        proxy_port = _free_port()
        backends = [f"http://127.0.0.1:{p}" for p in ports[:2]] + [dead]
        runners.append(await _serve(mod.make_app(backends), proxy_port))
        try:
            async with aiohttp.ClientSession() as s:
                async def one(i):
                    async with s.post(f"http://127.0.0.1:{proxy_port}/v1/completions", json={"prompt": i}) as r:
                        return r.status, await r.json()
                out = await asyncio.gather(*(one(i) for i in range(8)))
        finally:
            for r in runners:
                await r.cleanup()
        return out

    out = asyncio.run(run())
    assert all(status == 200 for status, _ in out)
    assert [body["echo"] for _, body in out] == list(range(8))           # each reply is its own request's
    assert {body["backend"] for _, body in out} == {"a", "b"}             # both live replicas used, dead one skipped


def test_every_request_reaches_the_backend_and_a_client_that_gives_up_cancels_it():
    mod = _proxy_module()
    seen = {"now": 0, "max": 0, "cancelled": 0}

    async def slow(request: web.Request) -> web.Response:
        seen["now"] += 1
        seen["max"] = max(seen["max"], seen["now"])
        try:
            await asyncio.sleep(float((await request.json())["prompt"]))
            return web.json_response({"ok": True})
        except asyncio.CancelledError:
            seen["cancelled"] += 1
            raise
        finally:
            seen["now"] -= 1

    async def run():
        backend = web.Application()
        backend.router.add_post("/v1/completions", slow)
        bport, pport = _free_port(), _free_port()
        runners = [await _serve(backend, bport, handler_cancellation=True),
                   await _serve(mod.make_app([f"http://127.0.0.1:{bport}"]), pport, **mod.RUN_KW)]
        url = f"http://127.0.0.1:{pport}/v1/completions"
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:   # 150 at once
                async def one(delay):
                    async with s.post(url, json={"prompt": delay}) as r:
                        return r.status
                assert set(await asyncio.gather(*(one(2.0) for _ in range(150)))) == {200}
                try:                                              # this client gives up after 0.2 s
                    async with s.post(url, json={"prompt": 30}, timeout=aiohttp.ClientTimeout(total=0.2)) as r:
                        await r.read()
                except asyncio.TimeoutError:
                    pass
                for _ in range(50):
                    if seen["cancelled"]:
                        break
                    await asyncio.sleep(0.1)
        finally:
            for r in runners:
                await r.cleanup()

    asyncio.run(run())
    assert seen["max"] > 100                  # aiohttp's default 100-connection limit would hold the rest back
    assert seen["cancelled"] == 1             # the backend stops work nobody waits for


def test_the_proxy_sets_no_read_timeout_of_its_own():
    # 2026-10-27: a pass@64 request waited over an hour in vLLM's queue; only the client's timeout may end it
    mod = _proxy_module()

    async def run():
        app = mod.make_app(["http://127.0.0.1:1"])
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            return app[mod.SESSION].timeout
        finally:
            await runner.cleanup()

    t = asyncio.run(run())
    assert t.sock_read is None and t.total is None and t.sock_connect == 30
