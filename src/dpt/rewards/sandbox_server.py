"""HTTP sandbox worker for the isolated CPU cluster (report §为什么这样选: "T2 的代码沙箱跑在独立的 CPU 集群上").

Protocol (JSON over HTTP, stdlib only so the sandbox image stays minimal):

    GET  /health  → {"ok": true, "warm": bool, "inflight": int}
    POST /run     ← SandboxRequest JSON        → SandboxResult JSON

Hidden tests are resolved **on this host** from ``--hidden-tests-dir``. The
training cluster sends only ``hidden_tests_id`` and never sees test source.
That is what "tests replayed outside the sandbox" means operationally.

Warm-up: at start the worker runs one trivial request, so the first real
request does not pay interpreter/cache cold-start. This was part of cutting
the reward tail from 0.5 to 0.3 min/step (report §单步时间预算: "沙箱预热").

Run one worker per core group inside a network-less container::

    python -m dpt.rewards.sandbox_server --port 8080 --hidden-tests-dir /srv/hidden_tests \
        --max-concurrency 16 --token-env DPT_SANDBOX_TOKEN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dpt.common.logging import get_logger
from dpt.rewards.code_sandbox import HiddenTestStore, LocalSubprocessSandbox, SandboxRequest, SandboxResult

log = get_logger(__name__)

_MAX_BODY = 4 * 1024 * 1024  # 4 MB of code is already absurd


class SandboxService:
    """Concurrency-limited wrapper around :class:`LocalSubprocessSandbox`."""

    def __init__(self, sandbox: LocalSubprocessSandbox, max_concurrency: int = 8, token: str | None = None):
        self.sandbox = sandbox
        self.sem = threading.BoundedSemaphore(max_concurrency)
        self.token = token
        self.warm = False
        self._inflight = 0
        self._lock = threading.Lock()

    def warmup(self) -> None:
        res = self.sandbox.run(
            SandboxRequest(
                code="def f():\n    return 1\n",
                tests="from solution import f\ndef test_f():\n    assert f() == 1\n",
                timeout_s=10,
            )
        )
        self.warm = res.status == "ok" and res.passed == 1
        log.info("sandbox warm-up: %s", res)

    def handle(self, payload: dict[str, Any]) -> SandboxResult:
        fields = SandboxRequest.__dataclass_fields__
        req = SandboxRequest(**{k: v for k, v in payload.items() if k in fields})
        with self.sem:
            with self._lock:
                self._inflight += 1
            try:
                return self.sandbox.run(req)
            finally:
                with self._lock:
                    self._inflight -= 1

    @property
    def inflight(self) -> int:
        return self._inflight


def make_handler(service: SandboxService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # route to our logger
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/health":
                self._send(200, {"ok": True, "warm": service.warm, "inflight": service.inflight})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self._send(404, {"error": "not found"})
                return
            if service.token and self.headers.get("X-Sandbox-Token") != service.token:
                self._send(403, {"error": "forbidden"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > _MAX_BODY:
                self._send(413, {"error": "bad body size"})
                return
            try:
                payload = json.loads(self.rfile.read(n))
                res = service.handle(payload)
            except Exception as e:  # report as infra_error; the client fails closed
                self._send(
                    200, SandboxResult(status="infra_error", detail=f"{type(e).__name__}: {e}").__dict__
                )
                return
            self._send(200, res.__dict__)

    return Handler


def serve(service: SandboxService, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """Create (but do not start) the HTTP server. ``port=0`` picks a free port (tests)."""
    return ThreadingHTTPServer((host, port), make_handler(service))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--hidden-tests-dir", required=True)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--cpu-s", type=int, default=10)
    ap.add_argument("--mem-mb", type=int, default=2048)
    ap.add_argument("--nproc", type=int, default=None, help="only with a dedicated sandbox user")
    ap.add_argument("--network-isolation", default="auto", choices=["auto", "unshare", "none"])
    ap.add_argument("--token-env", default=None, help="env var holding the shared request token")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the solutions (a small env)")
    ap.add_argument("--container-image", default=os.environ.get("DPT_SANDBOX_SIF"),
                    help="Apptainer image: solutions see only their sandbox dir (default $DPT_SANDBOX_SIF)")
    ap.add_argument("--container-network", default="none", choices=["none", "host"])
    args = ap.parse_args()

    sandbox = LocalSubprocessSandbox(
        hidden_store=HiddenTestStore(args.hidden_tests_dir),
        cpu_s=args.cpu_s,
        mem_mb=args.mem_mb,
        nproc=args.nproc,
        network_isolation=args.network_isolation,
        python=args.python,
        container_image=args.container_image,
        container_network=args.container_network,
    )
    token = os.environ.get(args.token_env) if args.token_env else None
    service = SandboxService(sandbox, max_concurrency=args.max_concurrency, token=token)
    service.warmup()
    httpd = serve(service, args.host, args.port)
    log.info("sandbox server on %s:%d (warm=%s)", args.host, httpd.server_address[1], service.warm)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
