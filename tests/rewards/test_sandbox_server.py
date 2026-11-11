"""Remote sandbox protocol: sandbox_server worker + RemoteSandboxClient over real local HTTP."""

from __future__ import annotations

import threading

import httpx
import pytest

from dpt.rewards.code_sandbox import (
    HiddenTestStore,
    LocalSubprocessSandbox,
    RemoteSandboxClient,
    SandboxRequest,
)
from dpt.rewards.sandbox_server import SandboxService, serve


@pytest.fixture()
def server(tmp_path):
    (tmp_path / "add_1.py").write_text("from solution import add\ndef test_a():\n    assert add(1, 2) == 3\n")
    svc = SandboxService(
        LocalSubprocessSandbox(hidden_store=HiddenTestStore(tmp_path)), max_concurrency=2, token="s3cret"
    )
    httpd = serve(svc, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", svc
    httpd.shutdown()


def test_remote_roundtrip_with_hidden_tests(server):
    url, _ = server
    client = RemoteSandboxClient(url, token="s3cret")
    r = client.run(SandboxRequest(code="def add(a, b):\n    return a + b\n", hidden_tests_id="add_1"))
    assert (r.passed, r.total, r.status) == (1, 1, "ok")


def test_bad_token_fails_closed(server):
    url, _ = server
    r = RemoteSandboxClient(url, token="wrong", retries=0).run(
        SandboxRequest(code="x=1", hidden_tests_id="add_1")
    )
    assert r.status == "infra_error"


def test_health_and_warmup(server):
    url, svc = server
    svc.warmup()
    h = httpx.get(f"{url}/health").json()
    assert h["ok"] and h["warm"]


def test_eval_hidden_test_client_speaks_the_same_protocol(server):
    """The eval side (independent grader) must interoperate with the sandbox workers."""
    from dpt.eval.graders.code import HiddenTestClient

    url, _ = server
    res = HiddenTestClient(url, token="s3cret").run("def add(a, b):\n    return a + b\n",
                                                    {"hidden_tests_id": "add_1", "entry_point": "add"})
    assert (res.passed, res.total) == (1, 1)
