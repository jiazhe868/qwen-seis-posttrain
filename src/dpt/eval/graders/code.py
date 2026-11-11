"""Code grader for D-Code and LiveCodeBench: hidden tests decide correctness.

D-Code reports pass@1, averaged over 4 samples, judged by hidden tests
(report §评测协议). A sample counts as passing only if **all** hidden tests
pass. ``score`` keeps the passed fraction for diagnostics.

Two runners share one interface:

* :class:`HiddenTestClient`: the production path. Hidden tests live only on
  the isolated CPU sandbox cluster and are referenced by id. The eval
  machine never sees them, and neither does any training process. The
  sandbox replays the tests outside the candidate's process, has no network,
  and exposes no environment variables that identify a test run. These are
  the defenses learned from the step-205 hack (report §Reward hacking).
* :class:`LocalTestRunner`: a development stand-in for the 4×A100
  workstation and for unit tests. It runs the candidate in a separate
  ``python -I`` subprocess with CPU/memory limits and an empty environment.
  It is **not** a security boundary: it has no network isolation and the
  candidate can read its own process. Never point it at untrusted model
  output outside a disposable machine.

Both fail closed. A timeout, crash, missing result record or transport error
counts as zero tests passed.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from dpt.eval.graders import GradeResult

_FENCE_RE = re.compile(r"```[ \t]*(?P<lang>[A-Za-z0-9_+\-]*)[ \t]*\n(?P<body>.*?)```", re.DOTALL)


def extract_code(answer_text: str, language: str = "python") -> str | None:
    """Return the last fenced code block (preferring blocks tagged with ``language``)."""
    blocks = [(m.group("lang").lower(), m.group("body")) for m in _FENCE_RE.finditer(answer_text)]
    if not blocks:
        return None
    tagged = [b for lang, b in blocks if lang in (language, "py" if language == "python" else language)]
    return (tagged or [b for _, b in blocks])[-1]


@dataclass(frozen=True)
class TestRunResult:
    passed: int
    total: int
    status: str = "ok"  # "ok" | "timeout" | "error" | "transport_error" | "no_result"

    @property
    def all_passed(self) -> bool:
        return self.status == "ok" and self.total > 0 and self.passed == self.total


class TestRunner(Protocol):
    def run(self, code: str, gold: dict[str, Any]) -> TestRunResult: ...


class HiddenTestClient:
    """Client for the sandbox service.

    Wire contract = the sandbox worker's (``dpt.rewards.sandbox_server``; the protocol is shared,
    the grading logic is not): ``POST {endpoint}/run`` with header ``X-Sandbox-Token`` (if the
    worker requires one) and body ``{"code", "hidden_tests_id", "entry_point", "language",
    "timeout_s", "kind": "hidden"}``. The response is ``{"status", "passed", "total", ...}``.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = 120.0, client: httpx.Client | None = None,
                 token: str | None = None, token_env: str = "DPT_SANDBOX_TOKEN"):
        import os

        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.client = client or httpx.Client(timeout=timeout_s)
        self.token = token if token is not None else os.environ.get(token_env)

    def run(self, code: str, gold: dict[str, Any]) -> TestRunResult:
        payload = {
            "code": code,
            "hidden_tests_id": gold["hidden_tests_id"],
            "entry_point": gold.get("entry_point", ""),
            "language": gold.get("language", "python"),
            "timeout_s": float(gold.get("timeout_s", 30.0)),
            "kind": "hidden",
        }
        headers = {"X-Sandbox-Token": self.token} if self.token else {}
        try:
            r = self.client.post(f"{self.endpoint}/run", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return TestRunResult(int(data["passed"]), int(data["total"]), str(data.get("status", "ok")))
        except Exception:
            return TestRunResult(0, 0, "transport_error")


_HARNESS = r"""
import json, sys, importlib.util, traceback
nonce = sys.stdin.readline().strip()
spec = importlib.util.spec_from_file_location("tests_mod", "tests.py")
mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, ".")
try:
    spec.loader.exec_module(mod)
except BaseException:
    print(nonce + " " + json.dumps({"passed": 0, "total": 0, "status": "error"}), flush=True)
    raise SystemExit(0)
tests = [getattr(mod, n) for n in sorted(dir(mod)) if n.startswith("test_") and callable(getattr(mod, n))]
passed = 0
for t in tests:
    try:
        t()
        passed += 1
    except BaseException:
        pass
print(nonce + " " + json.dumps({"passed": passed, "total": len(tests), "status": "ok"}), flush=True)
"""


def _limit_resources(mem_bytes: int, cpu_s: int):  # pragma: no cover - runs in child
    def apply():
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        os.setsid()

    return apply


class LocalTestRunner:
    """Dev-only runner: ``gold["visible_tests"]`` or ``gold["tests_source"]`` hold test functions.

    The candidate code is saved as ``solution.py``, and the tests are expected
    to ``from solution import ...``. The result line carries a per-run random
    nonce. A crash, ``sys.exit`` or ``os._exit`` before the harness prints
    that line counts as zero passed.
    """

    def __init__(self, *, timeout_s: float = 30.0, mem_mb: int = 2048, python: str = sys.executable):
        self.timeout_s, self.mem_mb, self.python = timeout_s, mem_mb, python

    def run(self, code: str, gold: dict[str, Any]) -> TestRunResult:
        tests = gold.get("tests_source") or gold.get("visible_tests")
        if not tests:
            return TestRunResult(0, 0, "error")
        nonce = secrets.token_hex(16)
        with tempfile.TemporaryDirectory(prefix="dpt_eval_code_") as tmp:
            Path(tmp, "solution.py").write_text(code)
            Path(tmp, "tests.py").write_text(tests)
            Path(tmp, "harness.py").write_text(_HARNESS)
            try:
                proc = subprocess.run(
                    [self.python, "-I", "harness.py"],
                    input=nonce + "\n",
                    cwd=tmp,
                    # One BLAS thread: under RLIMIT_AS, OpenBLAS's per-core buffers make `import numpy`
                    # fail on many-core nodes, which would score every numpy solution 0.
                    env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                         "MKL_NUM_THREADS": "1"},
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    preexec_fn=_limit_resources(self.mem_mb << 20, int(self.timeout_s) + 1),
                )
            except subprocess.TimeoutExpired:
                return TestRunResult(0, 0, "timeout")
        for line in proc.stdout.splitlines()[::-1]:
            if line.startswith(nonce + " "):
                data = json.loads(line[len(nonce) + 1 :])
                return TestRunResult(int(data["passed"]), int(data["total"]), str(data["status"]))
        return TestRunResult(0, 0, "no_result")


def grade_code(
    answer_text: str, gold: dict[str, Any], runner: TestRunner, *, language: str = "python"
) -> GradeResult:
    """Correct iff a code block exists and every hidden test passes."""
    code = extract_code(answer_text, language=gold.get("language", language))
    if code is None:
        return GradeResult.fail("no_code_block")
    res = runner.run(code, gold)
    if res.status != "ok":
        return GradeResult.fail(f"runner_{res.status}", passed=res.passed, total=res.total)
    frac = res.passed / res.total if res.total else 0.0
    return GradeResult(
        correct=res.all_passed,
        score=frac,
        reason="ok" if res.all_passed else "tests_failed",
        details={"passed": res.passed, "total": res.total},
    )
