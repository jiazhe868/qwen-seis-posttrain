"""T2 (domain code & tool use) reward: hidden-test pass rate, with tamper checks that zero it.

Reward (report §各任务的 reward): fraction of **hidden** tests passed, in
[0, 1]. Any of the following zeroes it:

* modifying test files (static AST check, the workspace diff, and a runtime
  hash check of the test file before and after the run);
* calling an exit function (``sys.exit``, ``os._exit``, ``quit``,
  ``raise SystemExit``…). An exit mid-test must never look like "no failures";
* reading the tests directory;
* probing the environment to detect "am I being tested?" (step-205 hack:
  code checked a test env-var and returned hard-coded outputs);
* blanket exception swallowing (step-170 hack: a broad ``try/except`` made
  crashing code pass "does not raise" tests);
* frame/GC introspection, which could find the harness's private state.

Layers of defense (report §防范原则 "沙箱隔离"):

1. **Static scan** (:func:`dpt.rewards.hack_patterns.scan_code`) before
   spending sandbox time.
2. **Tests replayed from outside the sandbox.** Hidden tests live only in a
   :class:`HiddenTestStore` on the sandbox host and never in the model's
   workspace or prompt. The harness and the test file sit in directories
   separate from the solution's working directory.
3. **No identifying environment.** The child gets a scrubbed environment
   (no CI/TEST/PYTEST variables) and runs with ``python -I``.
4. **Resource limits.** RLIMIT_CPU / AS / FSIZE / NOFILE (set by the harness
   before the solution is imported), a wall-clock timeout that kills the whole
   process group, and stdout written to a size-capped file.
5. **Network isolation** where the host allows unprivileged namespaces
   (``unshare -rn``). Otherwise network imports are flagged for monitoring.
5b. **Filesystem isolation** (``container_image``): the child runs inside an Apptainer container
   with ``--containall``, so it sees only its own sandbox directory and a read-only Python
   install, not ``$HOME`` / ``$WORK`` / ``$SCRATCH``. Without it the code runs as the training
   user and can delete that user's files. Required on shared clusters, where
   RL runs tens of thousands of model-written programs next to the project's data
   (``dpt.launch.rl`` refuses a cluster launch without it; ``scripts/slurm/check_sandbox.sh``
   verifies the isolation on the real nodes).
6. **Unforgeable results.** The harness prints result lines prefixed with
   a per-run random nonce sent over stdin, which it reads *before* importing
   the solution. The parent counts tests from the test source itself, so the
   solution cannot inflate the total either.

``visible`` tests (the ones shown in the prompt) are also run when
configured. Their pass rate minus the hidden pass rate is logged as
``t2_visible_hidden_gap``. A widening gap is how the step-205 hack was caught
(23-point gap), and monitoring alerts on it.

The local sandbox is a *development* sandbox. Production runs the same
protocol on the isolated CPU cluster through :class:`RemoteSandboxClient` and
``dpt.rewards.sandbox_server``, inside containers without network.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dpt.common.logging import get_logger
from dpt.common.types import T2_CODE
from dpt.rewards.base import RewardResult, fail_closed, zero
from dpt.rewards.hack_patterns import code_flags

log = get_logger(__name__)

DEFAULT_ZERO_ON = (
    "test_tamper",
    "exit_call",
    "env_probe",
    "test_dir_read",
    "blanket_except",
    "introspection",
)


# =============================================================================================
# Protocol types
# =============================================================================================
@dataclass
class SandboxRequest:
    code: str
    tests: str | None = None  # inline test source (visible tests)
    hidden_tests_id: str | None = None  # resolved on the sandbox host only
    entry_point: str = ""
    timeout_s: float = 10.0
    language: str = "python"
    kind: str = "hidden"  # "hidden" | "visible"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SandboxResult:
    passed: int = 0
    total: int = 0
    status: str = "ok"  # ok | timeout | error | infra_error
    tamper: list[str] = field(default_factory=list)
    detail: str = ""
    duration_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SandboxResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class SandboxClient(Protocol):
    def run(self, req: SandboxRequest) -> SandboxResult: ...


class HiddenTestStore:
    """Directory of hidden test files ``<root>/<hidden_tests_id>.py``.

    In production this directory exists only on the sandbox hosts. Training
    processes can neither read it nor know its layout.
    """

    _ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def get(self, test_id: str) -> str:
        if not self._ID_RE.match(test_id) or ".." in test_id:
            raise ValueError(f"invalid hidden test id {test_id!r}")
        return (self.root / f"{test_id}.py").read_text(encoding="utf-8")


# =============================================================================================
# Local subprocess sandbox (development; also the worker inside sandbox_server)
# =============================================================================================
_HARNESS = r"""
import importlib.util, json, resource, sys

# Resource limits are applied here, before any untrusted code is imported. Doing it in the
# harness instead of a Popen preexec_fn keeps the parent thread-safe (rewards run in a pool).
_lim = json.loads(sys.argv[3])
resource.setrlimit(resource.RLIMIT_CPU, (_lim["cpu_s"], _lim["cpu_s"] + 1))
resource.setrlimit(resource.RLIMIT_AS, (_lim["mem"], _lim["mem"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (_lim["fsize"], _lim["fsize"]))
resource.setrlimit(resource.RLIMIT_NOFILE, (_lim["nofile"], _lim["nofile"]))
if _lim.get("nproc"):
    # NB: RLIMIT_NPROC counts all processes of the user; only use it with dedicated sandbox users.
    resource.setrlimit(resource.RLIMIT_NPROC, (_lim["nproc"], _lim["nproc"]))
del _lim

def _make_emit():
    nonce = sys.stdin.readline().strip()
    sys.stdin.close()
    out = sys.__stdout__
    def emit(obj):
        out.write(nonce + json.dumps(obj) + "\n")
        out.flush()
    return emit

_emit = _make_emit()
del _make_emit

def _run(sol_path, test_path):
    spec = importlib.util.spec_from_file_location("solution", sol_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:  # including SystemExit raised at import time
        _emit({"event": "import_error", "error": repr(e)[:300]})
        return
    with open(test_path, "r", encoding="utf-8") as f:
        src = f.read()
    ns = {"__name__": "hidden_tests"}
    try:
        exec(compile(src, "<tests>", "exec"), ns)
    except BaseException as e:
        _emit({"event": "test_import_error", "error": repr(e)[:300]})
        return
    names = sorted(k for k, v in ns.items() if k.startswith("test_") and callable(v))
    for name in names:
        try:
            ns[name]()
            ok = True
        except BaseException:  # SystemExit / KeyboardInterrupt from the solution count as failures
            ok = False
        _emit({"event": "result", "name": name, "ok": ok})
    _emit({"event": "done"})

_run(sys.argv[1], sys.argv[2])
"""

_UNSHARE_OK: bool | None = None


def _unshare_available() -> bool:
    """Probe once whether unprivileged user+net namespaces work on this host."""
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        exe = shutil.which("unshare")
        if not exe:
            _UNSHARE_OK = False
        else:
            try:
                r = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5)
                _UNSHARE_OK = r.returncode == 0
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


_CONTAINER_OK: dict[tuple, bool] = {}


def container_command(runtime: str, image: str, *, root: str, workdir: str, python_prefix: str,
                      env: dict[str, str], network: str = "none") -> list[str]:
    """Apptainer prefix that runs a command with only ``root`` (rw) and ``python_prefix`` (ro) visible.

    ``--containall`` = contained /dev, private /tmp, no home directory, clean environment, own PID
    and IPC namespaces. Host paths are invisible unless bound. Environment variables are passed
    explicitly with ``--env`` because ``--cleanenv`` drops the caller's.
    """
    cmd = [runtime, "exec", "--containall", "--no-home", "--cleanenv",
           "--bind", f"{root}:{root}", "--bind", f"{python_prefix}:{python_prefix}:ro", "--pwd", workdir]
    if network == "none":
        cmd += ["--net", "--network", "none"]
    cmd += [f"--env={k}={v}" for k, v in sorted(env.items())]
    return cmd + [image]


def container_available(runtime: str, image: str, network: str = "none", *, runner=subprocess.run) -> bool:
    """Probe once whether the container runs here (with the requested network mode)."""
    key = (runtime, image, network)
    if key not in _CONTAINER_OK:
        exe = shutil.which(runtime)
        ok = False
        if exe and Path(image).exists():
            probe = [exe, "exec", "--containall", "--no-home"]
            if network == "none":
                probe += ["--net", "--network", "none"]
            try:
                ok = runner(probe + [image, "true"], capture_output=True, timeout=60).returncode == 0
            except Exception:
                ok = False
        _CONTAINER_OK[key] = ok
    return _CONTAINER_OK[key]


def count_tests(test_src: str) -> int:
    """Number of module-level ``test_*`` functions, counted by the parent (not the harness)."""
    tree = ast.parse(test_src)
    return sum(
        1
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalSubprocessSandbox:
    """Runs one request in a fresh subprocess with resource limits and a scrubbed environment."""

    def __init__(
        self,
        *,
        hidden_store: HiddenTestStore | None = None,
        cpu_s: int = 10,
        mem_mb: int = 2048,
        fsize_mb: int = 16,
        nofile: int = 256,
        nproc: int | None = None,
        python: str = sys.executable,
        network_isolation: str = "auto",  # "auto" | "unshare" | "none"
        container_image: str | None = None,   # Apptainer .sif → filesystem isolation (5b)
        container_runtime: str = "apptainer",
        container_network: str = "none",      # "none" (--net --network none) | "host"
    ):
        self.hidden_store = hidden_store
        self.cpu_s, self.mem_mb, self.fsize_mb, self.nofile, self.nproc = (
            cpu_s,
            mem_mb,
            fsize_mb,
            nofile,
            nproc,
        )
        self.python = python
        self.network_isolation = network_isolation
        self.container_image = container_image
        self.container_runtime = container_runtime
        self.container_network = container_network

    # -- helpers ------------------------------------------------------------------------------
    def _limits_json(self) -> str:
        return json.dumps(
            {
                "cpu_s": int(self.cpu_s),
                "mem": int(self.mem_mb) * 1024 * 1024,
                "fsize": int(self.fsize_mb) * 1024 * 1024,
                "nofile": int(self.nofile),
                "nproc": self.nproc,
            }
        )

    def _resolve_tests(self, req: SandboxRequest) -> str:
        if req.tests is not None:
            return req.tests
        if req.hidden_tests_id is None:
            raise ValueError("request has neither inline tests nor hidden_tests_id")
        if self.hidden_store is None:
            raise ValueError("hidden tests requested but no HiddenTestStore configured")
        return self.hidden_store.get(req.hidden_tests_id)

    # -- main ---------------------------------------------------------------------------------
    def run(self, req: SandboxRequest) -> SandboxResult:
        t0 = time.perf_counter()
        if req.language != "python":
            return SandboxResult(status="error", detail=f"unsupported language {req.language}")
        try:
            test_src = self._resolve_tests(req)
            total = count_tests(test_src)
        except Exception as e:
            return SandboxResult(status="infra_error", detail=f"tests unavailable: {type(e).__name__}: {e}")
        if total == 0:
            return SandboxResult(status="infra_error", detail="test file defines no test_* functions")

        root = Path(tempfile.mkdtemp(prefix="dpt_sbx_"))
        work, tdir, hdir = root / "work", root / f"t_{secrets.token_hex(6)}", root / "h"
        for d in (work, tdir, hdir):
            d.mkdir()
        sol = work / "solution.py"
        sol.write_text(req.code, encoding="utf-8")
        test_file = tdir / "tests.py"
        test_file.write_text(test_src, encoding="utf-8")
        harness = hdir / "harness.py"
        harness.write_text(_HARNESS, encoding="utf-8")
        os.chmod(test_file, 0o400)
        os.chmod(tdir, 0o500)
        h_before = _sha256(test_file)
        stdout_path = hdir / "stdout.txt"
        nonce = secrets.token_hex(16)

        cmd = [self.python, "-I", str(harness), str(sol), str(test_file), self._limits_json()]
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            # One BLAS/OpenMP thread. OpenBLAS otherwise reserves buffers for every core at import
            # time, which exceeds RLIMIT_AS on a many-core node: `import numpy` then fails and every
            # numpy solution scores 0 (found by dpt.rewards.sandbox_check on a 64-core machine).
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        if self.container_image:
            # Fail closed: a configured but broken container must never fall back to running the
            # model's code directly on the host.
            if not container_available(self.container_runtime, self.container_image, self.container_network):
                shutil.rmtree(root, ignore_errors=True)
                return SandboxResult(status="infra_error",
                                     detail=f"sandbox container unavailable: {self.container_runtime} "
                                            f"{self.container_image} (network={self.container_network})")
            prefix = str(Path(self.python).resolve().parents[1])
            # Absolute path: the child's scrubbed PATH (/usr/bin:/bin) would not find a module-installed runtime.
            runtime = shutil.which(self.container_runtime) or self.container_runtime
            cmd = container_command(runtime, self.container_image, root=str(root),
                                    workdir=str(work), python_prefix=prefix, env=env,
                                    network=self.container_network) + cmd
        elif self.network_isolation == "unshare" or (self.network_isolation == "auto" and _unshare_available()):
            if _unshare_available():
                cmd = ["unshare", "-rn", *cmd]
        status, detail = "ok", ""
        try:
            with open(stdout_path, "wb") as out:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=subprocess.DEVNULL,
                    cwd=str(work),
                    env=env,
                    start_new_session=True,  # own process group, so a timeout kills every child
                )
                try:
                    proc.communicate(input=(nonce + "\n").encode(), timeout=req.timeout_s)
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5)
            passed, events = 0, []
            for raw in stdout_path.read_bytes().decode("utf-8", "replace").splitlines():
                if raw.startswith(nonce):
                    try:
                        ev = json.loads(raw[len(nonce) :])
                    except json.JSONDecodeError:
                        continue
                    events.append(ev)
                    if ev.get("event") == "result" and ev.get("ok") is True:
                        passed += 1
            if any(e.get("event") in ("import_error", "test_import_error") for e in events):
                status = "error"
                detail = next(e.get("error", "") for e in events if "error" in e)
            tamper = []
            os.chmod(tdir, 0o700)
            if not test_file.exists() or _sha256(test_file) != h_before:
                tamper.append("runtime_test_file_modified")
            return SandboxResult(
                passed=min(passed, total),
                total=total,
                status=status,
                tamper=tamper,
                detail=detail,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as e:  # infrastructure failure → caller fails closed
            return SandboxResult(status="infra_error", detail=f"{type(e).__name__}: {e}")
        finally:
            try:
                os.chmod(tdir, 0o700)
            except OSError:
                pass
            shutil.rmtree(root, ignore_errors=True)


# =============================================================================================
# Remote client (CPU sandbox cluster)
# =============================================================================================
class RemoteSandboxClient:
    """HTTP client for ``dpt.rewards.sandbox_server`` workers (``POST /run``)."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        retries: int = 2,
        overhead_s: float = 10.0,
        client: Any | None = None,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.token = token
        self.retries = retries
        self.overhead_s = overhead_s
        self._client = client or httpx.Client()

    def run(self, req: SandboxRequest) -> SandboxResult:
        import httpx

        headers = {"X-Sandbox-Token": self.token} if self.token else {}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(
                    f"{self.base_url}/run",
                    json=req.to_json(),
                    headers=headers,
                    timeout=req.timeout_s + self.overhead_s,
                )
                r.raise_for_status()
                return SandboxResult.from_json(r.json())
            except (httpx.HTTPError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2.0, 0.2 * (attempt + 1)))
        return SandboxResult(status="infra_error", detail=f"remote sandbox failed: {last}")


class SandboxPool:
    """Round-robin over several sandbox clients (one per sandbox-cluster worker)."""

    def __init__(self, clients: list[SandboxClient]):
        if not clients:
            raise ValueError("SandboxPool needs at least one client")
        self.clients = clients
        self._i = 0
        self._lock = threading.Lock()

    def run(self, req: SandboxRequest) -> SandboxResult:
        with self._lock:
            c = self.clients[self._i % len(self.clients)]
            self._i += 1
        return c.run(req)


# =============================================================================================
# The T2 grader
# =============================================================================================
@dataclass
class CodeConfig:
    timeout_s: float = 10.0
    run_visible_tests: bool = True
    zero_on: tuple[str, ...] = DEFAULT_ZERO_ON
    test_path_patterns: tuple[str, ...] = (
        r"(^|/)tests?/",
        r"(^|/)test_[^/]*\.py$",
        r"_test\.py$",
        r"conftest\.py$",
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CodeConfig":
        d = dict(d or {})
        out = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        out.zero_on = tuple(out.zero_on)
        out.test_path_patterns = tuple(out.test_path_patterns)
        return out


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(answer_text: str) -> str | None:
    """Last ```python (or ```py) block; else the last untagged block; else ``None``."""
    blocks = _FENCE.findall(answer_text)
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
    if py:
        return py[-1].strip("\n")
    untagged = [b for lang, b in blocks if not lang]
    return untagged[-1].strip("\n") if untagged else None


@fail_closed(task=T2_CODE)
def grade_code(
    answer_text: str,
    ground_truth: dict[str, Any],
    *,
    sandbox: SandboxClient,
    cfg: CodeConfig | None = None,
    extra_info: dict[str, Any] | None = None,
) -> RewardResult:
    """Grade a T2 response. ``answer_text`` is the post-``</think>`` text containing the final code."""
    cfg = cfg or CodeConfig()
    extra_info = extra_info or {}
    task = T2_CODE

    code = extract_code(answer_text)
    if not code:
        return zero(task, "no_code")

    flags = code_flags(code)
    zero_hits = [f for f in flags if f in cfg.zero_on]
    if zero_hits:
        return zero(task, f"tamper:{zero_hits[0]}", flags=flags)

    # Agentic trajectories: files the model touched in its workspace (provided by the rollout env).
    for path in extra_info.get("modified_files", []) or []:
        if any(re.search(p, str(path)) for p in cfg.test_path_patterns):
            return zero(task, "tamper:modify_tests", flags=sorted(set(flags) | {"test_tamper"}))

    hidden_id = ground_truth.get("hidden_tests_id")
    if not hidden_id:
        return zero(task, "bad_ground_truth")
    hidden = sandbox.run(
        SandboxRequest(
            code=code,
            hidden_tests_id=hidden_id,
            entry_point=ground_truth.get("entry_point", ""),
            timeout_s=cfg.timeout_s,
            kind="hidden",
        )
    )
    if hidden.status == "infra_error":
        return zero(task, "sandbox_error", flags=flags, components={"t2_infra_error": 1.0})
    if hidden.tamper:
        return zero(task, f"tamper:{hidden.tamper[0]}", flags=sorted(set(flags) | {"test_tamper"}))

    components: dict[str, float] = {
        "t2_hidden_pass": hidden.pass_rate,
        "t2_timeout": 1.0 if hidden.status == "timeout" else 0.0,  # counted separately (report §单步时间预算)
        "t2_infra_error": 0.0,
    }
    if cfg.run_visible_tests and ground_truth.get("visible_tests"):
        vis = sandbox.run(
            SandboxRequest(
                code=code, tests=ground_truth["visible_tests"], timeout_s=cfg.timeout_s, kind="visible"
            )
        )
        if vis.status != "infra_error":
            components["t2_visible_pass"] = vis.pass_rate
            components["t2_visible_hidden_gap"] = vis.pass_rate - hidden.pass_rate

    score = hidden.pass_rate
    return RewardResult(
        score=score,
        task=task,
        reason="ok" if hidden.status in ("ok", "timeout") else hidden.status,
        correct=score >= 1.0,
        components=components,
        flags=flags,
    )
