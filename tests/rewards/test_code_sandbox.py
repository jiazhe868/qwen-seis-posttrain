"""T2 sandbox and grader: hidden-test pass rate, tamper zeroing, unforgeable results."""

from __future__ import annotations

import pytest

from dpt.rewards.code_sandbox import (
    CodeConfig,
    HiddenTestStore,
    LocalSubprocessSandbox,
    SandboxRequest,
    SandboxResult,
    count_tests,
    extract_code,
    grade_code,
)

TESTS = (
    "from solution import travel_time\n"
    "def test_basic():\n    assert abs(travel_time(120, 6) - 20) < 1e-9\n"
    "def test_unit():\n    assert travel_time(6, 6) == 1\n"
    "def test_zero_distance():\n    assert travel_time(0, 6) == 0\n"
    "def test_bad_velocity():\n"
    "    try:\n        travel_time(1, 0)\n    except ValueError:\n        return\n    assert False\n"
)
GOOD = "def travel_time(d, v):\n    if v <= 0:\n        raise ValueError('v must be positive')\n    return d / v\n"


@pytest.fixture()
def store(tmp_path):
    (tmp_path / "tt_001.py").write_text(TESTS)
    return HiddenTestStore(tmp_path)


@pytest.fixture()
def sandbox(store):
    return LocalSubprocessSandbox(hidden_store=store, cpu_s=5, mem_mb=1024)


def answer(code: str) -> str:
    return f"Here is the solution.\n```python\n{code}```\n"


def test_count_tests_from_source():
    assert count_tests(TESTS) == 4


def test_extract_code_takes_last_python_block():
    assert extract_code("```python\na=1\n```\ntext\n```python\nb=2\n```") == "b=2"
    assert extract_code("no code") is None


def test_full_pass(sandbox):
    r = sandbox.run(SandboxRequest(code=GOOD, hidden_tests_id="tt_001", timeout_s=10))
    assert (r.passed, r.total, r.status) == (4, 4, "ok")


def test_partial_pass(sandbox):
    r = sandbox.run(
        SandboxRequest(code="def travel_time(d, v):\n    return d / v\n", hidden_tests_id="tt_001")
    )
    assert r.passed == 3 and r.total == 4


def test_exit_at_import_scores_zero(sandbox):
    r = sandbox.run(SandboxRequest(code="import sys\nsys.exit(0)\n", hidden_tests_id="tt_001"))
    assert r.passed == 0 and r.total == 4 and r.status == "error"


def test_forged_result_lines_do_not_count(sandbox):
    """The solution cannot fake passes: result lines carry a per-run nonce it never sees."""
    forge = (
        'import json\nfor i in range(10):\n    print(json.dumps({"event": "result", "ok": True}))\n'
        "def travel_time(d, v):\n    return -1\n"
    )
    r = sandbox.run(SandboxRequest(code=forge, hidden_tests_id="tt_001"))
    assert r.passed == 0


@pytest.mark.slow
def test_timeout_is_reported(sandbox):
    r = sandbox.run(SandboxRequest(code="while True:\n    pass\n", hidden_tests_id="tt_001", timeout_s=1.5))
    assert r.status == "timeout" and r.passed == 0


def test_unknown_hidden_id_is_infra_error(sandbox):
    assert sandbox.run(SandboxRequest(code=GOOD, hidden_tests_id="nope")).status == "infra_error"
    assert sandbox.run(SandboxRequest(code=GOOD, hidden_tests_id="../etc/passwd")).status == "infra_error"


def test_grade_code_end_to_end(sandbox):
    gt = {"hidden_tests_id": "tt_001", "visible_tests": TESTS, "entry_point": "travel_time"}
    r = grade_code(answer(GOOD), gt, sandbox=sandbox)
    assert r.score == 1.0 and r.correct and r.components["t2_visible_hidden_gap"] == 0.0


@pytest.mark.parametrize(
    "code,reason",
    [
        ("import sys\ndef travel_time(d, v):\n    sys.exit(0)\n", "tamper:exit_call"),
        (
            "def travel_time(d, v):\n    try:\n        return d / v\n    except:\n        pass\n",
            "tamper:blanket_except",
        ),
        (
            "import os\ndef travel_time(d, v):\n    if os.getenv('PYTEST_CURRENT_TEST'):\n        return 20\n    return 0\n",
            "tamper:env_probe",
        ),
    ],
)
def test_tamper_zeroes_before_running(code, reason):
    class Exploding:
        def run(self, req):  # must never be called: static checks short-circuit
            raise AssertionError("sandbox should not run tampering code")

    r = grade_code(answer(code), {"hidden_tests_id": "x"}, sandbox=Exploding())
    assert r.score == 0.0 and r.reason == reason


def test_workspace_diff_touching_tests_is_tamper():
    class Never:
        def run(self, req):
            raise AssertionError

    r = grade_code(
        answer(GOOD),
        {"hidden_tests_id": "x"},
        sandbox=Never(),
        extra_info={"modified_files": ["src/a.py", "tests/test_a.py"]},
    )
    assert r.reason == "tamper:modify_tests"


def test_visible_hidden_gap_reported():
    """Monitoring signal for the step-205 hack (hidden tests use a different input distribution)."""

    class Fake:
        def run(self, req):
            return SandboxResult(passed=4 if req.kind == "visible" else 1, total=4)

    r = grade_code(
        answer(GOOD),
        {"hidden_tests_id": "x", "visible_tests": "def test_a():\n    pass\n"},
        sandbox=Fake(),
        cfg=CodeConfig(run_visible_tests=True),
    )
    assert r.score == 0.25 and r.components["t2_visible_hidden_gap"] == pytest.approx(0.75)


def test_infra_error_fails_closed():
    class Down:
        def run(self, req):
            return SandboxResult(status="infra_error", detail="down")

    r = grade_code(answer(GOOD), {"hidden_tests_id": "x"}, sandbox=Down())
    assert r.score == 0.0 and r.reason == "sandbox_error"


def test_no_code_block():
    assert grade_code("I would write a function.", {"hidden_tests_id": "x"}, sandbox=None).reason == "no_code"


# ---- filesystem isolation via a container (5b; required on the cluster) ---------------------------
def _fake_runtime(tmp_path, log):
    """A stand-in for `apptainer`: records its arguments, then runs the command after the image."""
    exe = tmp_path / "fake-apptainer"
    exe.write_text("#!/bin/bash\n"
                   f"printf '%s\\n' \"$*\" >> {log}\n"
                   "while [[ $# -gt 0 ]]; do a=$1; shift; [[ $a == *.sif ]] && break; done\n"
                   "exec \"$@\"\n")
    exe.chmod(0o755)
    image = tmp_path / "sandbox.sif"
    image.write_text("")
    return str(exe), str(image)


def test_container_command_binds_only_the_sandbox_and_python():
    from dpt.rewards.code_sandbox import container_command

    cmd = container_command("/opt/apptainer", "/w/sbx.sif", root="/tmp/dpt_sbx_x", workdir="/tmp/dpt_sbx_x/work",
                            python_prefix="/w/envs/sbx", env={"HOME": "/tmp/dpt_sbx_x/work"})
    assert cmd[:5] == ["/opt/apptainer", "exec", "--containall", "--no-home", "--cleanenv"]
    binds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    assert binds == ["/tmp/dpt_sbx_x:/tmp/dpt_sbx_x", "/w/envs/sbx:/w/envs/sbx:ro"]
    assert ["--net", "--network", "none"] == cmd[cmd.index("--net"):cmd.index("--net") + 3]
    assert "--env=HOME=/tmp/dpt_sbx_x/work" in cmd and cmd[-1] == "/w/sbx.sif"


def test_container_path_runs_tests_through_the_runtime(store, tmp_path):
    import dpt.rewards.code_sandbox as cs

    log = tmp_path / "calls.txt"
    exe, image = _fake_runtime(tmp_path, log)
    cs._CONTAINER_OK.clear()
    sb = LocalSubprocessSandbox(hidden_store=store, cpu_s=5, mem_mb=1024, container_image=image,
                                container_runtime=exe)
    r = sb.run(SandboxRequest(code=GOOD, hidden_tests_id="tt_001", timeout_s=10))
    assert (r.passed, r.total, r.status) == (4, 4, "ok")
    calls = log.read_text().splitlines()
    assert calls[0].endswith(f"{image} true")                       # availability probe
    assert "--containall" in calls[1] and "harness.py" in calls[1]  # the real run went through it


def test_broken_container_fails_closed_never_runs_on_host(store, tmp_path):
    import dpt.rewards.code_sandbox as cs

    marker = tmp_path / "ran_on_host"
    code = f"open({str(marker)!r}, 'w').write('x')\n" + GOOD
    cs._CONTAINER_OK.clear()
    sb = LocalSubprocessSandbox(hidden_store=store, container_image=str(tmp_path / "missing.sif"),
                                container_runtime="/bin/false")
    r = sb.run(SandboxRequest(code=code, hidden_tests_id="tt_001", timeout_s=10))
    assert r.status == "infra_error" and "container unavailable" in r.detail
    assert r.pass_rate == 0.0 and not marker.exists()


def test_numpy_solutions_import_under_the_memory_limit(tmp_path):
    """OpenBLAS reserves per-core buffers at import; under RLIMIT_AS that broke `import numpy` (score 0)."""
    pytest.importorskip("numpy")
    code = "import numpy as np\ndef total(xs):\n    return float(np.asarray(xs).sum())\n"
    test = "from solution import total\ndef test_total():\n    assert total([1, 2, 3]) == 6.0\n"
    r = LocalSubprocessSandbox(network_isolation="none", mem_mb=2048).run(
        SandboxRequest(code=code, tests=test, timeout_s=30))
    assert (r.status, r.passed, r.total) == ("ok", 1, 1), r.detail
