"""The isolation check must FAIL for an unisolated sandbox (this workstation has no container)."""

from __future__ import annotations

from pathlib import Path

from dpt.rewards.code_sandbox import LocalSubprocessSandbox
from dpt.rewards.sandbox_check import run_checks


def test_plain_subprocess_sandbox_is_reported_as_not_isolated(tmp_path):
    repo_readme = Path(__file__).resolve().parents[2] / "README.md"
    paths = {"home": str(Path.home()), "repo_readme": str(repo_readme), "work": str(tmp_path),
             "scratch": str(tmp_path), "eval_secure": ""}
    rep = run_checks(LocalSubprocessSandbox(network_isolation="none"), paths=paths,
                     network_required_blocked=False)
    assert not rep["isolated"]
    assert {"home_visible", "repo_readable", "work_writable", "scratch_visible"} <= set(rep["failed"])
    assert rep["probes"]["numpy_import"]["ok"]                       # positive control still works
    assert not list(tmp_path.glob("dpt_sandbox_probe_*"))            # the probe file is cleaned up
