"""Verify that the T2 sandbox really isolates model-written code, on the machine it will run on.

    python -m dpt.rewards.sandbox_check [--container-image $DPT_SANDBOX_SIF] [--python $DPT_SANDBOX_PYTHON]

RL executes tens of thousands of programs written by a policy that is being optimized for reward.
On a shared-cluster account (the cluster) they run next to the project's data, checkpoints and eval
sets. Each probe below is a "solution" that attempts one forbidden action and returns True if it
succeeded; the hidden test asserts that it did not. The check passes only if every probe is
blocked and the positive controls (tests run; numpy imports) work.

=================  ==================================================  =====================
probe              attempts                                            required
=================  ==================================================  =====================
home_visible       see the real $HOME                                  blocked
repo_readable      read this repo's README.md                          blocked
work_writable      create a file under $WORK                           blocked (and absent after)
scratch_visible    list $SCRATCH                                       blocked
eval_visible       see the eval-set directory                          blocked
network            open a TCP connection to pypi.org:443               blocked (warning only
                                                                        with container_network=host)
numpy_import       import numpy (positive control)                     works
=================  ==================================================  =====================

Exit code 0 = isolated. Run it via scripts/slurm/check_node.slurm on a compute node after building
the sandbox image (scripts/slurm/README.md §sandbox); the RL launcher refuses a cluster launch without
a container image, but only this check shows that the image actually isolates.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from dpt.rewards.code_sandbox import LocalSubprocessSandbox, SandboxRequest

_TEST_BLOCKED = "from solution import probe\ndef test_blocked():\n    assert probe() is False\n"
_TEST_WORKS = "from solution import probe\ndef test_works():\n    assert probe() is True\n"


def _probes(paths: dict[str, str], marker: str) -> dict[str, tuple[str, str]]:
    """name → (solution code, test source)."""
    def exists(p: str) -> str:
        return ("import os\n"
                f"def probe():\n    try:\n        return os.path.exists({p!r}) and len(os.listdir({p!r})) >= 0\n"
                "    except Exception:\n        return False\n")

    out = {
        "home_visible": (exists(paths["home"]), _TEST_BLOCKED),
        "repo_readable": (
            f"def probe():\n    try:\n        return len(open({paths['repo_readme']!r}).read()) > 0\n"
            "    except Exception:\n        return False\n", _TEST_BLOCKED),
        "work_writable": (
            f"def probe():\n    try:\n        open({marker!r}, 'w').write('x')\n        return True\n"
            "    except Exception:\n        return False\n", _TEST_BLOCKED),
        "network": (
            "import socket\n"
            "def probe():\n    try:\n        socket.create_connection(('pypi.org', 443), timeout=5).close()\n"
            "        return True\n    except Exception:\n        return False\n", _TEST_BLOCKED),
        "numpy_import": (
            "def probe():\n    try:\n        import numpy\n        return float(numpy.ones(3).sum()) == 3.0\n"
            "    except Exception:\n        return False\n", _TEST_WORKS),
    }
    if paths.get("scratch"):
        out["scratch_visible"] = (exists(paths["scratch"]), _TEST_BLOCKED)
    if paths.get("eval_secure"):
        out["eval_visible"] = (exists(paths["eval_secure"]), _TEST_BLOCKED)
    return out


def run_checks(sandbox: LocalSubprocessSandbox, *, paths: dict[str, str], network_required_blocked: bool = True,
               timeout_s: float = 30.0) -> dict:
    marker = str(Path(paths["work"]) / f"dpt_sandbox_probe_{secrets.token_hex(6)}")
    results: dict[str, dict] = {}
    for name, (code, test) in _probes(paths, marker).items():
        r = sandbox.run(SandboxRequest(code=code, tests=test, timeout_s=timeout_s))
        ok = r.status == "ok" and r.passed == r.total == 1
        results[name] = {"ok": ok, "status": r.status, "detail": r.detail[:200]}
    if Path(marker).exists():                         # the file must not exist outside, whatever the probe said
        results["work_writable"]["ok"] = False
        results["work_writable"]["detail"] = f"file created on the host: {marker}"
        Path(marker).unlink()
    required = [k for k in results if k != "network" or network_required_blocked]
    failed = [k for k in required if not results[k]["ok"]]
    warnings = [] if network_required_blocked or results["network"]["ok"] else [
        "network reachable from the sandbox (container_network=host): network imports are only flagged"]
    return {"isolated": not failed, "failed": failed, "warnings": warnings, "probes": results}


def default_paths(env=os.environ) -> dict[str, str]:
    from dpt.common.config import load_config

    repo = Path(__file__).resolve().parents[3]
    secure = str(load_config().project.paths.eval_secure_root)
    return {"home": os.path.expanduser("~"), "repo_readme": str(repo / "README.md"),
            "work": env.get("WORK") or str(repo), "scratch": env.get("SCRATCH", ""),
            "eval_secure": secure if os.path.exists(secure) else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--container-image", default=os.environ.get("DPT_SANDBOX_SIF"))
    ap.add_argument("--container-network", default="none", choices=["none", "host"])
    ap.add_argument("--python", default=os.environ.get("DPT_SANDBOX_PYTHON") or sys.executable)
    ap.add_argument("--json", default=None, help="also write the report here")
    a = ap.parse_args()
    sb = LocalSubprocessSandbox(python=a.python, container_image=a.container_image,
                                container_network=a.container_network, cpu_s=10, mem_mb=2048)
    rep = run_checks(sb, paths=default_paths(), network_required_blocked=a.container_network == "none")
    rep["config"] = {"container_image": a.container_image, "container_network": a.container_network,
                     "python": a.python}
    text = json.dumps(rep, indent=2)
    print(text)
    if a.json:
        Path(a.json).write_text(text)
    raise SystemExit(0 if rep["isolated"] else 1)


if __name__ == "__main__":
    main()
