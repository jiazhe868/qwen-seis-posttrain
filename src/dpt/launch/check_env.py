"""Environment check, step 00 of the run book. Run it on every machine before anything else.

    python -m dpt.launch.check_env [--profile cluster_2x8xa100_80g] [--require-train]

It reports, and with ``--require-train`` enforces:

* GPU model, count and memory against the hardware profile (A100 → BF16 only, no FP8)
* NVIDIA driver CUDA version vs the CUDA version torch was built with. The
  workstation's driver 560 supports CUDA ≤ 12.6, while current vLLM/verl wheels
  are built for ≥ 12.8 (use cuda-compat or the cluster image)
* versions of torch / verl / vllm / megatron-core / megatron-bridge / transfer_queue /
  flash-linear-attention / transformers / pyarrow, against the versions this repo was
  verified with (docs/verl_integration.md)
* NVLink topology (``nvidia-smi topo -m``) and free disk under the data/checkpoint roots
* reachability of the judge, sandbox and citation endpoints used by the rewards

Nothing here needs a GPU to *run*; on a CPU-only box it simply reports what is missing.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from typing import Any

from dpt.common.config import load_config, to_container

VERIFIED = {               # versions the adapters were written and checked against
    "verl": "0.9.1",
    "vllm": "0.24.0",      # verl 0.9.1 wheel pin
    "torch": "2.11.0",     # verl 0.9.1 wheel pin
    "transformers": "5.9.0",
    "megatron.bridge": "0.6.2",  # not pinned by verl; checked on the workstation smoke run (2026-10-07)
}
OPTIONAL = ["megatron.core", "transfer_queue", "fla", "pyarrow", "ray", "flash_attn", "sympy", "safetensors"]


# import name → distribution name, for modules without __version__ or that fail to import
_DIST = {"megatron.bridge": "megatron-bridge", "megatron.core": "megatron-core", "transfer_queue": "TransferQueue",
         "fla": "flash-linear-attention", "flash_attn": "flash-attn"}
_IMPORT_ERRORS: dict[str, str] = {}


def _version(mod: str) -> str | None:
    """Installed version of ``mod``, or None if not installed.

    An installed package that fails to import (e.g. Transformer Engine finding the system cuDNN
    before the env's) still reports its version, and the error goes to ``_IMPORT_ERRORS``. Reporting
    it as "missing" would hide the real cause.
    """
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None)
    except Exception as e:
        _IMPORT_ERRORS[mod] = f"{type(e).__name__}: {str(e)[:200]}"
        m, v = None, None
    if v is None:
        try:
            v = importlib.metadata.version(_DIST.get(mod, mod))
        except importlib.metadata.PackageNotFoundError:
            v = "installed" if m is not None else None
    if v is None:
        _IMPORT_ERRORS.pop(mod, None)         # simply not installed
    return None if v is None else str(v).split("+")[0]


def _nvidia_smi(args: list[str]) -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=20)
    except Exception:
        return ""


def gather(profile: str | None) -> dict[str, Any]:
    cfg = to_container(load_config(profile=profile))
    hw = cfg["hardware"]
    info: dict[str, Any] = {"profile": hw["profile"], "expected": {
        "gpus_per_node": hw["gpus_per_node"], "gpu_mem_gb": hw["gpu_mem_gb"]}}
    q = _nvidia_smi(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = [dict(zip(("name", "mem_mib", "driver"), [x.strip() for x in line.split(",")]))
            for line in q.strip().splitlines() if line.strip()]
    info["gpus"] = gpus
    head = _nvidia_smi([])
    info["driver_cuda"] = next((ln.split("CUDA Version:")[1].split()[0] for ln in head.splitlines()
                                if "CUDA Version:" in ln), None)
    info["topology"] = _nvidia_smi(["topo", "-m"]).strip().splitlines()[:12]
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info["torch_cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["torch_cuda_available"] = False
    info["packages"] = {m: _version(m) for m in [*VERIFIED, *OPTIONAL]}
    info["import_errors"] = dict(_IMPORT_ERRORS)
    info["verified_versions"] = VERIFIED
    roots = cfg["project"]["paths"]
    info["disk_free_gb"] = {k: round(shutil.disk_usage(_existing(v)).free / 1e9, 1)
                            for k, v in roots.items() if k in ("data_root", "ckpt_root")}
    info["cuda_compat_on_path"] = any("cuda-compat" in p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    info["endpoints"] = {
        "judge": os.environ.get("DPT_JUDGE_URL"), "sandbox": os.environ.get("DPT_SANDBOX_URL"),
    }
    return info


def _existing(path: str) -> str:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return p


def problems(info: dict[str, Any], require_train: bool) -> list[str]:
    out = []
    exp = info["expected"]
    if len(info["gpus"]) < exp["gpus_per_node"]:
        out.append(f"found {len(info['gpus'])} GPUs, profile expects {exp['gpus_per_node']}")
    for g in info["gpus"]:
        if abs(float(g["mem_mib"]) / 1024 - exp["gpu_mem_gb"]) > 4:
            out.append(f"GPU memory {g['mem_mib']} MiB does not match the profile's {exp['gpu_mem_gb']} GB")
            break
    if info["gpus"] and not info.get("torch_cuda_available"):
        out.append(f"torch cannot use the GPUs (torch built for CUDA {info.get('torch_cuda_build')}, "
                   f"driver supports {info.get('driver_cuda')}). On the workstation use the tested envs: "
                   "`bash scripts/00_setup_env.sh gpu-local` (torch 2.11 CUDA 12.6 + GDN kernels) and "
                   "`bash scripts/00_setup_env.sh vllm-local` (vLLM 0.24 via cuda-compat 12.9 forward compatibility)")
    if require_train:
        for m, v in VERIFIED.items():
            have = info["packages"].get(m)
            if have is None:
                out.append(f"missing {m} (verified with {v})")
            elif m in info.get("import_errors", {}):
                out.append(f"{m} {have} is installed but does not import: {info['import_errors'][m]}")
            elif have != v and m == "verl":
                out.append(f"verl {have} != {v}: re-verify dpt.verl_ext against the new source before training")
        for m in ("transfer_queue", "fla", "pyarrow"):
            if info["packages"].get(m) is None:
                out.append(f"missing {m}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--require-train", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    info = gather(a.profile)
    issues = problems(info, a.require_train)
    print(json.dumps(info, indent=2, default=str))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**info, "issues": issues}, f, indent=2, default=str)
    print("\n".join(["", "ISSUES:" if issues else "OK: no issues", *[f"  - {x}" for x in issues]]))
    raise SystemExit(1 if issues and a.require_train else 0)


if __name__ == "__main__":
    main()
