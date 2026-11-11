"""Launch stage-1 SFT (verl ``sft_trainer``, Megatron engine) from ``configs/sft/*.yaml``.

    python -m dpt.launch.sft --config configs/sft/sft_27b.yaml [--dry-run] [key=value ...]

Multi-node: run the same command on every node with ``NNODES``, ``NODE_RANK``,
``MASTER_ADDR`` and ``MASTER_PORT`` exported (``scripts/04_sft.sh`` does this).
The command is ``torchrun`` + ``-m verl.trainer.sft_trainer`` + rendered overrides.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dpt.common.config import to_container
from dpt.launch.common import as_list, compose_stage, execute, prepare_run, python_exe


def build(config: str, overrides: list[str], *, require_manifest: bool = True):
    cfg = compose_stage(config, overrides)
    hw = cfg.hardware
    nnodes = int(os.environ.get("NNODES", hw.nnodes))
    argv = [
        python_exe(), "-m", "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={hw.gpus_per_node}",
        f"--node_rank={os.environ.get('NODE_RANK', 0)}",
        f"--master_addr={os.environ.get('MASTER_ADDR', '127.0.0.1')}",
        f"--master_port={os.environ.get('MASTER_PORT', 29500)}",
        "-m", "verl.trainer.sft_trainer",
    ]
    sft = to_container(cfg)["sft"]
    data = as_list(sft["train_files"]) + as_list(sft["val_files"])
    env = {
        # Incident 4: fragmentation after repeated large allocations → expandable segments.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",   # required by Megatron TP/SP comm-compute overlap
        # Imported at `import verl` in every rank: fixes the inverted Qwen3.5 attention mask that
        # A100 training forwards would otherwise apply (dpt.verl_ext.patches).
        "VERL_USE_EXTERNAL_MODULES": "dpt.verl_ext.patches",
    }
    return prepare_run(name=cfg.sft.run_name, cfg=cfg, section="verl_sft", base_argv=argv,
                       run_dir=Path(cfg.sft.output_dir), data_files=data, env=env,
                       require_manifest=require_manifest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="sft/sft_27b.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    p.add_argument("--allow-missing-manifest", action="store_true", help="smoke tests only")
    p.add_argument("--no-hydra-check", action="store_true", help="skip composing verl's config before launch")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. sft.lr_peak=5e-6")
    a = p.parse_args()
    run = build(a.config, a.overrides, require_manifest=not a.allow_missing_manifest)
    raise SystemExit(execute(run, dry_run=a.dry_run or bool(os.environ.get("DPT_DRY_RUN")),
                             check_module=None if a.no_hydra_check else "verl.trainer.sft_trainer"))


if __name__ == "__main__":
    main()
