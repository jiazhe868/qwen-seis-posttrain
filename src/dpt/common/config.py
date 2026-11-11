"""Layered YAML configuration.

Configs are composed in a fixed order, later layers overriding earlier ones:

    configs/project.yaml                 # single source of truth: model ids, paths, gates, seeds
      + configs/model/<line>.yaml        # qwen38_27b (the report, default) | qwen35_9b
      + configs/hardware/<profile>.yaml  # local_4xa100_40g | cluster_2x8xa100_80g | slurm_3xa100_40g
      + stage config (e.g. configs/rl/rl_27b.yaml)
      + optional ablation overlay (configs/rl/ablations/A08_lr.yaml)
      + CLI dotlist overrides  (e.g. rl.lr=5e-7)

The model line and the hardware profile are chosen by environment variables
(``DPT_MODEL_LINE``, ``DPT_HW_PROFILE``), so every script and launcher reads the
same base model, architecture facts and gates without each command repeating them.

``${...}`` interpolation (OmegaConf) lets a stage config reference project
values (``${project.paths.data_root}``) so a path or threshold is defined once.

The resolved config is hashed (:func:`config_hash`) and written next to every
output, which is how a result is tied to the exact settings that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from dpt.common.hashing import sha256_obj

# Small arithmetic resolvers so derived values are computed from their sources instead of
# being copied by hand, e.g. ppo_mini_batch_size = ${div:${rl.prompts_per_step},${rl.num_updates_per_step}}.
for _name, _fn in {
    "div": lambda a, b: int(a) // int(b),
    "mul": lambda a, b: a * b,
    "sub": lambda a, b: a - b,
    "add": lambda a, b: a + b,
}.items():
    if not OmegaConf.has_resolver(_name):
        OmegaConf.register_new_resolver(_name, _fn)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"

# Hardware profile is chosen by env var so the *same* commands run on the
# workstation and on the cluster (README §3 "two machines, one code path").
PROFILE_ENV = "DPT_HW_PROFILE"
DEFAULT_PROFILE = "local_4xa100_40g"
# Model line: which base model the project trains (base id, architecture facts, eval protocol,
# gates). The default is the report's Qwen3.8-27B; qwen35_9b is the 9B main line.
MODEL_LINE_ENV = "DPT_MODEL_LINE"
DEFAULT_MODEL_LINE = "qwen38_27b"


def model_line_path(line: str) -> Path:
    p = CONFIG_ROOT / "model" / f"{line}.yaml"
    if not p.exists():
        known = sorted(q.stem for q in (CONFIG_ROOT / "model").glob("*.yaml"))
        raise FileNotFoundError(f"unknown model line {line!r} ({MODEL_LINE_ENV}); known: {known}")
    return p


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return p


def load_config(
    *stage_configs: str | os.PathLike,
    overrides: Sequence[str] | None = None,
    profile: str | None = None,
    include_project: bool = True,
    model_line: str | None = None,
) -> DictConfig:
    """Compose project + model line + hardware profile + stage configs + dotlist overrides."""
    layers: list[DictConfig] = []
    if include_project:
        layers.append(OmegaConf.load(CONFIG_ROOT / "project.yaml"))
        line = model_line or os.environ.get(MODEL_LINE_ENV) or DEFAULT_MODEL_LINE
        layers.append(OmegaConf.load(model_line_path(line)))
        profile = profile or os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
        layers.append(OmegaConf.load(CONFIG_ROOT / "hardware" / f"{profile}.yaml"))
    for sc in stage_configs:
        layers.append(OmegaConf.load(_resolve(sc)))
    if overrides:
        layers.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*layers) if layers else OmegaConf.create({})
    assert isinstance(cfg, DictConfig)
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out


def config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    """sha256 of the *resolved* config — logged with every run and written into manifests."""
    obj = to_container(cfg) if isinstance(cfg, DictConfig) else cfg
    return sha256_obj(obj)
