"""Shared launcher machinery: compose configs, write a run manifest, build and exec a command.

Every training launch (SFT, RL, ablation) goes through :func:`prepare_run`, which

1. composes ``project.yaml + hardware profile + stage config [+ overlay] + CLI overrides``;
2. verifies the manifest of every input data file. A file edited after the
   pipeline wrote it is a hard error;
3. writes ``<run_dir>/launch/``: the resolved config, the rendered Hydra
   overrides, their sha256 digests, the data content hashes and the git
   revision. Any number in a report can then be traced back to its inputs
   (report §数据-6, "保证任何结果都能追溯到确切数据").

This module never imports verl; it only builds argv lists. So everything up to
the final ``exec`` can be exercised on a laptop with ``--dry-run``.
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from dpt.common.config import CONFIG_ROOT, MODEL_LINE_ENV, REPO_ROOT, config_hash, load_config, to_container
from dpt.common.io import write_json, write_yaml
from dpt.common.logging import get_logger
from dpt.common.manifest import manifest_path_for, verify_manifest
from dpt.launch.overrides import overrides_digest, to_overrides

log = get_logger(__name__)


def _config_chain(path: Path, seen: tuple[Path, ...] = ()) -> list[Path]:
    """Files to merge for ``path``, parents first.

    A stage file may name its parents with ``defaults_from: base.yaml`` (single
    base, e.g. an ablation overlay) and/or ``defaults_from_list: [a.yaml, b.yaml]``
    (sibling configs merged in order before the file, e.g. rl/algorithm.yaml).
    Relative names resolve against the file's directory, then against configs/.
    """
    if path in seen:
        raise ValueError(f"config inheritance cycle at {path}")
    raw = OmegaConf.load(path)
    parents = list(raw.get("defaults_from_list", []) or [])
    if raw.get("defaults_from"):
        parents.append(raw.get("defaults_from"))
    chain: list[Path] = []
    for name in parents:
        cand = path.parent / name
        cand = cand if cand.exists() else CONFIG_ROOT / name
        for p in _config_chain(cand, seen + (path,)):
            if p not in chain:
                chain.append(p)
    chain.append(path)
    return chain


def compose_stage(stage_config: str | os.PathLike, overrides: Sequence[str] = (),
                  profile: str | None = None, overlays: Sequence[str | os.PathLike] = ()) -> DictConfig:
    """Load project + profile + a stage config (with its parents) + overlays + dotlist overrides."""
    def resolve(p: str | os.PathLike) -> Path:
        q = Path(p)
        return q if q.is_absolute() or q.exists() else CONFIG_ROOT / q

    chain = _config_chain(resolve(stage_config))
    for ov in overlays:
        chain += [p for p in _config_chain(resolve(ov)) if p not in chain]
    os.environ.setdefault("DPT_REPO", str(REPO_ROOT))
    plain, flat = split_flat_overrides(overrides)
    cfg = load_config(*chain, overrides=plain, profile=profile)
    for section, items in flat.items():
        for key, value in items.items():
            cfg[section][key] = value
    # A stage config written for one model line (e.g. rl/rl_9b.yaml) must not silently pick up
    # another line's base model, architecture and gates from project.yaml.
    want = cfg.get("requires_model_line")
    have = OmegaConf.select(cfg, "project.model.line")
    if want is not None and want != have:
        raise ValueError(f"{stage_config} requires model line {want!r}, but the active line is {have!r}: "
                         f"export {MODEL_LINE_ENV}={want}")
    for k in ("defaults_from", "defaults_from_list", "requires_model_line"):
        cfg.pop(k, None)
    return cfg


FLAT_SECTIONS = ("verl", "verl_sft")


def split_flat_overrides(overrides: Sequence[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Separate CLI overrides that target the *flat* verl sections.

    ``verl.actor_rollout_ref.actor.optim.lr=2e-6`` must set the literal key
    ``"actor_rollout_ref.actor.optim.lr"`` inside ``verl``. An OmegaConf dotlist
    would instead create a nested ``actor_rollout_ref → actor → …`` tree that
    nobody reads. Values use OmegaConf's dotlist grammar (``1e-6`` → float, ``[a,b]`` → list).
    """
    plain: list[str] = []
    flat: dict[str, dict[str, Any]] = {}
    for ov in overrides:
        head, _, rest = ov.partition(".")
        if head in FLAT_SECTIONS and "=" in rest:
            key, _, raw = rest.partition("=")
            flat.setdefault(head, {})[key] = OmegaConf.to_container(
                OmegaConf.from_dotlist([f"v={raw}"]), resolve=False)["v"]
        else:
            plain.append(ov)
    return plain, flat


@dataclass
class PreparedRun:
    name: str
    run_dir: Path
    argv: list[str]
    env: dict[str, str]
    overrides: list[str]
    data_hashes: dict[str, str] = field(default_factory=dict)

    def shell(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        return f"{env} \\\n  " + " \\\n    ".join(shlex.quote(a) for a in self.argv)


def verify_inputs(paths: Sequence[str], *, require_manifest: bool = True) -> dict[str, str]:
    """Return {path: content_hash} for every input; raise if a manifest is missing or stale."""
    hashes: dict[str, str] = {}
    for p in paths:
        if not Path(p).exists():
            if not require_manifest:   # smoke / dry-run mode
                log.warning("input data not found (allowed in smoke mode): %s", p)
                continue
            raise FileNotFoundError(f"input data not found: {p}")
        twin = Path(p).with_suffix(".jsonl")
        if manifest_path_for(p).exists():
            hashes[p] = verify_manifest(p).content_hash
        elif Path(p).suffix == ".parquet" and manifest_path_for(twin).exists():
            # The pipeline manifests the canonical JSONL; the parquet is written from the same rows
            # in the same call (dpt.data.rl_build.write_rows). It must not be older than its twin.
            if Path(p).stat().st_mtime + 1 < twin.stat().st_mtime:
                raise ValueError(f"{p} is older than {twin}; re-run the publish stage")
            hashes[p] = verify_manifest(twin).content_hash
        elif require_manifest:
            raise FileNotFoundError(f"{p} has no manifest; produce it with the dpt.data pipeline")
        else:
            log.warning("no manifest for %s (allowed: require_manifest=False)", p)
    return hashes


def prepare_run(
    *,
    name: str,
    cfg: DictConfig,
    section: str,
    base_argv: list[str],
    run_dir: Path,
    data_files: Sequence[str],
    env: dict[str, str] | None = None,
    require_manifest: bool = True,
    launch_dir: Path | None = None,
) -> PreparedRun:
    """Render overrides from ``cfg[section]``, record everything under ``run_dir/launch``."""
    section_cfg = to_container(cfg)[section]
    overrides = to_overrides(section_cfg)
    hashes = verify_inputs(data_files, require_manifest=require_manifest)
    launch_dir = launch_dir or new_launch_dir(run_dir)
    write_yaml(launch_dir / "resolved_config.yaml", to_container(cfg))
    (launch_dir / "overrides.txt").parent.mkdir(parents=True, exist_ok=True)
    (launch_dir / "overrides.txt").write_text("\n".join(overrides) + "\n")
    write_json(launch_dir / "data_hashes.json", hashes)
    write_json(launch_dir / "provenance.json", {
        "name": name,
        "config_hash": config_hash(cfg),
        "overrides_digest": overrides_digest(overrides),
        "data_content_hashes": hashes,
        "git_revision": _git_rev(),
        "argv": base_argv + overrides,
        "hardware_profile": cfg.hardware.profile,
    })
    run_env = {"DPT_REPO": str(REPO_ROOT), "PYTHONPATH": f"{REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
    run_env.update(env or {})
    log.info("prepared %s: %d overrides, config %s, launch record %s",
             name, len(overrides), config_hash(cfg)[:12], launch_dir)
    return PreparedRun(name=name, run_dir=run_dir, argv=base_argv + overrides, env=run_env,
                       overrides=overrides, data_hashes=hashes)


def new_launch_dir(run_dir: Path) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_dir / "launch" / stamp


def hydra_check(module: str, overrides: Sequence[str], env: dict[str, str]) -> None:
    """Compose verl's full config from our overrides (``--cfg job``) without starting anything.

    Catches unknown keys, struct violations ("use +key=...") and type errors in seconds, instead of
    after Ray and the model have started. Raises RuntimeError with Hydra's message on failure.
    """
    proc = subprocess.run([python_exe(), "-m", module, *overrides, "--cfg", "job"],
                          env={**os.environ, **env}, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = "\n".join(line for line in (proc.stderr + proc.stdout).splitlines()
                        if "warn" not in line.lower())[-3000:]
        raise RuntimeError(f"verl rejected the rendered config ({module} --cfg job):\n{msg}")
    log.info("hydra check passed: %s accepts all %d overrides", module, len(overrides))


def execute(run: PreparedRun, *, dry_run: bool, check_module: str | None = None) -> int:
    print(run.shell())
    if dry_run:
        return 0
    if check_module:
        hydra_check(check_module, run.overrides, run.env)
    env = {**os.environ, **run.env}
    return subprocess.call(run.argv, env=env)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def python_exe() -> str:
    return sys.executable


def as_list(x: Any) -> list[str]:
    """Normalize a config value (None, a scalar or a list) into a list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]
