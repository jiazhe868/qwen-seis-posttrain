"""Post-processing of exported Hugging Face checkpoints: exact freezing and change verification.

Why "restore frozen modules on export" instead of trusting the trainer to freeze?
------------------------------------------------------------------------------
The report freezes the vision tower for the whole project (vision is out of
scope; only regression-checked). In verl 0.9.1 the actor's
``freeze_vision_tower`` field is never read, and module freezing exists only on
the LoRA path. In text-only full-parameter training the vision weights get no
gradient. But AdamW's decoupled weight decay still shrinks every parameter the
optimizer owns: ``θ ← θ − lr·wd·θ``. That is a small, steady drift. Copying
the vision tensors back from the base checkpoint at export time makes the
freeze exact by construction, whatever the trainer did. :func:`verify_export`
then checks that the language-model weights really changed (a silent no-op
export is a classic failure) and that frozen tensors are bit-identical to base.

CLI::

    python -m dpt.launch.checkpoint_tools restore-frozen --ckpt <hf_dir> --base <hf_dir_or_hub_id> \
        --prefix model.visual.
    python -m dpt.launch.checkpoint_tools verify --ckpt <hf_dir> --base <...> --prefix model.visual.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from dpt.common.logging import get_logger

log = get_logger(__name__)


def _resolve(path_or_id: str) -> Path:
    p = Path(path_or_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_id, allow_patterns=["*.safetensors", "*.json"]))


def _weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name → shard filename, for sharded or single-file safetensors checkpoints."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    from safetensors import safe_open

    single = model_dir / "model.safetensors"
    with safe_open(str(single), framework="pt") as f:
        return {k: single.name for k in f.keys()}


def _matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def restore_frozen(ckpt_dir: str, base: str, prefixes: Sequence[str]) -> int:
    """Overwrite every tensor under ``prefixes`` in ``ckpt_dir`` with the base checkpoint's copy.

    Returns the number of tensors restored. Shards are rewritten atomically
    (write to temp file, then rename).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    ckpt, base_dir = Path(ckpt_dir), _resolve(base)
    cmap, bmap = _weight_map(ckpt), _weight_map(base_dir)
    targets = [n for n in cmap if _matches(n, prefixes)]
    missing = [n for n in targets if n not in bmap]
    if missing:
        raise KeyError(f"{len(missing)} frozen tensors absent from base, e.g. {missing[:3]}")
    if not targets:
        log.warning("no tensors match %s in %s; nothing restored", prefixes, ckpt)
        return 0

    by_shard: dict[str, list[str]] = defaultdict(list)
    for n in targets:
        by_shard[cmap[n]].append(n)
    restored = 0
    for shard, names in by_shard.items():
        with safe_open(str(ckpt / shard), framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            metadata = f.metadata()
        for n in names:
            with safe_open(str(base_dir / bmap[n]), framework="pt") as bf:
                t = bf.get_tensor(n)
            if t.shape != tensors[n].shape or t.dtype != tensors[n].dtype:
                c = tensors[n]
                raise ValueError(f"{n}: base {tuple(t.shape)}/{t.dtype} vs ckpt {tuple(c.shape)}/{c.dtype}")
            tensors[n] = t
            restored += 1
        tmp = ckpt / f".{shard}.tmp"
        save_file(tensors, str(tmp), metadata=metadata or {"format": "pt"})
        tmp.replace(ckpt / shard)
    log.info("restored %d frozen tensors (%s) from %s", restored, ",".join(prefixes), base)
    return restored


def verify_export(ckpt_dir: str, base: str, prefixes: Sequence[str], *, sample: int = 64) -> dict:
    """Frozen tensors must equal base exactly; at least one trainable tensor must differ."""
    import torch
    from safetensors import safe_open

    ckpt, base_dir = Path(ckpt_dir), _resolve(base)
    cmap, bmap = _weight_map(ckpt), _weight_map(base_dir)
    frozen = [n for n in cmap if _matches(n, prefixes)]
    trainable = [n for n in cmap if not _matches(n, prefixes) and n in bmap]
    step = max(1, len(trainable) // sample)
    probe = trainable[::step][:sample]

    def get(d: Path, m: dict[str, str], n: str) -> "torch.Tensor":
        with safe_open(str(d / m[n]), framework="pt") as f:
            return f.get_tensor(n)

    frozen_mismatch = [n for n in frozen if not torch.equal(get(ckpt, cmap, n), get(base_dir, bmap, n))]
    changed = sum(not torch.equal(get(ckpt, cmap, n), get(base_dir, bmap, n)) for n in probe)
    report = {"frozen_tensors": len(frozen), "frozen_mismatch": len(frozen_mismatch),
              "probed_trainable": len(probe), "changed_trainable": int(changed)}
    if frozen_mismatch:
        raise AssertionError(f"frozen tensors differ from base: {frozen_mismatch[:3]} … ({report})")
    if probe and changed == 0:
        raise AssertionError(f"no trainable tensor changed — export is a no-op? ({report})")
    log.info("export verified: %s", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("restore-frozen", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--base", required=True)
        s.add_argument("--prefix", action="append", default=None, help="repeatable; default model.visual.")
    a = p.parse_args()
    prefixes = a.prefix or ["model.visual."]
    if a.cmd == "restore-frozen":
        restore_frozen(a.ckpt, a.base, prefixes)
    else:
        print(json.dumps(verify_export(a.ckpt, a.base, prefixes), indent=2))


if __name__ == "__main__":
    main()
