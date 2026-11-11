"""Seeding and RNG-state capture.

Incident 2 (report §事故记录): resuming from a checkpoint that did not include
the sampler / curriculum / RNG state silently restarted the curriculum and
cost 30 steps. Every random source used by our code is therefore either
derived from an explicit seed or captured with :func:`capture_rng_state` and
restored with :func:`restore_rng_state`.
"""

from __future__ import annotations

import base64
import os
import pickle
import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for core modules
        pass


def capture_rng_state() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of python / numpy / torch RNG states."""
    state: dict[str, Any] = {
        "python": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii"),
        "numpy": base64.b64encode(pickle.dumps(np.random.get_state())).decode("ascii"),
    }
    try:
        import torch

        state["torch"] = base64.b64encode(pickle.dumps(torch.get_rng_state())).decode("ascii")
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(pickle.loads(base64.b64decode(state["python"])))
    np.random.set_state(pickle.loads(base64.b64decode(state["numpy"])))
    if "torch" in state:
        try:
            import torch

            torch.set_rng_state(pickle.loads(base64.b64decode(state["torch"])))
        except ImportError:
            pass


def derive_seed(base_seed: int, *names: object) -> int:
    """Derive an independent, reproducible sub-seed, e.g. ``derive_seed(seed, "curriculum", step)``."""
    import hashlib

    h = hashlib.sha256(("|".join([str(base_seed), *map(str, names)])).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)
