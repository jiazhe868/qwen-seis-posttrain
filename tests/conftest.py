"""Test configuration: make ``src/`` importable and auto-skip GPU / network tests.

Run everything CPU-side with:  python -m pytest
Run network tests explicitly:  python -m pytest -m network
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _cuda_available() -> bool:
    try:
        import warnings

        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old-driver warnings are noise for CPU-only runs
            return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    selected_markers = config.getoption("-m") or ""
    skip_gpu = pytest.mark.skip(reason="no CUDA device visible")
    skip_net = pytest.mark.skip(reason="network test; select with -m network")
    has_gpu = _cuda_available()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "network" in item.keywords and "network" not in selected_markers:
            item.add_marker(skip_net)
