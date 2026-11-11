"""Minimal logging setup so that library modules never call ``print``.

All loggers live under the ``dpt`` namespace; ``DPT_LOG_LEVEL`` controls verbosity.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root ``dpt`` handler once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("dpt")
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            # Include the rank when running under torchrun / Ray so multi-process logs stay readable.
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-"))
            handler.setFormatter(
                logging.Formatter(f"%(asctime)s [%(levelname)s] [rank {rank}] %(name)s: %(message)s")
            )
            root.addHandler(handler)
        root.setLevel(os.environ.get("DPT_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    if not name.startswith("dpt"):
        name = f"dpt.{name}"
    return logging.getLogger(name)
