"""Turn the ``verl:`` section of our YAML configs into Hydra command-line overrides.

Why not hand-written shell scripts full of ``key=value`` lines? Three reasons:

1. **One source of truth.** Values such as TP size, response cap, clip range
   and LR live in ``configs/`` and are interpolated (``${hardware.train.tp}``)
   into the verl section. A shell script would duplicate them, and duplicated
   numbers drift.
2. **Reviewable diffs + hashing.** The rendered override list is written into
   the run directory along with its sha256. "What exactly did run X use?"
   then has a precise answer.
3. **Testability.** Formatting rules (lists, dicts, booleans, ``null``, keys that
   need Hydra's ``+`` prefix) are easy to get subtly wrong. They are unit-tested
   here on CPU instead of being discovered 20 minutes into a cluster launch.

Conventions in the YAML ``verl:`` mapping:

* keys are dotted verl config paths, e.g. ``actor_rollout_ref.actor.optim.lr``
* a key starting with ``+`` adds a field that is not in verl's schema
  (e.g. ``+ray_kwargs.ray_init.runtime_env.env_vars.FOO``); a key starting
  with ``++`` forces add-or-override
* values may be scalars, lists or dicts; they are rendered in Hydra's grammar
* the value ``"__unset__"`` drops the key. An overlay config (e.g. the FSDP
  fallback) uses it to remove a key inherited from its base config.
* a key without dots whose value is a bare word (``engine: megatron``) is a
  Hydra *config-group selection* and is rendered first
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

UNSET = "__unset__"  # sentinel value: drop this key from the rendered overrides

# Characters that force quoting of a string value in Hydra's override grammar.
_NEEDS_QUOTES = re.compile(r"[\s,\[\]{}()=:'\"\\$*?#]|^$")


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _render_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    # A *string* that Hydra would otherwise parse as a number/bool/null must stay a string, e.g. Ray's
    # runtime_env env_vars (CUDA_DEVICE_MAX_CONNECTIONS: "1") must be str, not int.
    if s.lower() in {"true", "false", "null", "none", "nan", "inf", "-inf"} or _looks_numeric(s) \
            or _NEEDS_QUOTES.search(s):
        # Single-quote and escape; Hydra accepts '...' with \' escapes.
        return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return s


def render_value(v: Any) -> str:
    """Render a Python value in Hydra override syntax (lists → ``[a,b]``, dicts → ``{k:v}``)."""
    if isinstance(v, Mapping):
        return "{" + ",".join(f"{k}:{render_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(render_value(x) for x in v) + "]"
    return _render_scalar(v)


def to_overrides(verl_section: Mapping[str, Any]) -> list[str]:
    """Flatten ``{dotted_key: value}`` into a sorted list of ``key=value`` strings.

    Sorting makes the rendered list (and therefore its hash) independent of
    YAML key order.
    """
    groups, fields = [], []
    for key, value in verl_section.items():
        if not isinstance(key, str) or not key.lstrip("+"):
            raise ValueError(f"invalid override key: {key!r}")
        if value == UNSET:
            continue
        item = f"{key}={render_value(value)}"
        is_group = "." not in key and not key.startswith("+") and isinstance(value, str)
        (groups if is_group else fields).append(item)
    return sorted(groups) + sorted(fields, key=lambda s: s.lstrip("+"))


def overrides_digest(overrides: Iterable[str]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(list(overrides)).encode()).hexdigest()
