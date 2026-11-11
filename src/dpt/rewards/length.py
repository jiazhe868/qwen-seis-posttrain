"""Length control: the overlong penalty and the truncation flag (report §算法配置 "长度控制").

* Responses are capped at 24K tokens during RL.
* Between 20K and 24K the reward gets a linear penalty from 0 to −0.5. This
  is the DAPO-style "soft overlong punishment". It discourages over-thinking
  without a cliff at the cap.
* A **truncated** response (it hit the cap before finishing) is *not*
  treated as wrong. It is excluded from the loss entirely: "not finished" and
  "wrong" are different signals, and teaching the second when the first
  happened penalizes long-but-correct reasoning chains. The grader only flags
  it. The trainer applies the mask (``dpt.rl.filtering``).

The penalty is additive and is not floored at zero, so a long *wrong* answer
scores below a short wrong answer. That is intended: within a GRPO group only
relative values matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LengthConfig:
    max_response_tokens: int = 24576
    overlong_start: int = 20480
    max_penalty: float = 0.5
    # Rough chars-per-token used only when the true token count is unavailable
    # (Qwen tokenizer: ~3.5-4 chars/token for English, ~1.3 for Chinese).
    est_chars_per_token: float = 3.2
    # When another component already applies the token-exact overlong penalty (verl's
    # reward_manager=dapo uses valid_response_length), set this False to avoid double-penalizing.
    apply_penalty: bool = True
    # When False, truncation is flagged only from a real token count or finish_reason, never
    # from a character-based estimate (verl's trainer detects truncation from response_mask).
    truncation_from_estimate: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LengthConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def overlong_penalty(response_tokens: int, cfg: LengthConfig | None = None) -> float:
    """Return a value in ``[-max_penalty, 0]``: 0 up to ``overlong_start``, linear to the cap."""
    cfg = cfg or LengthConfig()
    if response_tokens <= cfg.overlong_start:
        return 0.0
    span = max(1, cfg.max_response_tokens - cfg.overlong_start)
    frac = min(1.0, (response_tokens - cfg.overlong_start) / span)
    return -cfg.max_penalty * frac


def is_truncated(
    response_tokens: int | None,
    cfg: LengthConfig | None = None,
    *,
    finish_reason: str | None = None,
) -> bool:
    """True if generation stopped because of the length cap (not because the model finished)."""
    cfg = cfg or LengthConfig()
    if finish_reason is not None:
        return finish_reason == "length"
    return response_tokens is not None and response_tokens >= cfg.max_response_tokens


def estimate_tokens(text: str, cfg: LengthConfig | None = None) -> int:
    cfg = cfg or LengthConfig()
    return int(len(text) / cfg.est_chars_per_token)
