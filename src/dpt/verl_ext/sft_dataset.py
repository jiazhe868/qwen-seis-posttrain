"""verl SFT dataset that renders each conversation with *its own* reasoning effort.

Plugged in via ``data.custom_cls.{path,name}`` in ``configs/sft/*.yaml``.

Why not verl's default ``MultiTurnSFTDataset``?
-----------------------------------------------
1. **Per-row effort.** The default dataset applies one global
   ``apply_chat_template_kwargs`` to every row. Our SFT mix deliberately
   contains all three effort levels. Domain data is rendered at ``medium``.
   The self-distilled general replay keeps the effort it was generated with,
   so the model keeps the low/medium/xhigh → thinking-length mapping. Losing
   that mapping was the first SFT incident in the report.
2. **Template compatibility.** The default dataset renders every message on
   its own and strips a precomputed system prefix. The Qwen3.8 template
   refuses to render a lone assistant message ("No user query found"), and it
   injects an effort sentence into the system turn, which breaks the
   prefix-stripping arithmetic.
3. **One tested code path.** We call :func:`dpt.chat.render.tokenize_with_assistant_mask`,
   the function the unit tests check against the real template, so the bytes
   trained on are the bytes tested.

Output format matches ``MultiTurnSFTDataset.__getitem__`` (``input_ids``,
``loss_mask``, ``position_ids``, plus ``attention_mask`` in ``right`` pad mode).
``loss_mask[i] = 1`` means token ``i`` is a training target; verl's SFT loss
shifts it by one (``torch.roll(..., -1)``). Text only: the vision tower is
frozen and out of scope.

Input: JSONL written by ``dpt.data.sft_build`` or parquet, with columns ``messages`` (list of dicts or JSON string),
``reasoning_effort`` (``low|medium|xhigh``), optional ``tools`` (list or JSON).
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset

from dpt.chat.render import tokenize_with_assistant_mask
from dpt.common.logging import get_logger

log = get_logger(__name__)


def _as_list(x: Any) -> Any:
    """Normalize parquet-decoded values (JSON strings, numpy arrays of dicts) into plain Python lists.

    A missing value is ``None``. pandas reads a key that only some JSONL rows have (``tools``: the tool
    trajectories only) as NaN in the others, and ``list(nan)`` stopped the 9B SFT at its first batch
    (2026-10-28).
    """
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x is None or (isinstance(x, float) and x != x):
        return None
    if isinstance(x, str):
        return json.loads(x) if x else None
    if isinstance(x, (list, tuple)):
        return [_as_list(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for v in x]
    if isinstance(x, dict):
        return {k: _as_list(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in x.items()}
    return x


class DPTSFTDataset(Dataset):
    """Constructed by verl as ``cls(parquet_files=..., tokenizer=..., config=..., processor=..., max_samples=...)``."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        import pandas as pd

        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_mode = config.get("pad_mode", "no_padding")
        self.max_length = int(config.get("max_length", 32768))
        if config.get("truncation", "error") != "error":
            # Truncating a reasoning trace teaches the model to stop mid-thought; oversized
            # samples are removed by the SFT builder instead (dpt.data.sft_build).
            raise ValueError("DPTSFTDataset requires data.truncation=error")
        files = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)

        def _read(f: str) -> "pd.DataFrame":
            # JSONL straight from dpt.data.sft_build (no pyarrow needed), or parquet.
            return pd.read_json(f, lines=True) if str(f).endswith(".jsonl") else pd.read_parquet(f)

        df = pd.concat([_read(f) for f in files], ignore_index=True)
        if max_samples and max_samples > 0:
            df = df.iloc[:max_samples]
        missing = {"messages", "reasoning_effort"} - set(df.columns)
        if missing:
            raise ValueError(f"SFT parquet is missing columns {missing}; build it with dpt.data.sft_build")
        self.messages = [_as_list(m) for m in df["messages"].tolist()]
        self.efforts = df["reasoning_effort"].astype(str).tolist()
        self.tools = [_as_list(t) for t in df["tools"].tolist()] if "tools" in df.columns else [None] * len(df)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        log.info("DPTSFTDataset: %d rows, effort mix %s", len(df), df["reasoning_effort"].value_counts().to_dict())

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        enc = tokenize_with_assistant_mask(
            self.tokenizer, self.messages[i], effort=self.efforts[i], tools=self.tools[i] or None,
            max_length=self.max_length,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        loss_mask = torch.tensor(enc["loss_mask"], dtype=torch.long)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        if self.pad_mode == "no_padding":
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        # right padding (used only with the FSDP fallback engine)
        pad = self.max_length - len(input_ids)
        attention_mask = torch.ones_like(input_ids)
        if pad > 0:
            input_ids = torch.cat([input_ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, dtype=torch.long)])
            position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "position_ids": position_ids, "loss_mask": loss_mask}
