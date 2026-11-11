"""Model endpoints for the generators: the teacher (Qwen3.8-27B), the judge (Gemma 4) and the base (Qwen3.5-9B).

On the cluster each model is a vLLM server started by ``scripts/slurm/serve.slurm``; ``scripts/slurm/data.slurm``
passes its URL and served name. The servers run **no reasoning parser**, so a reply is the raw text
``<thinking></think>\\n\\n<answer>`` (``dpt.chat.thinking``). :class:`ChatModel` returns that text together
with the finish reason, because a reply cut off at ``max_tokens`` must never become training data.

Qwen models are prompted the way SFT data and RL rollouts are rendered: with the model's own chat
template and an explicit reasoning effort (``dpt.chat.render``), sent to ``/completions``. Without a
local tokenizer the request goes to ``/chat/completions`` and the server renders it.

Several servers of one model (``--url`` given more than once) share the requests round-robin.

The hosted eval model (``dpt.eval.hosted``) is not reachable from here: its output may not enter
training data (``tests/eval/test_independence.py``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dpt.chat.thinking import THINK_END, ThinkSplit, split_thinking
from dpt.common.logging import get_logger
from dpt.data.backends import OpenAICompatClient, load_hf_tokenizer

log = get_logger(__name__)


@dataclass(frozen=True)
class Reply:
    text: str                      # raw: thinking, THINK_END, answer
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def split(self) -> ThinkSplit:
        return split_thinking(self.text)

    @property
    def answer(self) -> str | None:
        """The final answer, or None if the reply was cut off or has no end of thinking."""
        s = self.split
        return s.answer if (s.has_end and not self.truncated) else None


class ChatModel:
    """One model behind one or more OpenAI-compatible servers."""

    def __init__(self, clients: Sequence[OpenAICompatClient], *, model_id: str, tokenizer: Any | None = None,
                 effort: str | None = "medium", temperature: float = 1.0, top_p: float = 1.0,
                 max_tokens: int = 16384):
        if not clients:
            raise ValueError("ChatModel needs at least one server")
        self.clients = list(clients)
        self.model_id = model_id           # recorded in source.model / meta.generated_by (the HF id)
        self.tokenizer = tokenizer
        self.effort = effort
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self._rr = itertools.cycle(range(len(self.clients)))
        self._lock = threading.Lock()

    def _client(self) -> OpenAICompatClient:
        with self._lock:
            return self.clients[next(self._rr)]

    def generate(self, messages: Sequence[dict[str, Any]], *, n: int = 1, seed: int | None = None,
                 max_tokens: int | None = None, temperature: float | None = None) -> list[Reply]:
        c = self._client()
        common: dict[str, Any] = {
            "model": c.model, "n": n, "max_tokens": int(max_tokens or self.max_tokens),
            "temperature": self.temperature if temperature is None else temperature, "top_p": self.top_p,
        }
        if seed is not None:
            common["seed"] = seed
        if self.tokenizer is not None and self.effort is not None:
            from dpt.chat.render import render_chat

            prompt = render_chat(self.tokenizer, messages, effort=self.effort, add_generation_prompt=True)
            # _post, not complete(): the finish reason tells a cut-off reply from a finished one
            out = c._post("/completions", {**common, "prompt": prompt})
            return [Reply(ch["text"], ch.get("finish_reason")) for ch in sorted(out["choices"], key=_index)]
        kwargs: dict[str, Any] = {"enable_thinking": True}
        if self.effort is not None:
            kwargs["reasoning_effort"] = self.effort
        out = c._post("/chat/completions", {**common, "messages": list(messages),
                                            "chat_template_kwargs": kwargs})
        replies = []
        for ch in sorted(out["choices"], key=_index):
            msg = ch.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            text = f"{reasoning}\n{THINK_END}\n\n{content}" if reasoning is not None else content
            replies.append(Reply(text, ch.get("finish_reason")))
        return replies

    def count_tokens(self, text: str) -> int:
        """Exact with the model's tokenizer, else the graders' estimate (3.2 characters per token)."""
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return int(len(text) / 3.2)


def _index(choice: dict[str, Any]) -> int:
    return int(choice.get("index", 0))


def last_json(text: str, required: str) -> Any:
    """The value of key ``required`` in the last JSON object of a reply that has it.

    Thinking comes first in these replies and may hold draft JSON, so the final answer is searched first
    and objects are scanned from the end.
    """
    final = split_thinking(text or "")
    for t in ([final.answer, text] if final.has_end else [text]):
        for candidate in (t, _escape_stray_backslashes(t)):     # strict first; repaired only if that fails
            dec, pos = json.JSONDecoder(), (candidate or "").rfind("{")
            while pos >= 0:
                try:
                    obj, _ = dec.raw_decode(candidate, pos)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and required in obj:
                    return obj[required]
                pos = candidate.rfind("{", 0, pos)
    raise ValueError(f"no JSON object with {required!r} in the reply")


_HEX4 = re.compile(r"[0-9a-fA-F]{4}")


def _escape_stray_backslashes(text: str | None) -> str:
    """Repair JSON in which a model wrote LaTeX unescaped (``\\sigma``, ``\\frac``, ``\\tau``).

    Used only after strict parsing failed. A backslash stays a JSON escape when it cannot be the start of a
    LaTeX command: ``\\\\``, ``\\"``, ``\\/``, ``\\uXXXX``, or one of ``\\b \\f \\n \\r \\t`` NOT followed by a letter
    (``\\n`` + space is a newline, ``\\nabla`` is LaTeX). Every other backslash becomes a literal one.
    """
    t, out, i = text or "", [], 0
    while i < len(t):
        c = t[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = t[i + 1] if i + 1 < len(t) else ""
        after = t[i + 2] if i + 2 < len(t) else ""
        if nxt in ('\\', '"', "/"):
            out.append(t[i:i + 2])
            i += 2
        elif nxt == "u" and _HEX4.fullmatch(t[i + 2:i + 6] or ""):
            out.append(t[i:i + 6])
            i += 6
        elif nxt in "bfnrt" and nxt and not after.isalpha():
            out.append(t[i:i + 2])
            i += 2
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def last_block(text: str, tag: str) -> str | None:
    """Content of the last ``<tag>…</tag>`` in the final answer (None if there is none)."""
    s = split_thinking(text or "")
    t = s.answer if s.has_end else ""
    end = t.rfind(f"</{tag}>")
    start = t.rfind(f"<{tag}>", 0, end) if end >= 0 else -1
    if start < 0:
        return None
    return t[start + len(tag) + 2:end].strip()


# --------------------------------------------------------------------------- construction


def add_model_args(ap: argparse.ArgumentParser, *, prefix: str = "", default_effort: str | None = "medium",
                   max_tokens: int = 16384) -> None:
    p = f"--{prefix}-" if prefix else "--"
    ap.add_argument(f"{p}url", action="append", required=True,
                    help="server base URL (http://host:8000 or …/v1); repeat for several servers")
    ap.add_argument(f"{p}model", action="append", required=True,
                    help="served model name (serve.slurm serves <name> as <name>): once, or once per --url")
    ap.add_argument(f"{p}tokenizer", default=None,
                    help="HF id of the model: renders prompts locally and is recorded as the model id")
    ap.add_argument(f"{p}effort", default=default_effort, help="reasoning effort (Qwen chat template)")
    ap.add_argument(f"{p}max-tokens", type=int, default=max_tokens)
    ap.add_argument(f"{p}temperature", type=float, default=1.0)
    # A request also waits in the server's queue: on the shared H100 teacher a 20K-token reply took ~26 min at
    # ~13 tok/s, so a 30-min timeout re-sent the longest replies from scratch (2026-10-24).
    ap.add_argument(f"{p}timeout", type=float, default=7200.0, help="seconds per request, queue wait included")


def model_from_args(args: argparse.Namespace, *, prefix: str = "", concurrency: int = 64) -> ChatModel:
    a = (lambda k: getattr(args, f"{prefix}_{k}" if prefix else k))
    tok_id = a("tokenizer")
    urls, names = list(a("url")), list(a("model"))
    if len(names) not in (1, len(urls)):
        raise SystemExit(f"give --{prefix + '-' if prefix else ''}model once or once per --url ({len(urls)})")
    names = names * len(urls) if len(names) == 1 else names
    tokenizer = load_hf_tokenizer(tok_id) if tok_id else None
    timeout = getattr(args, f"{prefix}_timeout" if prefix else "timeout", 7200.0)
    clients = [OpenAICompatClient(_v1(u), m, max_concurrency=concurrency, timeout=timeout) for u, m in zip(urls, names)]
    return ChatModel(clients, model_id=tok_id or names[0], tokenizer=tokenizer, effort=a("effort"),
                     temperature=a("temperature"), max_tokens=a("max_tokens"))


def _v1(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


# --------------------------------------------------------------------------- resumable JSONL output


class Appender:
    """Thread-safe append-only JSONL (one flush per record), so an interrupted run loses at most a line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, records: Iterable[dict[str, Any]]) -> None:
        lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
        if not lines:
            return
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()


def read_existing(path: str | Path) -> list[dict[str, Any]]:
    """Records of a possibly half-written JSONL file; a truncated last line is ignored."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s: skipping an unreadable line (interrupted write?)", p)
    return out
