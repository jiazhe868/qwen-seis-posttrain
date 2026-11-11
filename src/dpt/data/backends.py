"""Model backends for the data pipeline: OpenAI-compatible HTTP clients plus deterministic offline fakes.

Online, the pipeline talks to **vLLM servers** through the OpenAI-compatible API.
Every model-dependent stage uses one: difficulty calibration (base @ medium,
8 samples), answer verification (two other model families), embeddings, the
paraphrase judge and the memorization probe.

    vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --max-model-len 32768   # 4×A100-40GB box

Two ways to generate
--------------------
1. ``/v1/completions`` with a prompt **we render ourselves** via
   ``dpt.chat.render.render_chat``. This is preferred: it is the same template
   path, with the same reasoning-effort check, that SFT data and RL rollouts
   use. The raw completion text is "<reasoning></think>\\n\\n<answer>". That is
   exactly what graders expect (``dpt.chat.thinking``).
2. ``/v1/chat/completions`` with ``chat_template_kwargs``. This is the fallback
   when no local tokenizer is available. If the server runs a reasoning parser,
   the reasoning comes back in ``reasoning_content`` and we re-join it with
   ``</think>``, so downstream code sees the same raw format either way.

Offline fakes
-------------
``Fake*`` classes are deterministic functions of their inputs (blake2b hashes,
no RNG state). They exist for unit tests and for ``pipeline --offline`` runs.
Those runs exercise every stage's plumbing, manifests and funnel without GPUs.
Their outputs mean nothing about model quality.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from dpt.chat.thinking import THINK_END
from dpt.common.logging import get_logger

log = get_logger(__name__)

GenerateFn = Callable[..., list[str]]  # (messages, *, n, effort, max_tokens, seed) -> list[str]


def _u01(*parts: object) -> float:
    """Deterministic uniform(0,1) from arbitrary parts."""
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2**64


# =============================================================================================
# Online clients
# =============================================================================================
class OpenAICompatClient:
    """Minimal synchronous client for an OpenAI-compatible server (vLLM, SGLang, a judge service).

    Retries with exponential backoff on transport errors and 5xx. ``max_concurrency``
    bounds the thread pool used by :meth:`map`.

    With ``url_file`` (a ``serve.slurm`` endpoint file: ``http://host:port``, no ``/v1``), the server
    address is re-read whenever the server cannot be reached, and a request waits up to ``wait_s`` seconds
    for a server to answer instead of failing: a judge window that closes and reopens on another node then
    costs a wait, not the request (2026-10-24).
    """

    poll_s = 30.0                       # between attempts while waiting for a server (``url_file``)

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 1800.0,
        max_retries: int = 4,
        max_concurrency: int = 64,
        transport: Any | None = None,
        url_file: str | None = None,
        wait_s: float = 0.0,
    ):
        import httpx  # core dependency, imported lazily to keep import time low

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.url_file, self.wait_s = url_file, float(wait_s)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # ``transport`` lets tests inject ``httpx.MockTransport``; production uses the default. The pool must
        # hold ``max_concurrency`` connections: httpx allows 100, which capped bank verification at 100
        # requests in flight (2026-10-24).
        limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
        self._http = httpx.Client(timeout=timeout, headers=headers, transport=transport, limits=limits)

    def _server(self) -> str:
        """The base URL: from ``url_file`` when it names a server, else the one given."""
        if self.url_file:
            try:
                root = Path(self.url_file).read_text().strip().rstrip("/")
            except OSError:
                root = ""
            if root:
                return f"{root}/v1"
        return self.base_url

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        delay, deadline = 2.0, time.monotonic() + self.wait_s
        attempt = 0
        while True:
            try:
                r = self._http.post(f"{self._server()}{path}", json=payload)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                down = isinstance(e, httpx.TransportError) or e.response.status_code >= 500
                if self.url_file and down and time.monotonic() < deadline:   # wait for a server to answer
                    log.warning("POST %s failed (%s); server down, waiting %.0fs", path, e, self.poll_s)
                    time.sleep(self.poll_s)
                    continue
                if attempt == self.max_retries:
                    raise
                attempt += 1
                log.warning("POST %s failed (%s); retry %d in %.0fs", path, e, attempt, delay)
                time.sleep(delay)
                delay *= 2

    def complete(
        self, prompt: str, *, n: int = 1, temperature: float = 1.0, top_p: float = 1.0,
        max_tokens: int = 1024, seed: int | None = None, stop: Sequence[str] | None = None,
    ) -> list[str]:
        payload: dict[str, Any] = {
            "model": self.model, "prompt": prompt, "n": n, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = list(stop)
        out = self._post("/completions", payload)
        return [c["text"] for c in sorted(out["choices"], key=lambda c: c["index"])]

    def chat(
        self, messages: Sequence[dict[str, Any]], *, n: int = 1, temperature: float = 1.0, top_p: float = 1.0,
        max_tokens: int = 1024, seed: int | None = None, effort: str | None = None,
        enable_thinking: bool = True,
    ) -> list[str]:
        kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        payload: dict[str, Any] = {
            "model": self.model, "messages": list(messages), "n": n, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens, "chat_template_kwargs": kwargs,
        }
        if seed is not None:
            payload["seed"] = seed
        out = self._post("/chat/completions", payload)
        texts = []
        for c in sorted(out["choices"], key=lambda c: c["index"]):
            msg = c["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            # Re-join parsed reasoning so callers always see the raw "<thinking></think>answer" format.
            texts.append(f"{reasoning}\n{THINK_END}\n\n{content}" if reasoning is not None else content)
        return texts

    def embed(self, texts: Sequence[str], *, batch_size: int = 256) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out = self._post("/embeddings", {"model": self.model, "input": list(texts[i:i + batch_size])})
            vecs.extend(d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"]))
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        """Run ``fn`` over ``items`` concurrently (order preserved)."""
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(fn, items))


class VLLMGenerator:
    """``generate(messages, n=, effort=, max_tokens=, seed=)`` backed by a vLLM server.

    With a ``tokenizer``, it renders prompts locally (preferred, see module docstring)
    and calls ``/completions``. Without one, it uses ``/chat/completions``.
    """

    def __init__(self, client: OpenAICompatClient, tokenizer: Any | None = None,
                 temperature: float = 1.0, top_p: float = 1.0):
        self.client = client
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, messages: Sequence[dict[str, Any]], *, n: int = 1, effort: str | None = "medium",
                 max_tokens: int = 16384, seed: int | None = None) -> list[str]:
        # effort=None: do not send any effort field (other model families, e.g. verification models).
        if self.tokenizer is not None and effort is not None:
            from dpt.chat.render import render_chat

            prompt = render_chat(self.tokenizer, messages, effort=effort, add_generation_prompt=True)
            return self.client.complete(prompt, n=n, temperature=self.temperature, top_p=self.top_p,
                                        max_tokens=max_tokens, seed=seed)
        return self.client.chat(messages, n=n, temperature=self.temperature, top_p=self.top_p,
                                max_tokens=max_tokens, seed=seed, effort=effort)


class CachedGenerator:
    """A generator whose replies are appended to a JSONL file as they arrive and reused on the next run.

    A stage that dies with its server (a 2-hour dev-queue window) then resumes where it stopped instead of
    sampling everything again. Replies are keyed by ``label`` (one per model) and the exact request
    (messages, n, effort, max_tokens, seed); the server URL is not part of the key, so a server that
    comes back on another node still hits the cache. Delete the file to sample afresh. A torn last line
    (the process killed mid-write) is skipped.
    """

    def __init__(self, gen: Callable[..., list[str]], path: str | Path, label: str):
        import threading

        self.gen, self.path, self.label = gen, Path(path), label
        self._lock = threading.Lock()
        self._rows: dict[str, list[str]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("label") == label:
                        self._rows[r["key"]] = r["out"]
        self.reused = 0

    def __len__(self) -> int:
        return len(self._rows)

    def __call__(self, messages: Sequence[dict[str, Any]], **kw: Any) -> list[str]:
        key = hashlib.sha256(json.dumps([list(messages), kw], sort_keys=True, ensure_ascii=False)
                             .encode()).hexdigest()
        with self._lock:
            if key in self._rows:
                self.reused += 1
                return list(self._rows[key])
        out = list(self.gen(messages, **kw))
        with self._lock:
            self._rows[key] = out
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"label": self.label, "key": key, "out": out}, ensure_ascii=False) + "\n")
        return out


class LocalEmbedder:
    """Sentence embeddings computed in-process with Hugging Face transformers: decontamination without an
    embedding server (``backends.embed.local: true``).

    ``pooling="last"`` is what Qwen3-Embedding expects (the last token's hidden state, with left padding);
    ``"mean"`` averages over the attention mask. Rows are L2-normalized like :meth:`OpenAICompatClient.embed`.
    On a CPU node the 0.6B model embeds a few hundred short problems per second.
    """

    def __init__(self, model_id: str, *, pooling: str = "last", batch_size: int = 64, max_length: int = 512,
                 device: str | None = None, model: Any | None = None, tokenizer: Any | None = None):
        import torch

        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling, self.batch_size, self.max_length = pooling, batch_size, max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if model is None:
            from transformers import AutoModel

            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.tok, self.model = tokenizer, model.to(self.device).eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tok(list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**batch).last_hidden_state.float()
                if self.pooling == "last":
                    v = h[:, -1]
                else:
                    m = batch["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)


class LLMParaphraseJudge:
    """Asks a strong model whether two problems are the same problem (decontam layer 2).

    The instruction says that rewording or changing numeric values still counts
    as the same problem (report §污染检查 method 2).
    """

    PROMPT = (
        "You are auditing an evaluation set for contamination. Decide whether PROBLEM A and PROBLEM B are "
        "the SAME problem. They count as the same if one is a rewording, translation, or reformatting of "
        "the other, or if only numeric values, names or units were changed while the solution method and "
        "structure are identical. They are different if they ask for different quantities or need "
        "different reasoning.\n\nPROBLEM A:\n{a}\n\nPROBLEM B:\n{b}\n\n"
        "Answer with exactly one word: SAME or DIFFERENT."
    )

    def __init__(self, client: OpenAICompatClient):
        self.client = client

    def same_problem(self, a: str, b: str) -> bool:
        out = self.client.chat([{"role": "user", "content": self.PROMPT.format(a=a, b=b)}],
                               temperature=0.0, max_tokens=4096, effort="low")[0]
        answer = out.rsplit(THINK_END, 1)[-1].strip().upper()
        return answer.startswith("SAME")


# =============================================================================================
# Offline fakes (deterministic; for tests and `pipeline --offline`)
# =============================================================================================
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _masked_tokens(text: str) -> list[str]:
    """Lowercased word tokens with every number replaced by <num> (paraphrase-with-new-numbers proxy)."""
    t = _NUM_RE.sub(" <num> ", text.lower())
    return [w if w != "num" else "<num>" for w in re.findall(r"<num>|[a-z]+|[\u4e00-\u9fff]", t)]


class FakeEmbedder:
    """Hashing-trick bag of words with numbers masked; cosine ≈ topical overlap. Deterministic."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _masked_tokens(t):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class FakeParaphraseJudge:
    """'Same problem' iff number-masked token Jaccard ≥ ``threshold``."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def same_problem(self, a: str, b: str) -> bool:
        sa, sb = set(_masked_tokens(a)), set(_masked_tokens(b))
        return bool(sa and sb) and len(sa & sb) / len(sa | sb) >= self.threshold


class FakeGenerator:
    """Simulated policy for difficulty calibration.

    Each problem gets a deterministic skill level ``p ∈ {0, 0.5, 1}`` from its
    question hash. Each sample is correct with probability ``p``, again via a
    hash. The response embeds a marker that :func:`fake_grade` reads.
    Outcomes therefore cover 0/8, mixed and 8/8 without any RNG state.
    """

    def __init__(self, levels: Sequence[float] = (0.0, 0.5, 1.0), salt: str = "fake-gen"):
        self.levels = tuple(levels)
        self.salt = salt

    def skill(self, question: str) -> float:
        return self.levels[int(_u01(self.salt, "skill", question) * len(self.levels)) % len(self.levels)]

    def __call__(self, messages: Sequence[dict[str, Any]], *, n: int = 1, effort: str = "medium",
                 max_tokens: int = 16384, seed: int | None = None) -> list[str]:
        q = messages[-1]["content"]
        p = self.skill(q)
        outs = []
        for i in range(n):
            ok = _u01(self.salt, "sample", q, seed, i) < p
            outs.append(f"(simulated reasoning)\n{THINK_END}\n\n<!--fake:{'correct' if ok else 'wrong'}-->")
        return outs


def fake_grade(problem: Any, response: str) -> float:
    """Grader paired with :class:`FakeGenerator`: 1.0 iff the response carries the 'correct' marker."""
    return 1.0 if "<!--fake:correct-->" in response else 0.0


class FakeSolverFamily:
    """Simulated model family for answer verification.

    It answers with ``truth_fn(problem)``, which defaults to the programmatic
    solver value, else the label. ``error_rate`` of samples are wrong in a
    deterministic way. It returns ``\\boxed{}`` answers like a real model.
    """

    def __init__(self, name: str, truth_fn: Callable[[Any], float | None], error_rate: float = 0.1):
        self.name = name
        self.truth_fn = truth_fn
        self.error_rate = error_rate

    def __call__(self, problem: Any, n: int) -> list[str]:
        truth = self.truth_fn(problem)
        unit = problem.answer.get("unit", "")
        outs = []
        for i in range(n):
            if truth is None:
                outs.append(f"{THINK_END}\n\nI cannot determine the answer.")
                continue
            wrong = _u01(self.name, problem.uid, i) < self.error_rate
            v = truth * (1.37 if wrong else 1.0)
            outs.append(f"(reasoning)\n{THINK_END}\n\n\\boxed{{{v:.6g} {unit}}}".replace(" }", "}"))
        return outs


class FakeCompleter:
    """Base-model 'completion' for memorization probes: recites a memorized text or emits filler."""

    def __init__(self, memorized: Sequence[str] = ()):
        self.memorized = list(memorized)

    def __call__(self, prefix: str, max_tokens: int = 64) -> str:
        for text in self.memorized:
            if text.startswith(prefix):
                return text[len(prefix):]
        return " and therefore the answer follows from the given data."


# =============================================================================================
# Offline tokenizer (renders the real chat template; char-level tokens)
# =============================================================================================
class JinjaCharTokenizer:
    """Char-level tokenizer that renders a Hugging Face chat template file with HF's Jinja settings.

    Used by ``pipeline --offline`` so SFT rendering, the effort check and the
    loss masks run against the real Qwen3.8 template without downloading the
    tokenizer. Token counts are characters, so they are about 3–4× BPE counts.
    That is fine for plumbing, but useless for real bucket statistics.
    """

    def __init__(self, template_path: str | Path):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def _raise(msg: str):
            raise jinja2.exceptions.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(Path(template_path).read_text(encoding="utf-8"))

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        if tokenize:
            raise ValueError("JinjaCharTokenizer only renders text")
        return self.template.render(messages=messages, tools=tools,
                                    add_generation_prompt=add_generation_prompt, **kw)

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        out: dict[str, Any] = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def load_hf_tokenizer(name_or_path: str) -> Any:
    """Lazily load a Hugging Face fast tokenizer (needs ``transformers``)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=False)
