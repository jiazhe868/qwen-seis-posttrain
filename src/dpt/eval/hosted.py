"""A chat model behind ``/v1/chat/completions``, for the eval side only: D-Report judges and eval drafting.

The same class talks to a vLLM-served open model and to a hosted API. Request shapes differ, so they are
configurable (the keys of an ``eval_suite`` judge block):

=================  ===========================  =========================================
key                vLLM (Gemma judge)           hosted reasoning model (GPT-6 Luna)
=================  ===========================  =========================================
temperature        0.0                          null: omitted (reasoning models reject it)
max_tokens_field   max_tokens                   max_completion_tokens (includes reasoning)
extra_body         {}                           {reasoning_effort: xhigh}
=================  ===========================  =========================================

Terms: on the 9B plan the hosted model is GPT-6 Luna, and OpenAI's terms forbid using its output
to develop competing models. Nothing it produces may enter training data, so it is used for eval
material and eval judging only, and no training package imports this module
(``tests/eval/test_independence.py``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from dpt.common.logging import get_logger

log = get_logger(__name__)


class JudgeVersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatReply:
    content: str
    finish_reason: str | None
    model: str | None


def host_only(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    return e[: -len("/v1")] if e.endswith("/v1") else e


class ChatEndpoint:
    """One model behind an OpenAI-compatible chat endpoint, with retries and version pinning."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        pinned_model_version: str | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 2048,
        temperature: float | None = 0.0,
        max_tokens_field: str = "max_tokens",
        extra_body: dict[str, Any] | None = None,
        retries: int = 4,
        retry_delay_s: float = 2.0,
        client: httpx.Client | None = None,
    ):
        self.endpoint = host_only(endpoint)  # accept host or host/v1
        self.model = model
        self.pinned = pinned_model_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tokens_field = max_tokens_field
        self.extra_body = {k: v for k, v in (extra_body or {}).items() if v not in (None, "None", "null")}
        self.retries, self.retry_delay_s = retries, retry_delay_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout_s, headers=headers)

    def body(self, messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            self.max_tokens_field: self.max_tokens,
            "messages": list(messages),
            **self.extra_body,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def request(self, messages: Sequence[dict[str, Any]]) -> ChatReply:
        body, delay = self.body(messages), self.retry_delay_s
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post(f"{self.endpoint}/v1/chat/completions", json=body)
                if r.status_code == 429 or r.status_code >= 500:  # rate limit / server error: back off
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
        r.raise_for_status()
        data = r.json()
        if self.pinned is not None and data.get("model") != self.pinned:
            raise JudgeVersionError(f"model served {data.get('model')!r}, pinned {self.pinned!r}")
        choice = data["choices"][0]
        return ChatReply(
            (choice.get("message") or {}).get("content") or "", choice.get("finish_reason"), data.get("model")
        )


def endpoint_from_config(jcfg: dict[str, Any] | None, cls: type = ChatEndpoint, **kw: Any) -> Any:
    """``cls`` built from an eval-suite judge block, or None when it is unset or unusable from here.

    A hosted model that this machine cannot use (``require_api_key`` and no key; ``probe`` and the
    endpoint unreachable, e.g. a compute node without internet) is skipped with a warning, not an
    error: the caller leaves its results pending and a later run where it works fills them in.
    """
    jcfg = jcfg or {}
    if not (jcfg.get("endpoint") and jcfg.get("model")):
        return None
    key_env = jcfg.get("api_key_env") or ""
    key = os.environ.get(key_env) if key_env else None
    if jcfg.get("require_api_key") and not key:
        log.warning("%s skipped: no API key in $%s", jcfg["model"], key_env or "<api_key_env unset>")
        return None
    if jcfg.get("probe", False):
        try:
            httpx.get(host_only(str(jcfg["endpoint"])) + "/v1/models", timeout=15.0,
                      headers={"Authorization": f"Bearer {key}"} if key else {})
        except httpx.TransportError as e:
            log.warning("%s skipped: %s unreachable from here (%s)", jcfg["model"], jcfg["endpoint"], e)
            return None
    temp = jcfg.get("temperature", 0.0)
    return cls(
        str(jcfg["endpoint"]),
        str(jcfg["model"]),
        api_key=key,
        pinned_model_version=jcfg.get("pinned_model_version"),
        timeout_s=float(jcfg.get("timeout_s", 300)),
        max_tokens=int(jcfg.get("max_tokens", 2048)),
        temperature=None if temp is None else float(temp),
        max_tokens_field=str(jcfg.get("max_tokens_field", "max_tokens")),
        extra_body=jcfg.get("extra_body") or None,
        retries=int(jcfg.get("retries", 4)),
        **kw,
    )
