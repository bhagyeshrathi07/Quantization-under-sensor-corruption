"""Thin client for a running vLLM OpenAI-compatible server."""

import math
import time
from dataclasses import dataclass

from openai import OpenAI

from qflip.provenance import effective_base_url

OPTION_LETTERS = ("A", "B", "C", "D")


@dataclass
class StreamedCompletion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_s: float | None      # time to first content token
    latency_s: float          # time to final chunk
    error: str | None = None

    @property
    def decode_tok_s(self) -> float | None:
        """Tokens per second after the first token; None when fewer than two tokens."""
        if self.error or self.ttft_s is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / max(self.latency_s - self.ttft_s, 1e-6)


@dataclass
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    first_token_top_logprobs: list[dict] | None  # [{"token": str, "logprob": float}, ...]
    error: str | None = None


def make_client(cfg: dict) -> OpenAI:
    s = cfg["server"]
    # QFLIP_BASE_URL overrides the config so a batch job can pick a free port;
    # the run guard checks the same endpoint (qflip.provenance.effective_base_url).
    return OpenAI(base_url=effective_base_url(cfg), api_key=s["api_key"], timeout=s["timeout_s"])


def chat(client: OpenAI, model: str, messages: list[dict], gen: dict) -> Completion:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=gen["temperature"],
            max_tokens=gen["max_tokens"],
            seed=gen.get("seed"),
            logprobs=True,
            top_logprobs=gen["top_logprobs"],
        )
    except Exception as e:  # noqa: BLE001 - record and continue
        return Completion("", None, None, time.perf_counter() - t0, None, error=repr(e))
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    top = None
    if choice.logprobs and choice.logprobs.content:
        first = choice.logprobs.content[0]
        top = [{"token": t.token, "logprob": t.logprob} for t in (first.top_logprobs or [])]
    usage = resp.usage
    return Completion(
        text=choice.message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        latency_s=latency,
        first_token_top_logprobs=top,
    )


def option_margin(top_logprobs: list[dict] | None) -> tuple[str | None, float | None, dict]:
    """From the first generated token's top-k logprobs, collect probability mass
    per option letter and return (top_letter, p_top1 - p_top2, per-letter probs).

    Returns (None, None, {}) when no option letter appears in the top-k, which
    means the answer did not start with a letter; the caller records that as
    margin-unavailable rather than guessing.
    """
    if not top_logprobs:
        return None, None, {}
    probs: dict[str, float] = {}
    for entry in top_logprobs:
        letter = entry["token"].strip().strip(".:)").upper()
        if letter in OPTION_LETTERS:
            probs[letter] = probs.get(letter, 0.0) + math.exp(entry["logprob"])
    if not probs:
        return None, None, {}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_letter, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return top_letter, p1 - p2, probs


def consume_stream(chunks, t0: float) -> StreamedCompletion:
    """Walk a streaming chat response; returns timings. Split out from
    chat_stream so it can be tested without a server."""
    text, ttft, usage = [], None, None
    for chunk in chunks:
        now = time.perf_counter()
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if ttft is None:
                    ttft = now - t0
                text.append(delta.content)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    latency = time.perf_counter() - t0
    return StreamedCompletion(
        text="".join(text),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        ttft_s=ttft,
        latency_s=latency,
    )


def chat_stream(client: OpenAI, model: str, messages: list[dict], gen: dict) -> StreamedCompletion:
    """Same request as chat() but streamed, for time-to-first-token. No logprobs."""
    t0 = time.perf_counter()
    try:
        chunks = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=gen["temperature"],
            max_tokens=gen["max_tokens"],
            seed=gen.get("seed"),
            stream=True,
            stream_options={"include_usage": True},
        )
        return consume_stream(chunks, t0)
    except Exception as e:  # noqa: BLE001
        return StreamedCompletion("", None, None, None, time.perf_counter() - t0, error=repr(e))
