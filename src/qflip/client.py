"""Thin client for a running vLLM OpenAI-compatible server."""

import math
import time
from dataclasses import dataclass

from openai import OpenAI

OPTION_LETTERS = ("A", "B", "C", "D")


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
    return OpenAI(base_url=s["base_url"], api_key=s["api_key"], timeout=s["timeout_s"])


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
