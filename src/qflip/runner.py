"""Shared per-question execution: build the prompt, query the server, and
produce the result record that 03/06 write. Kept in one place so stability
reruns and the main runs produce byte-compatible records."""

import random
from collections import defaultdict
from pathlib import Path

from qflip.client import chat, option_margin
from qflip.drivebench import Question, build_messages, normalize_mcq, normalize_mcq_strict


def select_stratified(questions: list[Question], n: int, seed: int = 0) -> list[Question]:
    """Deterministic round-robin over cells so every eligible cell appears.
    n <= 0 returns every eligible question."""
    eligible = [q for q in questions if q.cell is not None]
    if n <= 0:
        return eligible
    by_cell = defaultdict(list)
    for q in eligible:
        by_cell[q.cell].append(q)
    rng = random.Random(seed)
    cells = sorted(by_cell)
    for c in cells:
        rng.shuffle(by_cell[c])
    picked, i = [], 0
    while len(picked) < n and any(by_cell.values()):
        c = cells[i % len(cells)]
        if by_cell[c]:
            picked.append(by_cell[c].pop())
        i += 1
    return picked


def messages_for(q: Question, cfg: dict, condition: str, system_prompt: str) -> list[dict]:
    return build_messages(
        q,
        system_prompt,
        condition,
        cfg["data"]["conditions"][condition],
        cfg["data"]["root"],
        cfg["images"]["resize_to"],
        tuple(cfg["images"]["noimage_size"]),
        cfg["prompt"]["mcq_suffix"],
    )


def run_question(client, model: str, q: Question, cfg: dict, condition: str, precision: str, system_prompt: str) -> dict:
    c = chat(client, model, messages_for(q, cfg, condition, system_prompt), cfg["generation"])
    rec = {
        "question_id": q.question_id,
        "scene_token": q.scene_token,
        "frame_token": q.frame_token,
        "question_type": q.question_type,
        "tag": q.tag,
        "cell": q.cell,
        "views": q.views,
        "n_views": q.n_views,
        "condition": condition,
        "precision": precision,
        "model": model,
        "gt": q.answer,
        "pred": c.text,
        "error": c.error,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "latency_s": round(c.latency_s, 4),
        "first_token_top_logprobs": c.first_token_top_logprobs,
    }
    if q.is_mcq:
        letter, margin, probs = option_margin(c.first_token_top_logprobs)
        strict = normalize_mcq_strict(c.text)
        released = normalize_mcq(c.text)
        gt = normalize_mcq(q.answer)
        rec.update(
            {
                "mcq_pred": released,
                "mcq_pred_strict": strict,
                "mcq_gt": gt,
                # strict = option letter at the start of the answer (primary);
                # released = DriveBench evaluate/utils.py rule, which also matches the article "a"
                "mcq_correct": strict == gt and strict != "",
                "mcq_correct_released": released == gt and released != "",
                "margin_letter": letter,
                "margin": margin,
                "option_probs": probs,
            }
        )
    return rec


def read_system_prompt(cfg: dict) -> str:
    return Path(cfg["prompt"]["system_file"]).read_text()
