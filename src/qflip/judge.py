"""VQA rubric judge, benchmark-compatible.

Reproduces what the released DriveBench evaluator does with `--eval-gpt`
(evaluate/eval.py + request.py), with the parts that make it unusable as
shipped replaced:

- The released request.py calls `call_chatgpt(prompt, max_tokens=512)` but
  `call_chatgpt` takes no such argument, so every call raises and the retry
  loop never ends. Here the request is made with the openai client.
- The judge model is hard-coded to gpt-3.5-turbo upstream; here it comes
  from config (`judge.model`) and is written into every record.
- A reply without "Total Score:" raises upstream; here it is recorded as an
  error so a rerun can retry it.

What is kept verbatim: the prompt templates (judge_prompts.py), the system
line, temperature 0, the visual-description lookup, and the score regex.
The API key comes from OPENAI_API_KEY only; it is never read from a file.
"""

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from qflip import judge_prompts

# Released toolkit file, not part of the HF dataset. Downloaded on demand.
VISUAL_DESCRIPTION_URL = "https://raw.githubusercontent.com/drive-bench/toolkit/main/data/visual_description.json"
VISUAL_DESCRIPTION_SHA256 = "4cf48342f2a325144ed1153516ad63f70c8bd2a3bc3be9af2afc22698271fd58"  # main, Oct 20, 2026

# upstream request.py prepare_chatgpt_message
SYSTEM_LINE = "You are an evaluator who rates answers based on their closeness to the correct answer."

# upstream eval.py forward(): cell -> prompt template
PROMPT_BY_CELL = {
    "perception/MCQ": ("PERCEPTION_MCQ_PROMPT", judge_prompts.PERCEPTION_MCQ_PROMPT),
    "perception/VQA": ("PERCEPTION_VQA_PROMPT", judge_prompts.PERCEPTION_VQA_PROMPT),
    "prediction/VQA": ("PREDICTION_VQA_PROMPT", judge_prompts.PREDICTION_VQA_PROMPT),
    "planning/VQA": ("PLANNING_VQA_PROMPT", judge_prompts.PLANNING_VQA_PROMPT),
    "behavior/MCQ": ("BEHAVIOR_MCQ_PROMPT", judge_prompts.BEHAVIOR_MCQ_PROMPT),
}
VQA_CELLS = ("perception/VQA", "prediction/VQA", "planning/VQA")

# upstream eval.py extract_visual_description workaround
_NO_DESC_QUESTION = "In this scenario, what are safe actions to take for the ego vehicle?"
_NO_DESC_TEXT = "No visual description needed for this question."
_SCORE_RE = re.compile(r"Total Score:\s*(\d{1,3})\b")


def ensure_visual_description(path: str | Path) -> Path:
    """Download data/visual_description.json from the released toolkit if absent."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[judge] fetching {VISUAL_DESCRIPTION_URL} -> {p}")
        with urllib.request.urlopen(VISUAL_DESCRIPTION_URL, timeout=60) as r:  # noqa: S310 - fixed https URL
            p.write_bytes(r.read())
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != VISUAL_DESCRIPTION_SHA256:
        print(f"[judge] WARNING {p} sha256 {digest[:12]} differs from the recorded {VISUAL_DESCRIPTION_SHA256[:12]}")
    return p


def load_visual_description(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def visual_description(desc_data: dict, scene_token: str, frame_token: str, question: str) -> str | None:
    """Upstream extract_visual_description, unchanged in behavior: the object
    tag `<c1,CAM_FRONT,x,y>` in the question selects the key object by its
    `<c1,CAM_FRONT` prefix; no tag -> None; tag without a match -> ValueError."""
    if question == _NO_DESC_QUESTION:
        return _NO_DESC_TEXT
    key_object = desc_data[scene_token]["key_frames"][frame_token]["key_object_infos"]
    m = re.search(r"<(.*)>", question)
    if not m:
        return None
    prefix = ",".join(m.group(1).split(",")[:2])
    key = next((k for k in key_object if k.startswith(f"<{prefix}")), None)
    if key is None:
        raise ValueError(f"No matching object found for prefix: {prefix}")
    return key_object[key]["Visual_description"].lower()


def prompt_template(cell: str) -> tuple[str, str]:
    """(template name, template text) for a cell; KeyError for cells the released scorer does not judge."""
    return PROMPT_BY_CELL[cell]


def template_sha(cell: str) -> str:
    return hashlib.sha1(prompt_template(cell)[1].encode()).hexdigest()[:12]


def build_prompt(cell: str, question: str, gt: str, pred: str, desc: str | None) -> str:
    """Upstream request.py forward(): the template is filled with DESC only when
    a description exists; templates without a {DESC} placeholder ignore it."""
    _, template = prompt_template(cell)
    if not desc:
        return template.format(GT=gt, PRED=pred, QUESTION=question)
    return template.format(GT=gt, PRED=pred, QUESTION=question, DESC=desc)


def needs_description(cell: str) -> bool:
    return "{DESC}" in prompt_template(cell)[1]


def extract_score(reply: str) -> int | None:
    """Upstream extract_answer regex; None instead of raising when absent."""
    m = _SCORE_RE.search(reply or "")
    return int(m.group(1)) if m else None


def judge_messages(prompt: str) -> list[dict]:
    """Upstream sends system line and prompt joined by a newline as ONE user message."""
    return [{"role": "user", "content": "\n".join([SYSTEM_LINE, prompt])}]


def request_judge(client, model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None, retries: int = 5) -> dict:
    """One judge call. Retries transport errors with backoff; never loops forever."""
    err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=judge_messages(prompt),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            reply = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            return {
                "reply": reply,
                "score": extract_score(reply),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "latency_s": round(time.perf_counter() - t0, 3),
                "error": None if extract_score(reply) is not None else "no Total Score in reply",
            }
        except Exception as e:  # noqa: BLE001 - record, back off, retry
            err = repr(e)
            time.sleep(min(2**attempt, 30))
    return {"reply": "", "score": None, "prompt_tokens": None, "completion_tokens": None, "latency_s": None, "error": err}


def load_scores(path: str | Path) -> dict[str, dict]:
    """Judge sidecar -> {question_id: record}; last record wins; errored ones are
    kept so the caller can see them but are not treated as scored."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["question_id"]] = rec
    return out


def scored_ids(scores: dict[str, dict]) -> set[str]:
    return {qid for qid, r in scores.items() if r.get("score") is not None}


def sidecar_path(results_file: str | Path) -> Path:
    """results/<run>/bf16_clean.jsonl -> results/<run>/bf16_clean.judge.jsonl"""
    p = Path(results_file)
    return p.with_name(p.stem + ".judge.jsonl")
