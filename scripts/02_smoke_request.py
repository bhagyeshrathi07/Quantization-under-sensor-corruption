"""Pilot step 2: one six-view request against the running server.

Checks, in order: server reachable, six images accepted, prompt token count
(image tokens per view), option-letter logprobs present, peak GPU memory.

    python scripts/02_smoke_request.py --precision bf16
    python scripts/02_smoke_request.py --precision int4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import chat, make_client, option_margin  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import build_messages, load_questions  # noqa: E402
from qflip.gpu import PeakMemorySampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "int4"], required=True)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--condition", default="clean")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = cfg["models"][args.precision]
    client = make_client(cfg)

    models = [m.id for m in client.models.list().data]
    print(f"server models: {models}")
    if model not in models:
        sys.exit(f"served model {model!r} not found; check --served-model-name in 01_serve.sh")

    questions = load_questions(cfg["data"]["annotations"])
    # Prefer a six-view MCQ so both the multi-image path and the margin path are exercised.
    q = next((x for x in questions if x.is_mcq and x.n_views == 6), None) or next(x for x in questions if x.n_views == 6)
    print(f"question {q.question_id} cell={q.cell} views={q.n_views}")
    print(f"Q: {q.question[:200]}")

    system_prompt = Path(cfg["prompt"]["system_file"]).read_text()
    messages = build_messages(
        q,
        system_prompt,
        args.condition,
        cfg["data"]["conditions"].get(args.condition),
        cfg["data"]["root"],
        cfg["images"]["resize_to"],
        tuple(cfg["images"]["noimage_size"]),
        cfg["prompt"]["mcq_suffix"],
    )

    with PeakMemorySampler() as mem:
        # warm-up then measured
        chat(client, model, messages, cfg["generation"])
        c = chat(client, model, messages, cfg["generation"])

    if c.error:
        sys.exit(f"request failed: {c.error}")

    per_view = (c.prompt_tokens or 0) / max(q.n_views, 1)
    letter, margin, probs = option_margin(c.first_token_top_logprobs)
    report = {
        "precision": args.precision,
        "model": model,
        "condition": args.condition,
        "question_id": q.question_id,
        "n_views": q.n_views,
        "prompt_tokens": c.prompt_tokens,
        "approx_tokens_per_view_incl_text": round(per_view, 1),
        "completion_tokens": c.completion_tokens,
        "latency_s": round(c.latency_s, 3),
        "gpu_mem_baseline_mib": mem.baseline_mib,
        "gpu_mem_peak_mib": mem.peak_mib,
        "answer": c.text,
        "gt": q.answer,
        "first_token_top_logprobs": c.first_token_top_logprobs,
        "margin_letter": letter,
        "margin": margin,
        "option_probs": probs,
    }
    print(json.dumps(report, indent=2))

    out = Path(cfg["pilot"]["results_dir"]) / f"smoke_{args.precision}_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    print("\nGATE CHECKS")
    print(f"  six views accepted:        {'OK' if c.prompt_tokens else 'FAIL'}")
    print(f"  logprobs returned:         {'OK' if c.first_token_top_logprobs else 'FAIL (server started without logprob support?)'}")
    if q.is_mcq:
        print(f"  option letter in top-k:    {'OK' if letter else 'FAIL -> answer did not start with a letter; check mcq_suffix'}")
    print(f"  peak GPU memory (MiB):     {mem.peak_mib}")


if __name__ == "__main__":
    main()
