"""Pilot step 3: run N DriveBench questions through the running server.

Writes two files per (precision, condition):
  results/pilot/<precision>_<condition>.jsonl            our per-question records (resumable)
  results/pilot/<precision>_<condition>.drivebench.json  upstream format (original fields + "pred")
                                                          for evaluate/eval.py

    python scripts/03_smoke_drivebench.py --precision bf16
    python scripts/03_smoke_drivebench.py --precision int4
    python scripts/03_smoke_drivebench.py --precision bf16 --condition LowLight
    python scripts/03_smoke_drivebench.py --precision bf16 --n 0   # all questions

Question selection is deterministic (seeded), stratified so every eligible cell
appears, and identical across precisions and conditions.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import chat, make_client, option_margin  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import build_messages, load_questions, normalize_mcq, normalize_mcq_strict  # noqa: E402
from qflip.gpu import PeakMemorySampler  # noqa: E402
from qflip.records import append_record, load_done_ids, read_records  # noqa: E402


def select_pilot_questions(questions, n, seed=0):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "int4"], required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--n", type=int, default=None, help="questions to run; 0 = all; default from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.condition not in cfg["data"]["conditions"]:
        sys.exit(f"condition {args.condition!r} not in config; known: {list(cfg['data']['conditions'])}")
    model = cfg["models"][args.precision]
    client = make_client(cfg)
    n = cfg["pilot"]["n_questions"] if args.n is None else args.n

    questions = select_pilot_questions(load_questions(cfg["data"]["annotations"]), n)
    system_prompt = Path(cfg["prompt"]["system_file"]).read_text()
    out_dir = Path(cfg["pilot"]["results_dir"])
    jsonl = out_dir / f"{args.precision}_{args.condition}.jsonl"
    done = load_done_ids(jsonl)
    todo = [q for q in questions if q.question_id not in done]
    print(f"{len(questions)} selected, {len(done)} done, {len(todo)} to run -> {jsonl}")

    with PeakMemorySampler() as mem:
        for q in tqdm(todo):
            messages = build_messages(
                q,
                system_prompt,
                args.condition,
                cfg["data"]["conditions"][args.condition],
                cfg["data"]["root"],
                cfg["images"]["resize_to"],
                tuple(cfg["images"]["noimage_size"]),
                cfg["prompt"]["mcq_suffix"],
            )
            c = chat(client, model, messages, cfg["generation"])
            rec = {
                "question_id": q.question_id,
                "scene_token": q.scene_token,
                "frame_token": q.frame_token,
                "question_type": q.question_type,
                "tag": q.tag,
                "cell": q.cell,
                "views": q.views,
                "n_views": q.n_views,
                "condition": args.condition,
                "precision": args.precision,
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
                rec.update(
                    {
                        "mcq_pred": normalize_mcq(c.text),
                        "mcq_pred_strict": normalize_mcq_strict(c.text),
                        "mcq_gt": normalize_mcq(q.answer),
                        "mcq_correct": normalize_mcq(c.text) == normalize_mcq(q.answer) and normalize_mcq(c.text) != "",
                        "margin_letter": letter,
                        "margin": margin,
                        "option_probs": probs,
                    }
                )
            append_record(jsonl, rec)

    # upstream-format file for evaluate/eval.py
    records = {r["question_id"]: r for r in read_records(jsonl)}
    upstream = []
    for q in questions:
        if q.question_id in records:
            item = dict(q.raw)
            item["image_path"] = q.image_path
            item["pred"] = records[q.question_id]["pred"]
            upstream.append(item)
    db_out = out_dir / f"{args.precision}_{args.condition}.drivebench.json"
    db_out.write_text(json.dumps(upstream, indent=2))

    # summary
    recs = list(records.values())
    errors = sum(1 for r in recs if r["error"])
    mcq = [r for r in recs if r["cell"] and r["cell"].endswith("MCQ")]
    acc = sum(r["mcq_correct"] for r in mcq) / len(mcq) if mcq else None
    no_margin = sum(1 for r in mcq if r["margin"] is None)
    lat = sorted(r["latency_s"] for r in recs if not r["error"])
    print("\nSUMMARY")
    print(f"  questions: {len(recs)}  errors: {errors}")
    print(f"  cells: {dict(sorted(((c, sum(1 for r in recs if r['cell'] == c)) for c in {r['cell'] for r in recs})))}")
    print(f"  MCQ accuracy (released rule): {acc}  ({len(mcq)} MCQ, {no_margin} without option margin)")
    if lat:
        print(f"  latency s p50={lat[len(lat)//2]:.2f} p95={lat[int(len(lat)*0.95)-1 if len(lat) > 1 else 0]:.2f}")
    print(f"  prompt tokens mean: {sum(r['prompt_tokens'] or 0 for r in recs)/max(len(recs),1):.0f}")
    print(f"  GPU peak MiB: {mem.peak_mib} (baseline {mem.baseline_mib})")
    print(f"\n  wrote {jsonl}\n  wrote {db_out}")
    print(f"\n  upstream scorer (MCQ accuracy only, no GPT):\n    cd <DriveBench repo> && PYTHONPATH=. python evaluate/eval.py {db_out.resolve()}")
    print("    (needs data/visual_description.json and the language_evaluation package in that repo)")


if __name__ == "__main__":
    main()
