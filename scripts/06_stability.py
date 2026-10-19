"""Nondeterminism floor: rerun a fixed question slice R times against the same
server and count how often the model disagrees with itself. Flip rates between
precisions are only meaningful above this floor.

    python scripts/06_stability.py --precision bf16 --condition clean          # run 3 repeats of 60 dev-scene questions
    python scripts/06_stability.py --precision int4 --condition clean
    python scripts/06_stability.py --precision bf16 --condition clean --report  # summarize existing repeats

Writes results/stability/<precision>_<condition>_rep<k>.jsonl (same record
format as 03, resumable) and a <precision>_<condition>.summary.json.

Notes: temperature 0, fixed seed, --max-num-seqs 1. Remaining variation comes
from non-deterministic kernels and from vLLM prefix caching, which reuses the
cached prompt KV on repeats 2 and 3 (the server's default; repeats 2+ are then
closer to a warm production path than repeat 1). Start the server with
--no-enable-prefix-caching to measure kernels alone.
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import make_client  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.records import append_record, load_done_ids, read_records  # noqa: E402
from qflip.runner import read_system_prompt, run_question, select_stratified  # noqa: E402
from qflip.split import filter_split  # noqa: E402


def stability_summary(reps: list[list[dict]]) -> dict:
    """reps: one record list per repeat. Compares every pair of repeats on the
    MCQ questions present in all of them."""
    by_rep = [{r["question_id"]: r for r in rep if (r.get("cell") or "").endswith("MCQ") and not r.get("error")} for rep in reps]
    common = sorted(set.intersection(*(set(d) for d in by_rep))) if by_rep else []
    pairwise = []
    for i, j in combinations(range(len(by_rep)), 2):
        letter_diff = sum(by_rep[i][q]["mcq_pred_strict"] != by_rep[j][q]["mcq_pred_strict"] for q in common)
        correct_diff = sum(by_rep[i][q]["mcq_correct"] != by_rep[j][q]["mcq_correct"] for q in common)
        text_diff = sum(by_rep[i][q]["pred"] != by_rep[j][q]["pred"] for q in common)
        pairwise.append({"reps": (i + 1, j + 1), "n": len(common), "letter_changes": letter_diff,
                         "correctness_changes": correct_diff, "text_changes": text_diff})
    margins = []
    for q in common:
        ms = [by_rep[k][q].get("margin") for k in range(len(by_rep))]
        if all(m is not None for m in ms):
            margins.append(max(ms) - min(ms))
    n_pairs = max(len(pairwise), 1)
    all_recs = [r for rep in reps for r in rep]
    return {
        "repeats": len(reps),
        "mcq_common": len(common),
        "questions_per_repeat": [len(rep) for rep in reps],
        "errors": sum(1 for r in all_recs if r.get("error")),
        "pairwise": pairwise,
        # floor rates to compare against the BF16-vs-INT4 change_rate and acc delta
        "self_change_rate": sum(p["letter_changes"] for p in pairwise) / (n_pairs * max(len(common), 1)),
        "self_correctness_change_rate": sum(p["correctness_changes"] for p in pairwise) / (n_pairs * max(len(common), 1)),
        "self_text_change_rate": sum(p["text_changes"] for p in pairwise) / (n_pairs * max(len(common), 1)),
        "margin_range_mean": sum(margins) / len(margins) if margins else None,
        "margin_range_max": max(margins) if margins else None,
        "cells": dict(Counter(r["cell"] for r in reps[0])) if reps else {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "int4"], required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split", default="development")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--seed", type=int, default=1, help="slice seed; differs from 03's so the slice is not the pilot set")
    ap.add_argument("--out-dir", default="results/stability")
    ap.add_argument("--report", action="store_true", help="only summarize existing repeat files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.condition not in cfg["data"]["conditions"]:
        sys.exit(f"condition {args.condition!r} not in config")
    out_dir = Path(args.out_dir)
    files = [out_dir / f"{args.precision}_{args.condition}_rep{k}.jsonl" for k in range(1, args.repeats + 1)]

    if not args.report:
        model = cfg["models"][args.precision]
        client = make_client(cfg)
        questions = select_stratified(filter_split(load_questions(cfg["data"]["annotations"]), args.split, args.split_file), args.n, args.seed)
        system_prompt = read_system_prompt(cfg)
        print(f"{len(questions)} questions x {args.repeats} repeats, {args.precision} {args.condition}")
        for k, path in enumerate(files, 1):
            done = load_done_ids(path)
            todo = [q for q in questions if q.question_id not in done]
            for q in tqdm(todo, desc=f"rep{k}"):
                append_record(path, run_question(client, model, q, cfg, args.condition, args.precision, system_prompt))

    reps = [read_records(p) for p in files if p.exists()]
    if len(reps) < 2:
        sys.exit(f"need at least 2 repeat files under {out_dir}")
    summary = {"precision": args.precision, "condition": args.condition, "model": cfg["models"][args.precision], **stability_summary(reps)}
    out = out_dir / f"{args.precision}_{args.condition}.summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
