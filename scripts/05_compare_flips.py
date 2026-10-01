"""Pair two result files (same condition, different precisions) and classify
MCQ flips. Pilot-grade: counts only, no bootstrap yet.

    python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.records import read_records  # noqa: E402


def classify(ref: dict, quant: dict) -> str:
    if ref["mcq_correct"] and not quant["mcq_correct"]:
        return "harmful"
    if not ref["mcq_correct"] and quant["mcq_correct"]:
        return "beneficial"
    if ref["mcq_pred"] != quant["mcq_pred"]:
        return "neutral_change"
    return "unchanged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("quantized")
    args = ap.parse_args()

    ref = {r["question_id"]: r for r in read_records(args.reference) if r.get("cell", "").endswith("MCQ")}
    qnt = {r["question_id"]: r for r in read_records(args.quantized) if r.get("cell", "").endswith("MCQ")}
    common = sorted(set(ref) & set(qnt))
    if not common:
        sys.exit("no common MCQ questions")

    classes = Counter()
    by_cell = Counter()
    low_margin = Counter()
    for qid in common:
        c = classify(ref[qid], qnt[qid])
        classes[c] += 1
        by_cell[(ref[qid]["cell"], c)] += 1
        m = ref[qid].get("margin")
        if m is not None:
            low_margin[("low" if m < 0.5 else "high", c)] += 1

    ref_correct = sum(ref[q]["mcq_correct"] for q in common)
    print(f"paired MCQ questions: {len(common)}")
    print(f"reference accuracy: {ref_correct/len(common):.3f}   quantized accuracy: {sum(qnt[q]['mcq_correct'] for q in common)/len(common):.3f}")
    print(f"flips: {dict(classes)}")
    if ref_correct:
        print(f"harmful-flip rate F_h = {classes['harmful']}/{ref_correct} = {classes['harmful']/ref_correct:.3f}")
    print("\nby cell:")
    for (cell, c), n in sorted(by_cell.items()):
        print(f"  {cell:16s} {c:15s} {n}")
    print("\nby BF16 margin (<0.5 vs >=0.5; pilot bin, freeze real bins on dev scenes):")
    for (b, c), n in sorted(low_margin.items()):
        print(f"  {b:5s} {c:15s} {n}")
    print("\nharmful examples:")
    for qid in common:
        if classify(ref[qid], qnt[qid]) == "harmful":
            print(f"  {qid} {ref[qid]['cell']} gt={ref[qid]['mcq_gt']} bf16={ref[qid]['mcq_pred']} int4={qnt[qid]['mcq_pred']} margin={ref[qid].get('margin')}")


if __name__ == "__main__":
    main()
