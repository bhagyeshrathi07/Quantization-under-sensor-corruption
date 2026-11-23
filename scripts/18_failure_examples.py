"""Failure examples for the report: harmful MCQ flips (BF16 correct, INT4
incorrect) from 05's JSON, with the question text, the options, both answers,
the BF16 option probabilities and margin, and the image files of the views.

    python scripts/18_failure_examples.py --compare results/final/compare.json --results results/final \\
        --limit 12 --out results/final/failure_examples.md --json results/final/failure_examples.json

Selection is deterministic: lowest BF16 margin first (the pre-registered
contrast says those flip most), at most --per-cell per cell and --per-condition
per condition, so the examples are not all one kind. Image paths come from the
annotations (the views the question used, clean condition; corrupted views are
the same file names under the corruption folder). Images present under
--images-root are listed as available; the JSON names the missing files so
they can be fetched from the host.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.records import read_records  # noqa: E402

OPTION_RE = re.compile(r"(?:^|\s)([A-D])\.\s")


def split_options(question: str) -> tuple[str, dict]:
    """The stem and {letter: text} of a DriveBench MCQ (options follow 'Please select ... options:')."""
    head, sep, tail = question.partition("options:")
    if not sep:
        return question.strip(), {}
    parts = re.split(r"\s(?=[A-D]\.\s)", " " + tail.strip())
    opts = {}
    for p in parts:
        p = p.strip()
        m = re.match(r"([A-D])\.\s*(.*)$", p, re.S)
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return head.strip(), opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="JSON written by 05 --out (harmful_examples)")
    ap.add_argument("--results", required=True, help="directory of the result files the comparison was made from")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--images-root", default="data", help="where the annotation image paths resolve")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--per-condition", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--json")
    args = ap.parse_args()
    comp = json.loads(Path(args.compare).read_text())
    harmful = comp.get("harmful_examples") or []
    if not harmful:
        sys.exit(f"{args.compare}: no harmful_examples")
    cfg = load_config(args.config)
    questions = {q.question_id: q for q in load_questions(cfg["data"]["annotations"])}
    recs = {}
    for f in sorted(Path(args.results).glob("*_*.jsonl")):
        if f.name.endswith(".judge.jsonl"):
            continue
        for r in read_records(f):
            recs[(r["precision"], r["condition"], r["question_id"])] = r
    ranked = sorted(harmful, key=lambda h: (h["margin"] if h.get("margin") is not None else 9, h["question_id"]))
    chosen, per_cell, per_cond = [], Counter(), Counter()
    for h in ranked:
        if per_cell[h["cell"]] >= args.per_cell or per_cond[h["condition"]] >= args.per_condition:
            continue
        chosen.append(h)
        per_cell[h["cell"]] += 1
        per_cond[h["condition"]] += 1
        if len(chosen) >= args.limit:
            break
    md = [f"# Failure examples: harmful MCQ flips (BF16 correct, INT4 incorrect), {len(chosen)} of {len(harmful)}", "",
          f"Source: `{args.compare}` and the raw answers in `{args.results}`. Lowest BF16 margin first, at most {args.per_cell} per cell and {args.per_condition} per condition. The margin is BF16's probability gap between its top two options at the first token; the pre-registered contrast C1 found flips concentrated in the lowest margin bin.", ""]
    records_out = []
    for n, h in enumerate(chosen, 1):
        qid, cond = h["question_id"], h["condition"]
        q = questions.get(qid)
        ref, quant = recs.get(("bf16", cond, qid)), recs.get(("int4", cond, qid))
        if q is None or ref is None or quant is None:
            sys.exit(f"{qid} ({cond}): question or answers not found")
        stem, opts = split_options(q.question)
        probs = ref.get("option_probs") or {}
        views = q.image_path
        img_paths = {}
        for cam, rel in views.items():
            p = rel if cond == "clean" else rel.replace("nuscenes/samples", f"corruption/{cond}") if cond != "NoImage" else None
            img_paths[cam] = p
        present = {cam: (p is not None and (Path(args.images_root) / p).exists()) for cam, p in img_paths.items()}
        md += [f"## {n}. {h['cell']}, {cond}, margin {h['margin']:.3f}", "",
               f"Question `{qid}`, scene `{h['scene_token'][:8]}`, {h['n_views']} view(s).", "",
               f"**{stem}**", ""]
        for letter in sorted(opts):
            mark = " (correct)" if letter == ref.get("mcq_gt") else ""
            md.append(f"- {letter}. {opts[letter]}{mark}")
        md += ["", f"BF16 answered **{ref.get('mcq_pred_strict')}** (option probabilities at the first token: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(probs.items())) + f"); INT4 answered **{quant.get('mcq_pred_strict')}**.", "",
               "BF16 text: " + (ref.get("pred") or "").strip().replace("\n", " ")[:300], "",
               "INT4 text: " + (quant.get("pred") or "").strip().replace("\n", " ")[:300], "",
               "Views: " + ", ".join(f"{cam} ({'present' if present[cam] else 'missing'})" for cam in views) if cond != "NoImage" else "Views: none shown (NoImage control)", ""]
        records_out.append({"rank": n, "question_id": qid, "scene_token": h["scene_token"], "condition": cond, "cell": h["cell"], "margin": h["margin"], "correct": ref.get("mcq_gt"),
                            "bf16": ref.get("mcq_pred_strict"), "int4": quant.get("mcq_pred_strict"), "option_probs_bf16": probs, "stem": stem, "options": opts,
                            "images": img_paths, "images_present": present})
    missing = sorted({p for r in records_out for cam, p in r["images"].items() if p and not r["images_present"][cam]})
    md += [f"Images missing locally: {len(missing)} (listed in the JSON; fetch from the host with rsync before rendering).", ""]
    text = "\n".join(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}: {len(chosen)} examples")
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps({"source": args.compare, "n_harmful": len(harmful), "examples": records_out, "missing_images": missing}, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.json}: {len(missing)} image file(s) missing locally")


if __name__ == "__main__":
    main()
