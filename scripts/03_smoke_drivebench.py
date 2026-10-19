"""Pilot step 3: run N DriveBench questions through the running server.

Writes two files per (precision, condition):
  results/pilot/<precision>_<condition>.jsonl            our per-question records (resumable)
  results/pilot/<precision>_<condition>.drivebench.json  upstream format (original fields + "pred")
                                                          for evaluate/eval.py

    python scripts/03_smoke_drivebench.py --precision bf16
    python scripts/03_smoke_drivebench.py --precision int4
    python scripts/03_smoke_drivebench.py --precision bf16 --condition LowLight
    python scripts/03_smoke_drivebench.py --precision bf16 --n 0   # all questions
    python scripts/03_smoke_drivebench.py --precision bf16 --n 0 --split final   # full run on final scenes only

Question selection is deterministic (seeded), stratified so every eligible cell
appears, and identical across precisions and conditions.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.client import make_client  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.gpu import PeakMemorySampler  # noqa: E402
from qflip.records import append_record, load_done_ids, read_records  # noqa: E402
from qflip.runner import read_system_prompt, run_question, select_stratified  # noqa: E402
from qflip.split import filter_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "int4"], required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--n", type=int, default=None, help="questions to run; 0 = all; default from config")
    ap.add_argument("--split", default="all", help="scene split to draw from: all, calibration, development, final")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.condition not in cfg["data"]["conditions"]:
        sys.exit(f"condition {args.condition!r} not in config; known: {list(cfg['data']['conditions'])}")
    model = cfg["models"][args.precision]
    client = make_client(cfg)
    n = cfg["pilot"]["n_questions"] if args.n is None else args.n

    questions = select_stratified(filter_split(load_questions(cfg["data"]["annotations"]), args.split, args.split_file), n)
    system_prompt = read_system_prompt(cfg)
    out_dir = Path(cfg["pilot"]["results_dir"])
    jsonl = out_dir / f"{args.precision}_{args.condition}.jsonl"
    done = load_done_ids(jsonl)
    todo = [q for q in questions if q.question_id not in done]
    print(f"{len(questions)} selected, {len(done)} done, {len(todo)} to run -> {jsonl}")

    with PeakMemorySampler() as mem:
        for q in tqdm(todo):
            rec = run_question(client, model, q, cfg, args.condition, args.precision, system_prompt)
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
    acc_rel = sum(r["mcq_correct_released"] for r in mcq) / len(mcq) if mcq else None
    no_margin = sum(1 for r in mcq if r["margin"] is None)
    lat = sorted(r["latency_s"] for r in recs if not r["error"])
    print("\nSUMMARY")
    print(f"  questions: {len(recs)}  errors: {errors}")
    print(f"  cells: {dict(sorted(((c, sum(1 for r in recs if r['cell'] == c)) for c in {r['cell'] for r in recs})))}")
    print(f"  MCQ accuracy strict={acc}  released-rule={acc_rel}  ({len(mcq)} MCQ, {no_margin} without option margin)")
    if lat:
        print(f"  latency s p50={lat[len(lat)//2]:.2f} p95={lat[int(len(lat)*0.95)-1 if len(lat) > 1 else 0]:.2f}")
    print(f"  prompt tokens mean: {sum(r['prompt_tokens'] or 0 for r in recs)/max(len(recs),1):.0f}")
    print(f"  GPU peak MiB: {mem.peak_mib} (baseline {mem.baseline_mib})")
    print(f"\n  wrote {jsonl}\n  wrote {db_out}")
    print(f"\n  upstream scorer (MCQ accuracy only, no GPT):\n    cd <DriveBench repo> && PYTHONPATH=. python evaluate/eval.py {db_out.resolve()}")
    print("    (needs data/visual_description.json and the language_evaluation package in that repo)")


if __name__ == "__main__":
    main()
