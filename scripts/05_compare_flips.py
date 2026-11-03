"""Pair reference (BF16) and quantized (INT4) result files, classify flips,
and report point estimates with paired bootstrap CIs over scenes.

One pair:
    python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl

Several conditions at once (ref quant ref quant ...), which enables --by condition:
    python scripts/05_compare_flips.py \\
        results/final/bf16_clean.jsonl results/final/int4_clean.jsonl \\
        results/final/bf16_Fog.jsonl   results/final/int4_Fog.jsonl --by cell n_views margin condition

With judged VQA cells (sidecars from scripts/08_judge_vqa.py next to each file):
    python scripts/05_compare_flips.py ... --vqa --judge-threshold 60

MCQ flip classes (strict rule: option letter at the start of the answer):
  harmful        reference correct, quantized wrong
  beneficial     reference wrong, quantized correct
  neutral_change answer changed, both right or both wrong
  unchanged      same letter
VQA pairs use the judge score against the threshold and have no neutral_change.

F_h (harmful_rate) = harmful / reference-correct. CIs are percentile intervals
from resampling scenes with replacement (default 2000 draws, seed 0). Groups
below the coverage minimum (--min-n questions or --min-scenes scenes, defaults
from config analysis:) are printed as unassessed and carry assessed=false in
the JSON report.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.flips import bootstrap_scenes, coverage, group_by, pair_records, pair_vqa_records, summarize  # noqa: E402
from qflip.judge import load_scores, sidecar_path  # noqa: E402
from qflip.records import read_records  # noqa: E402


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def report(label: str, pairs, n_boot: int, seed: int, min_n: int, min_scenes: int = 1) -> dict:
    s = summarize(pairs)
    assessed = coverage(s, min_n, min_scenes)
    if not assessed:
        print(f"  {label:24s} n={s.get('n', 0)} scenes={s.get('scenes', 0)}  UNASSESSED (below {min_n} questions or {min_scenes} scenes)")
        return {"label": label, **s, "assessed": False}
    ci = bootstrap_scenes(pairs, n_boot=n_boot, seed=seed)
    hr = f"{s['harmful_rate']:.3f}" if s["harmful_rate"] is not None else "n/a"
    score = ""
    if s.get("ref_score_mean") is not None:
        score = f"  score {s['ref_score_mean']:.1f}->{s['quant_score_mean']:.1f}"
    print(
        f"  {label:24s} n={s['n']:4d} scenes={s['scenes']:3d}  acc {s['ref_acc']:.3f}->{s['quant_acc']:.3f} "
        f"d={s['acc_delta']:+.3f} {fmt_ci(ci.get('acc_delta'))}  "
        f"harm={s['harmful']} ben={s['beneficial']} neut={s['neutral_change']}  "
        f"F_h={hr} {fmt_ci(ci.get('harmful_rate'))}  change={s['change_rate']:.3f} {fmt_ci(ci.get('change_rate'))}{score}"
    )
    return {"label": label, **s, "assessed": True, "ci": ci}


def load_vqa_pairs(ref_path: str, quant_path: str, threshold: float):
    ref_scores, quant_scores = load_scores(sidecar_path(ref_path)), load_scores(sidecar_path(quant_path))
    if not ref_scores or not quant_scores:
        print(f"  no judge sidecar for {ref_path} / {quant_path}; run scripts/08_judge_vqa.py first")
        return []
    return pair_vqa_records(read_records(ref_path), read_records(quant_path), ref_scores, quant_scores, threshold)


def section(title: str, pairs, by, edges, args, min_n, min_scenes) -> dict:
    out = {"groups": {}}
    print(f"\n{title}, overall:")
    out["overall"] = report("all", pairs, args.n_boot, args.seed, 1, 1)
    for factor in by:
        print(f"\n{title}, by {factor}:")
        out["groups"][factor] = [
            report(name, grp, args.n_boot, args.seed, min_n, min_scenes) for name, grp in group_by(pairs, factor, edges).items()
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="ref quant [ref quant ...]")
    ap.add_argument("--by", nargs="*", default=["cell", "n_views", "margin"], choices=["cell", "n_views", "margin", "condition"])
    ap.add_argument("--margin-edges", default="0.25,0.5", help="bin edges for the BF16 option margin; freeze on dev scenes")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--min-n", type=int, help="coverage minimum, questions per group (default analysis.min_questions)")
    ap.add_argument("--min-scenes", type=int, help="coverage minimum, scenes per group (default analysis.min_scenes)")
    ap.add_argument("--vqa", action="store_true", help="also pair judged VQA cells from the .judge.jsonl sidecars")
    ap.add_argument("--judge-threshold", type=float, help="rubric score counted as correct (default judge.correct_threshold)")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--show-harmful", type=int, default=20, help="print up to N harmful examples")
    args = ap.parse_args()
    if len(args.files) % 2:
        sys.exit("files must come in ref/quant pairs")
    edges = tuple(float(x) for x in args.margin_edges.split(",") if x)
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    an = cfg.get("analysis", {})
    min_n = args.min_n if args.min_n is not None else an.get("min_questions", 10)
    min_scenes = args.min_scenes if args.min_scenes is not None else an.get("min_scenes", 2)

    pairs = []
    for ref_path, quant_path in zip(args.files[::2], args.files[1::2]):
        p = pair_records(read_records(ref_path), read_records(quant_path))
        print(f"{ref_path} vs {quant_path}: {len(p)} paired MCQ")
        pairs.extend(p)
    if not pairs:
        sys.exit("no common MCQ questions")

    out = {"margin_edges": edges, "n_boot": args.n_boot, "seed": args.seed, "min_questions": min_n, "min_scenes": min_scenes}
    out["mcq"] = section("MCQ", pairs, args.by, edges, args, min_n, min_scenes)

    if args.vqa:
        threshold = args.judge_threshold if args.judge_threshold is not None else cfg.get("judge", {}).get("correct_threshold")
        if threshold is None:
            sys.exit("--vqa needs a threshold: --judge-threshold or judge.correct_threshold in config (TO DECIDE on development scenes)")
        vqa = []
        for ref_path, quant_path in zip(args.files[::2], args.files[1::2]):
            vqa.extend(load_vqa_pairs(ref_path, quant_path, threshold))
        print(f"\n{len(vqa)} paired VQA at threshold {threshold}")
        if vqa:
            out["vqa"] = {"threshold": threshold, **section("VQA (judged)", vqa, [f for f in args.by if f != "margin"], edges, args, min_n, min_scenes)}
            pairs = pairs + vqa

    harmful = [p for p in pairs if p.flip == "harmful"]
    if harmful and args.show_harmful:
        print(f"\nharmful examples ({len(harmful)} total, showing {min(len(harmful), args.show_harmful)}):")
        for p in harmful[: args.show_harmful]:
            m = f"{p.margin:.2f}" if p.margin is not None else "n/a"
            sc = f" score {p.ref_score:.0f}->{p.quant_score:.0f}" if p.ref_score is not None else ""
            print(f"  {p.question_id} {p.condition:9s} {p.cell:15s} views={p.n_views} margin={m}{sc}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=list))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
