"""Pair reference (BF16) and quantized (INT4) result files, classify MCQ
flips, and report point estimates with paired bootstrap CIs over scenes.

One pair:
    python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl

Several conditions at once (ref quant ref quant ...), which enables --by condition:
    python scripts/05_compare_flips.py \\
        results/full/bf16_clean.jsonl results/full/int4_clean.jsonl \\
        results/full/bf16_Fog.jsonl   results/full/int4_Fog.jsonl --by cell n_views margin condition

Flip classes (strict rule: option letter at the start of the answer):
  harmful        reference correct, quantized wrong
  beneficial     reference wrong, quantized correct
  neutral_change answer changed, both right or both wrong
  unchanged      same letter

F_h (harmful_rate) = harmful / reference-correct. CIs are percentile intervals
from resampling scenes with replacement (default 2000 draws, seed 0). VQA cells
are not scored here; they need the judge.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.flips import bootstrap_scenes, group_by, pair_records, summarize  # noqa: E402
from qflip.records import read_records  # noqa: E402


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def report(label: str, pairs, n_boot: int, seed: int, min_n: int) -> dict:
    s = summarize(pairs)
    if s["n"] < min_n:
        print(f"  {label:24s} n={s['n']} (below --min-n, skipped)")
        return {"label": label, **s}
    ci = bootstrap_scenes(pairs, n_boot=n_boot, seed=seed)
    hr = f"{s['harmful_rate']:.3f}" if s["harmful_rate"] is not None else "n/a"
    print(
        f"  {label:24s} n={s['n']:4d} scenes={s['scenes']:3d}  acc {s['ref_acc']:.3f}->{s['quant_acc']:.3f} "
        f"d={s['acc_delta']:+.3f} {fmt_ci(ci.get('acc_delta'))}  "
        f"harm={s['harmful']} ben={s['beneficial']} neut={s['neutral_change']}  "
        f"F_h={hr} {fmt_ci(ci.get('harmful_rate'))}  change={s['change_rate']:.3f} {fmt_ci(ci.get('change_rate'))}"
    )
    return {"label": label, **s, "ci": ci}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="ref quant [ref quant ...]")
    ap.add_argument("--by", nargs="*", default=["cell", "n_views", "margin"], choices=["cell", "n_views", "margin", "condition"])
    ap.add_argument("--margin-edges", default="0.25,0.5", help="bin edges for the BF16 option margin; freeze on dev scenes")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=10, help="skip CIs for groups smaller than this")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--show-harmful", type=int, default=20, help="print up to N harmful examples")
    args = ap.parse_args()
    if len(args.files) % 2:
        sys.exit("files must come in ref/quant pairs")
    edges = tuple(float(x) for x in args.margin_edges.split(",") if x)

    pairs = []
    for ref_path, quant_path in zip(args.files[::2], args.files[1::2]):
        p = pair_records(read_records(ref_path), read_records(quant_path))
        print(f"{ref_path} vs {quant_path}: {len(p)} paired MCQ")
        pairs.extend(p)
    if not pairs:
        sys.exit("no common MCQ questions")

    out = {"margin_edges": edges, "n_boot": args.n_boot, "seed": args.seed, "groups": {}}
    print("\noverall:")
    out["overall"] = report("all", pairs, args.n_boot, args.seed, 1)
    for factor in args.by:
        print(f"\nby {factor}:")
        out["groups"][factor] = [
            report(name, grp, args.n_boot, args.seed, args.min_n) for name, grp in group_by(pairs, factor, edges).items()
        ]

    harmful = [p for p in pairs if p.flip == "harmful"]
    if harmful and args.show_harmful:
        print(f"\nharmful examples ({len(harmful)} total, showing {min(len(harmful), args.show_harmful)}):")
        for p in harmful[: args.show_harmful]:
            m = f"{p.margin:.2f}" if p.margin is not None else "n/a"
            print(f"  {p.question_id} {p.condition:9s} {p.cell:15s} views={p.n_views} margin={m}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=list))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
