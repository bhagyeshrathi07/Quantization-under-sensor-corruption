"""The VQA evidence tables for the report, as markdown, from the consensus
sheet, its key and the calibration report. Formats only; no number is
computed anywhere else than from these files.

    python scripts/17_vqa_tables.py --sheet manifests/judge_label_sheet_2026-11-17.csv \\
        --key results/dev_vqa512/label_sheet_key.json --calibration results/dev_vqa512/judge_calibration.json \\
        --models fable astra --out results/dev_vqa512/vqa_tables.md

V1: human-adjudicated correctness of the clean development answers per cell
and precision (the consensus column; one human labeller per item, model
labellers as the second opinion, see manifests/judge_calibration_2026-11-22.md).
V2: paired outcome per question (BF16 against INT4 on the same question): both
correct, harmful (BF16 correct, INT4 not), beneficial, both incorrect; no
interval (131 questions on 26 scenes, descriptive).
V3: the judge calibration summary and verdict from the report JSON.
V4: agreement of the labellers (pre-discussion human, each model, the judge at
the fitted threshold) with the final consensus; not an independent validation.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import label_column, parse_label  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--calibration", help="JSON written by 11 --out")
    ap.add_argument("--assignment", help="default <sheet stem>.assignment.json beside the sheet")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    sheet = Path(args.sheet)
    rows = read_rows(sheet)
    key = json.loads(Path(args.key).read_text())["items"]
    cons = {}
    for r in rows:
        c = parse_label(r.get("consensus"), f"{r['item']} consensus")
        if c is None:
            sys.exit(f"{sheet}: {r['item']} has no consensus; the tables are built from a complete sheet")
        cons[r["item"]] = c
    missing = [it for it in cons if it not in key]
    if missing:
        sys.exit(f"{args.key}: {len(missing)} sheet item(s) not in the key")
    cell = {it: key[it]["cell"] for it in cons}
    prec = {it: key[it]["precision"] for it in cons}
    qid = {it: key[it]["question_id"] for it in cons}
    out = []
    # V1
    out += ["### V1. Human-adjudicated correctness of the clean development VQA answers (consensus labels)", "",
            f"Source: `{sheet}` (consensus column), key `{args.key}`. One human labeller per item, two model labellers as the second opinion, disagreements settled by the three humans; the criterion is `manifests/labelling_guide.md`. Development scenes, clean condition, 512-token amendment; no claim about the final scenes or the corrupted conditions.", "",
            "| cell | BF16 | INT4 | both |", "|---|---|---|---|"]
    cells = sorted(set(cell.values()))
    tot = Counter()
    for c in cells + ["all"]:
        items = [it for it in cons if c == "all" or cell[it] == c]
        parts = []
        for p in ("bf16", "int4"):
            mine = [it for it in items if prec[it] == p]
            parts.append(pct(sum(cons[it] for it in mine), len(mine)))
        parts.append(pct(sum(cons[it] for it in items), len(items)))
        out.append(f"| {c} | " + " | ".join(parts) + " |")
    out.append("")
    # V2
    by_q = defaultdict(dict)
    for it in cons:
        by_q[qid[it]][prec[it]] = cons[it]
    out += ["### V2. Paired outcome per development question (BF16 against INT4)", "",
            "Descriptive: 131 questions on 26 development scenes, no interval. Harmful = BF16 correct and INT4 incorrect; beneficial = the reverse.", "",
            "| cell | questions | both correct | harmful | beneficial | both incorrect |", "|---|---|---|---|---|---|"]
    for c in cells + ["all"]:
        qs = [q for q, d in by_q.items() if len(d) == 2 and (c == "all" or cell[next(it for it in cons if qid[it] == q)] == c)]
        k = Counter()
        for q in qs:
            b, i = by_q[q]["bf16"], by_q[q]["int4"]
            k["both correct" if b and i else "harmful" if b else "beneficial" if i else "both incorrect"] += 1
        out.append(f"| {c} | {len(qs)} | {k['both correct']} | {k['harmful']} | {k['beneficial']} | {k['both incorrect']} |")
    out.append("")
    # V3
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        c = cal.get("calibration", cal)
        fit = c.get("fit", {})
        loso = c.get("loso", {})
        held = loso.get("pooled", {})
        folds = sorted({f.get("threshold") for f in loso.get("folds", []) if f.get("threshold") is not None}) or loso.get("fold_thresholds", [])
        bars = c.get("bars", {})
        out += ["### V3. Judge calibration (decision 4, pre-registered)", "",
                f"Source: `{args.calibration}`. Verdict: **{'ACCEPTED' if c.get('accepted') else 'NOT ACCEPTED'}**" + (f"; reasons: {'; '.join(c.get('reasons', []))}" if c.get("reasons") else "") + ".", "",
                "| | fit on all items | leave-one-scene-out, pooled held-out | bar |", "|---|---|---|---|"]
        def g(d, k):
            v = d.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "threshold" else (str(v) if v is not None else "")
        for label, k, bar in (("threshold", "threshold", ""), ("agreement", "agreement", bars.get("agreement")), ("Cohen's kappa", "kappa", bars.get("kappa")),
                              ("recall, human-correct", "recall_correct", bars.get("recall")), ("recall, human-incorrect", "recall_incorrect", bars.get("recall"))):
            out.append(f"| {label} | {g(fit, k)} | {g(held, k)} | {bar if bar is not None else ''} |")
        out.append(f"| items (human-correct / human-incorrect) | {c.get('n_items', '')} ({fit.get('n_human_correct', '')} / {fit.get('n_human_incorrect', '')}) | | {bars.get('min_items', '')} ({bars.get('min_per_class', '')} per class) |")
        if folds:
            out.append(f"| fold thresholds | | {', '.join(str(t) for t in folds)} | |")
        out.append("")
        out.append(f"Held-out: leave-one-scene-out over {loso.get('n_scenes', '')} scenes, predictions pooled; the fit of each fold chooses its own threshold (fold thresholds {', '.join(str(t) for t in folds)}).")
        out.append("")
    # V4
    if args.models:
        human = {}
        assignment_path = Path(args.assignment) if args.assignment else sheet.with_name(f"{sheet.stem}.assignment.json")
        who = {}
        if assignment_path.exists():
            for n, items in json.loads(assignment_path.read_text())["assignment"].items():
                for it in items:
                    who[it] = n
        for r in rows:
            n = who.get(r["item"])
            names = [n] if n else [c[len("label_"):] for c in r if c.startswith("label_") and (r[c] or "").strip()]
            labs = [parse_label(r.get(label_column(x)), f"{r['item']} {x}") for x in names]
            labs = [l for l in labs if l is not None]
            human[r["item"]] = labs[0] if len(labs) == 1 else None
        cols = {"human (pre-discussion)": human}
        for m in args.models:
            mrows = read_rows(sheet.with_name(f"{sheet.stem}.{m}{sheet.suffix}"))
            cols[m] = {r["item"]: parse_label(r.get(label_column(m)), f"{m} {r['item']}") for r in mrows}
        out += ["### V4. Labellers against the final consensus (not an independent validation)", "",
                "The consensus inherits the human label on the agreed items and was settled with the models' reasons in view on the others; these rows measure agreement with that assisted consensus.", "",
                "| labeller | agrees | of | says correct |", "|---|---|---|---|"]
        for name, lab in cols.items():
            items = [it for it in cons if lab.get(it) is not None]
            out.append(f"| {name} | {sum(1 for it in items if lab[it] == cons[it])} | {len(items)} | {sum(lab[it] for it in items)} |")
        out.append(f"| consensus | {len(cons)} | {len(cons)} | {sum(cons.values())} |")
        out.append("")
    md = "\n".join(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
