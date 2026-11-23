"""The disagreement list for one human labeller: their labels against the two
model labellers' labels, with the models' one-sentence reasons, for the
discussion among the three humans.

    python scripts/16_disagreements.py --labeller ronak --sheet manifests/judge_label_sheet_2026-11-17.csv --models fable astra

Refuses until the labeller has labelled every item of their assignment (so
nobody sees a model label before their own label on that item is given), and
reads the labeller's copy as it is on disk (commit it first). Reads the model
copies and their `results/dev_vqa512/llm_<model>/labels.json` records. Writes
`results/dev_vqa512/disagreements_<labeller>.md` and prints a summary. The
models' labels never become a consensus by themselves; the three humans decide
the consensus in discussion (manifests/judge_decisions_2026-11-13.md,
November 19).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import CalibrationError, label_column, load_assignment, parse_label  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]


def copy_path(sheet: Path, name: str) -> Path:
    return sheet.with_name(f"{sheet.stem}.{name}{sheet.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--models", nargs="+", default=["fable", "astra"])
    ap.add_argument("--assignment", help="default <sheet stem>.assignment.json; 'none' for every item")
    ap.add_argument("--records-dir", default="results/dev_vqa512", help="where llm_<model>/labels.json live")
    ap.add_argument("--out", help="default <records-dir>/disagreements_<labeller>.md")
    ap.add_argument("--allow-partial", action="store_true", help="tests only: report on the labelled prefix even if the assignment is unfinished")
    args = ap.parse_args()
    sheet = Path(args.sheet)
    master = read_rows(sheet)
    human = read_rows(copy_path(sheet, args.labeller))
    if [r["item"] for r in human] != [r["item"] for r in master]:
        sys.exit(f"{copy_path(sheet, args.labeller)}: items differ from the master sheet")
    assignment_path = sheet.with_name(f"{sheet.stem}.assignment.json") if args.assignment is None else (None if args.assignment == "none" else Path(args.assignment))
    if assignment_path is not None and not assignment_path.exists():
        assignment_path = None
    try:
        mine = load_assignment(assignment_path, master, [args.labeller], require_all=False)[args.labeller] if assignment_path else [r["item"] for r in master]
        hl = {r["item"]: parse_label(r.get(label_column(args.labeller)), f"{args.labeller} {r['item']}") for r in human}
    except CalibrationError as e:
        sys.exit(str(e))
    unlabelled = [it for it in mine if hl[it] is None]
    if unlabelled and not args.allow_partial:
        sys.exit(f"{args.labeller} has {len(unlabelled)} assigned item(s) without a label (first {unlabelled[0]}); the disagreement list is produced only after the whole assignment is labelled")
    # and every other block too: the list shows model labels, and no human may still be labelling (review of November 23)
    if assignment_path and not args.allow_partial:
        everyone = load_assignment(assignment_path, master, [c[len("label_"):] for c in master[0] if c.startswith("label_")])
        for other, items in everyone.items():
            if other == args.labeller:
                continue
            try:
                orows = read_rows(copy_path(sheet, other))
            except FileNotFoundError:
                sys.exit(f"{copy_path(sheet, other)} missing; every labeller's block must be complete before any disagreement list is produced")
            ol = {r["item"]: (r.get(label_column(other)) or "").strip() for r in orows}
            missing = [it for it in items if not ol.get(it)]
            if missing:
                sys.exit(f"{other} has {len(missing)} assigned item(s) without a label (first {missing[0]}); the disagreement list is produced only after every block is labelled")
    done = [it for it in mine if hl[it] is not None]
    models = {}
    for m in args.models:
        rows = read_rows(copy_path(sheet, m))
        if [r["item"] for r in rows] != [r["item"] for r in master]:
            sys.exit(f"{copy_path(sheet, m)}: items differ from the master sheet")
        rec_path = Path(args.records_dir) / f"llm_{m}" / "labels.json"
        reasons = json.loads(rec_path.read_text())["items"] if rec_path.exists() else {}
        models[m] = {r["item"]: (parse_label(r.get(label_column(m)), f"{m} {r['item']}"), reasons.get(r["item"], {})) for r in rows}
    text_of = {r["item"]: r for r in master}
    lines = [f"# Disagreements for {args.labeller} ({len(done)} labelled item(s) of {len(mine)} assigned)", "",
             f"Human label against the model labels ({', '.join(args.models)}). The humans decide the consensus in discussion; a model label is a second opinion, never a consensus by itself.", ""]
    n_dis = 0
    stats = Counter()
    for it in done:
        ml = {m: models[m][it][0] for m in args.models}
        agree = all(v == hl[it] for v in ml.values())
        for m, v in ml.items():
            stats[f"{m}_agree"] += int(v == hl[it])
        if agree:
            continue
        n_dis += 1
        r = text_of[it]
        lines += [f"## {it} ({r['cell']}): {args.labeller} {hl[it]}, " + ", ".join(f"{m} {ml[m] if ml[m] is not None else '-'}" for m in args.models), "",
                  f"**Question:** {r['question']}", "", f"**Reference:** {r['ground_truth']}", ""]
        for m in args.models:
            rec = models[m][it][1]
            if rec:
                lines.append(f"- {m} ({rec.get('label')}{', flagged ambiguous' if rec.get('ambiguous') else ''}): {rec.get('reason', '')}")
        lines += ["", "**Consensus:** _____ (reason: _____)", ""]
    lines.insert(2, f"Disagreements: {n_dis} of {len(done)}. Agreement with {args.labeller}: " + ", ".join(f"{m} {stats[f'{m}_agree']}/{len(done)}" for m in args.models) + ".")
    out = Path(args.out) if args.out else Path(args.records_dir) / f"disagreements_{args.labeller}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}: {n_dis} disagreement(s) of {len(done)}; agreement with {args.labeller}: " + ", ".join(f"{m} {stats[f'{m}_agree']}/{len(done)}" for m in args.models))


if __name__ == "__main__":
    main()
