"""Merge the labellers' independent copies into the master label sheet.

    python scripts/12_merge_label_copies.py --sheet manifests/judge_label_sheet_2026-11-14.csv \
        manifests/judge_label_sheet_2026-11-14.ronak.csv manifests/judge_label_sheet_2026-11-14.bhagyesh.csv manifests/judge_label_sheet_2026-11-14.akshay.csv

Each copy (written by 10 --copies) carries the blind columns and one
`label_<name>` column. A copy must hold the master's items in the master's
order with identical displayed text (cell, question, ground truth, answer);
its labels must be 0, 1 or empty. The master's `label_<name>` column takes the
copy's values; a master cell that already holds a different value is a
conflict and the merge is refused. The consensus columns are never touched.
The master sheet is rewritten in place; nothing else is written.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import BLIND_COLUMNS, CalibrationError, parse_label  # noqa: E402


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    return rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("copies", nargs="+", help="one labelled copy per labeller")
    ap.add_argument("--sheet", required=True, help="the master sheet (rewritten in place)")
    args = ap.parse_args()
    master_path = Path(args.sheet)
    master, mcols = read_csv(master_path)
    for c in BLIND_COLUMNS:
        if c not in mcols:
            sys.exit(f"{master_path}: column {c} missing")
    filled = {}
    for cp in args.copies:
        rows, cols = read_csv(Path(cp))
        label_cols = [c for c in cols if c.startswith("label_")]
        if len(label_cols) != 1:
            sys.exit(f"{cp}: a copy carries exactly one label_<name> column, found {label_cols}")
        col = label_cols[0]
        if col not in mcols:
            sys.exit(f"{cp}: column {col} is not on the master sheet (have {[c for c in mcols if c.startswith('label_')]})")
        if col in filled:
            sys.exit(f"{cp}: a second copy for {col} ({filled[col]['copy']} already given)")
        if [r["item"] for r in rows] != [r["item"] for r in master]:
            sys.exit(f"{cp}: items or their order differ from the master sheet")
        for mr, r in zip(master, rows):
            for c in BLIND_COLUMNS[1:]:
                if (mr.get(c) or "") != (r.get(c) or ""):
                    sys.exit(f"{cp}: item {r['item']}: {c} differs from the master sheet; the copy was edited or built from other answers")
        n_set, conflicts = 0, []
        for mr, r in zip(master, rows):
            try:
                v = parse_label(r.get(col), f"{cp} {r['item']}")
                old = parse_label(mr.get(col), f"{master_path} {mr['item']}")
            except CalibrationError as e:
                sys.exit(str(e))
            if v is None:
                continue
            if old is not None and old != v:
                conflicts.append(mr["item"])
                continue
            mr[col] = str(v)
            n_set += 1
        if conflicts:
            sys.exit(f"{cp}: {len(conflicts)} item(s) already carry a different {col} on the master sheet ({conflicts[:5]}); nothing written")
        filled[col] = {"copy": cp, "labels": n_set, "blank": sum(1 for r in rows if parse_label(r.get(col), "") is None)}
    with master_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=mcols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in mcols} for r in master])
    for col, info in filled.items():
        print(f"{master_path}: {col} <- {info['copy']}: {info['labels']} label(s), {info['blank']} blank")


if __name__ == "__main__":
    main()
