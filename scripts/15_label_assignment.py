"""Write the label assignment: who labels which items of the blind sheet.

    python scripts/15_label_assignment.py --sheet manifests/judge_label_sheet_2026-11-17.csv --labellers ronak bhagyesh akshay
    python scripts/15_label_assignment.py --sheet ... --labellers ronak bhagyesh akshay --sizes 88 87 87

Contiguous blocks in sheet order, one per labeller in the order given, equal
sizes with the remainder on the first blocks unless --sizes is given. Written
to `<sheet stem>.assignment.json` beside the sheet (never overwritten), which
scripts/13_label_ui.py and scripts/11_calibrate_judge.py pick up by default.
Decision of November 19, 2026 (manifests/judge_decisions_2026-11-13.md): one
human labeller per item, the two model labellers as the second opinion, the
disagreements discussed by all three humans.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import CalibrationError, label_column, load_assignment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--labellers", nargs="+", required=True, help="block order")
    ap.add_argument("--sizes", nargs="*", type=int, help="block sizes in that order (default: equal, remainder on the first blocks)")
    ap.add_argument("--out", help="default <sheet stem>.assignment.json beside the sheet")
    ap.add_argument("--note", default="", help="recorded in the file")
    args = ap.parse_args()
    sheet = Path(args.sheet)
    out = Path(args.out) if args.out else sheet.with_name(f"{sheet.stem}.assignment.json")
    if out.exists():
        sys.exit(f"{out} exists; an assignment is never overwritten (labels were given under it)")
    with sheet.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        items = [r["item"] for r in reader if (r.get("item") or "").strip()]
    missing = [n for n in args.labellers if label_column(n) not in cols]
    if missing:
        sys.exit(f"{sheet}: no label column for {missing}")
    if len(set(args.labellers)) != len(args.labellers):
        sys.exit("labeller names must be distinct")
    n = len(items)
    if args.sizes:
        if len(args.sizes) != len(args.labellers) or sum(args.sizes) != n or any(s <= 0 for s in args.sizes):
            sys.exit(f"--sizes must give one positive size per labeller summing to {n}")
        sizes = args.sizes
    else:
        q, r = divmod(n, len(args.labellers))
        sizes = [q + (1 if i < r else 0) for i in range(len(args.labellers))]
    assignment, start = {}, 0
    for name, size in zip(args.labellers, sizes):
        assignment[name] = items[start:start + size]
        start += size
    rows = [{"item": it} for it in items]
    data = {"sheet": str(sheet), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "contiguous blocks in sheet order, one human labeller per item; labelled in order within the block",
            "note": args.note, "sizes": dict(zip(args.labellers, sizes)), "assignment": assignment}
    out.write_text(json.dumps(data, indent=2) + "\n")
    try:
        load_assignment(out, rows, list(args.labellers))
    except CalibrationError as e:
        out.unlink()
        sys.exit(f"assignment invalid: {e}")
    print(f"{out}: " + ", ".join(f"{name} {assignment[name][0]} to {assignment[name][-1]} ({len(assignment[name])})" for name in args.labellers))


if __name__ == "__main__":
    main()
