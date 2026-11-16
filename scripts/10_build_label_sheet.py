"""Build the blind label sheet for the judge calibration, its key, and one
copy per labeller.

    python scripts/10_build_label_sheet.py results/dev/bf16_clean.jsonl results/dev/int4_clean.jsonl \
        --labellers ronak bhagyesh akshay \
        --sheet manifests/judge_label_sheet_2026-11-14.csv --key results/dev/label_sheet_key.json --copies

Inputs must be development-split answers produced under the core freeze
(qflip.guard.prepare_calibration_files: sidecars, population, `freeze_checked`
equal to the freeze, sidecar settings equal to the frozen ones); `--no-provenance`
is for sidecar-less synthetic files (tests) and marks the output exploratory.

One item per VQA answer of every input file (both precisions are items), in a
shuffled order fixed by --seed. The sheet is blind (decision 3 of
manifests/judge_decisions_2026-11-13.md): cell, question, ground truth and
answer, nothing else; no precision, no question id, no scene, no judge score or
text. The master sheet has one `label_<name>` column per labeller (1 correct,
0 incorrect, by the rubric's correctness criterion), a `consensus` column and a
`consensus_reason` column, all empty. `--copies` writes one copy per labeller
(`<sheet>.<name>.csv`, the blind columns plus that labeller's column) so each
person labels independently; scripts/12_merge_label_copies.py merges them into
the master sheet. The key (item -> question id, precision, scene, cell, answer
digest, digest of the displayed text; plus the sheet order) is written to a
separate file and is not opened until the label columns and the consensus are
complete (scripts/11_calibrate_judge.py). Existing sheet, copies and key are
never overwritten: labels live in them.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import BLIND_COLUMNS, FORBIDDEN_SHEET_COLUMNS, display_sha256, label_column, sheet_columns  # noqa: E402
from qflip.config import load_config  # noqa: E402
from qflip.freeze import FreezeError  # noqa: E402
from qflip.guard import prepare_calibration_files  # noqa: E402
from qflip.judge import VQA_CELLS  # noqa: E402
from qflip.provenance import ProvenanceError  # noqa: E402
from qflip.records import read_records  # noqa: E402


def answer_digest(pred: str) -> str:
    return hashlib.sha256(pred.encode("utf-8")).hexdigest()


def copy_path(sheet: Path, name: str) -> Path:
    return sheet.with_name(f"{sheet.stem}.{name}{sheet.suffix}")


def check_inputs(files: list[str]) -> None:
    for f in files:
        if f.endswith(".judge.jsonl") or f.endswith(".meta.json"):
            sys.exit(f"{f} is a sidecar, not a result file; name the raw answer files")
    if len(set(Path(f).resolve() for f in files)) != len(files):
        sys.exit("the same file is given more than once")


def collect_items(files: list[str]) -> list[dict]:
    """Every non-errored VQA answer of every file, with the fields the sheet
    and the key need. Files must share one condition and carry distinct
    precisions, so the sheet is one population answered once per precision."""
    items, conditions, precisions = [], set(), []
    for f in files:
        recs = read_records(f)
        if not recs:
            sys.exit(f"{f}: no records")
        precs = {r.get("precision") for r in recs}
        conds = {r.get("condition") for r in recs}
        if len(precs) != 1 or len(conds) != 1:
            sys.exit(f"{f}: records carry {len(precs)} precision(s) and {len(conds)} condition(s); one of each expected")
        precisions.append(precs.pop())
        conditions |= conds
        for r in recs:
            if r.get("cell") not in VQA_CELLS or r.get("error"):
                continue
            if not r.get("pred") or not r.get("question") or r.get("gt") is None:
                sys.exit(f"{f}: VQA record {r.get('question_id')} lacks question, ground truth or answer")
            items.append({"file": f, "question_id": r["question_id"], "precision": r["precision"], "scene_token": r["scene_token"], "cell": r["cell"],
                          "question": r["question"], "gt": r["gt"], "pred": r["pred"]})
    if len(conditions) != 1:
        sys.exit(f"files span {len(conditions)} conditions ({sorted(conditions)}); one condition expected")
    if len(set(precisions)) != len(precisions):
        sys.exit(f"two files carry the same precision ({precisions}); one file per precision expected")
    if not items:
        sys.exit("no VQA answers in the given files")
    return items


def build(files: list[str], labellers: list[str], seed: int, prefix: str = "D") -> tuple[list[dict], dict]:
    """Rows of the blind sheet (in sheet order) and the key."""
    items = collect_items(files)
    items.sort(key=lambda it: (it["question_id"], it["precision"]))  # order before the shuffle is fixed by content, not file order
    rng = random.Random(seed)
    rng.shuffle(items)
    width = max(3, len(str(len(items))))
    rows, key_items = [], {}
    for i, it in enumerate(items, 1):
        item_id = f"{prefix}{i:0{width}d}"
        row = {"item": item_id, "cell": it["cell"], "question": it["question"], "ground_truth": it["gt"], "answer": it["pred"]}
        for n in labellers:
            row[label_column(n)] = ""
        row["consensus"] = row["consensus_reason"] = ""
        rows.append(row)
        key_items[item_id] = {"question_id": it["question_id"], "precision": it["precision"], "scene_token": it["scene_token"], "cell": it["cell"],
                              "pred_sha256": answer_digest(it["pred"]), "display_sha256": display_sha256(it["cell"], it["question"], it["gt"], it["pred"])}
    key = {"created": date.today().isoformat(), "source": list(files), "seed": seed, "labellers": list(labellers), "n_items": len(rows),
           "order": [r["item"] for r in rows],
           "blind": "the sheet carries cell, question, ground truth and answer only; precision, question id, scene and judge scores are in this key, which stays closed until every label column and the consensus column are complete",
           "items": key_items}
    return rows, key


def write_csv(rows: list[dict], cols: list[str], path: Path) -> None:
    for c in cols:
        assert not any(bad in c for bad in FORBIDDEN_SHEET_COLUMNS), c
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        w.writeheader()
        w.writerows([{c: row.get(c, "") for c in cols} for row in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="raw answer files (one per precision, same condition), e.g. results/dev/bf16_clean.jsonl results/dev/int4_clean.jsonl")
    ap.add_argument("--labellers", nargs="+", required=True, help="one label column per name, e.g. ronak bhagyesh akshay")
    ap.add_argument("--sheet", required=True, help="output master CSV (never overwritten)")
    ap.add_argument("--key", required=True, help="output key JSON (never overwritten)")
    ap.add_argument("--copies", action="store_true", help="also write one blind copy per labeller next to the sheet (<sheet>.<name>.csv)")
    ap.add_argument("--seed", type=int, default=20261113, help="shuffle seed (default: the decision date)")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--no-provenance", action="store_true", help="sidecar-less synthetic files only (tests); the output is exploratory and never labelled for the study")
    args = ap.parse_args()
    check_inputs(args.files)
    if len(args.labellers) % 2 == 0:
        sys.exit("an odd number of labellers is required (a majority label must always exist)")
    if len(set(args.labellers)) != len(args.labellers):
        sys.exit("labeller names must be distinct")
    try:
        prov = prepare_calibration_files(args.files, load_config(args.config), no_provenance=args.no_provenance, split_file=args.split_file, exclusions_file=args.exclusions_file)
    except (ProvenanceError, FreezeError) as e:
        sys.exit(f"inputs refused: {e}")
    if prov.get("note"):
        print(prov["note"])
    sheet, key_path = Path(args.sheet), Path(args.key)
    outputs = [sheet, key_path] + ([copy_path(sheet, n) for n in args.labellers] if args.copies else [])
    for p in outputs:
        if p.exists():
            sys.exit(f"{p} exists; sheets, copies and keys are never overwritten (labels live in them). Remove or rename it yourself if a rebuild is wanted.")
    rows, key = build(args.files, args.labellers, args.seed)
    key["sheet"] = str(sheet)
    key["provenance"] = {"analysis_class": prov["analysis_class"], "split": prov["split"], "frozen": prov.get("frozen"), "note": prov.get("note")}
    sheet.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, sheet_columns(args.labellers), sheet)
    if args.copies:
        for n in args.labellers:
            write_csv(rows, [*BLIND_COLUMNS, label_column(n)], copy_path(sheet, n))
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n")
    cells = Counter(r["cell"] for r in rows)
    print(f"{sheet}: {len(rows)} items ({prov['analysis_class']}), columns {sheet_columns(args.labellers)}")
    print("  cells: " + ", ".join(f"{c} {n}" for c, n in sorted(cells.items())))
    if args.copies:
        print("  copies: " + ", ".join(str(copy_path(sheet, n)) for n in args.labellers))
    print(f"{key_path}: key for {len(rows)} items (closed until the label and consensus columns are complete)")


if __name__ == "__main__":
    main()
