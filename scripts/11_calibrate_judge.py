"""Calibrate the judge threshold against the human labels, by the rule of
manifests/judge_decisions_2026-11-13.md (decision 4), and report whether the
judge is accepted.

    # before the discussion: inter-labeller statistics from the label columns alone (no key, no scores)
    python scripts/11_calibrate_judge.py --labels-only --sheet manifests/judge_label_sheet_2026-11-17.csv --labellers ronak bhagyesh akshay

    # after the consensus column is complete
    python scripts/11_calibrate_judge.py results/dev_vqa512/bf16_clean.jsonl results/dev_vqa512/int4_clean.jsonl --amendment vqa512 \
        --sheet manifests/judge_label_sheet_2026-11-17.csv --key results/dev_vqa512/label_sheet_key.json \
        --out results/dev_vqa512/judge_calibration.json

The sheet is read and checked first; the key is opened only when every
labeller's labels are the sheet's first rows (items are labelled in sheet
order; a labelled row after a blank one is refused), every item all labellers
reached carries a consensus and no later item does, and the sheet holds exactly
the issued items in the issued order. The complete items are the rows every
labeller reached; the tail (time ran out for someone) is counted, not used, and
labels on it stay as they are. The answer files must
be development answers produced under the core freeze
(qflip.guard.prepare_calibration_files); `--no-provenance` (synthetic tests)
marks everything exploratory and never prints a freeze command. Each used item
must show the text its key digest was built from (cell, question, ground truth,
answer: on the sheet and in the current answer file) and needs the production
aggregate (mean of judge.production_replicates usable, distinct judgings).

Reported: pairwise agreement and Fleiss' kappa of the pre-discussion labels;
the agreement curve and the chosen threshold on all calibration items with
Cohen's kappa and the two error counts; the leave-one-scene-out held-out
metrics and the acceptance verdict with its reasons (compared as exact
fractions); the sensitivity rows (majority label; replicate 0 alone), which
never enter the verdict. Nothing here writes the judge freeze.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qflip.calibrate import (  # noqa: E402
    BLIND_COLUMNS,
    FORBIDDEN_SHEET_COLUMNS,
    CalibrationError,
    Item,
    acceptance_bars,
    calibrate,
    classify_sheet,
    display_sha256,
    fleiss_kappa,
    item_dict,
    label_column,
    majority_label,
    pairwise_agreement,
)
from qflip.config import load_config  # noqa: E402
from qflip.freeze import FreezeError  # noqa: E402
from qflip.freeze import AMENDMENTS, amendment  # noqa: E402
from qflip.guard import judge_settings_for, prepare_calibration_files  # noqa: E402
from qflip.judge import ensure_visual_description, load_scores, load_visual_description, needs_description, production_scores, sidecar_path  # noqa: E402
from qflip.provenance import ProvenanceError  # noqa: E402
from qflip.records import read_records  # noqa: E402


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    for bad in FORBIDDEN_SHEET_COLUMNS:
        if any(bad in c for c in cols):
            sys.exit(f"{path}: a column containing {bad!r} must not be on the blind sheet (have {cols})")
    missing = [c for c in BLIND_COLUMNS if c not in cols]
    if missing:
        sys.exit(f"{path}: blind column(s) {missing} missing; the sheet must show every issued field (have {cols})")
    return rows, cols


def labellers_of(cols: list[str], given: list[str] | None) -> list[str]:
    found = [c[len("label_"):] for c in cols if c.startswith("label_")]
    if given:
        missing = [n for n in given if label_column(n) not in cols]
        if missing:
            sys.exit(f"label columns missing for {missing} (have {cols})")
        return list(given)
    if not found:
        sys.exit(f"no label_<name> column on the sheet (have {cols})")
    return found


def inter_labeller(complete: dict, labellers: list[str]) -> dict:
    """Pre-discussion statistics over the complete items."""
    order = sorted(complete)
    labels_by = {n: [complete[i]["labels"][n] for i in order] for n in labellers}
    rows = [[complete[i]["labels"][n] for n in labellers] for i in order]
    out = {
        "n_items": len(order),
        "pairwise_agreement": pairwise_agreement(labels_by),
        "fleiss_kappa": fleiss_kappa(rows),
        "labelled_correct_per_labeller": {n: sum(labels_by[n]) for n in labellers},
        "majority_correct": sum(majority_label(r) for r in rows),
        "unanimous": sum(1 for r in rows if len(set(r)) == 1),
    }
    if all(complete[i]["consensus"] is not None for i in order):
        out["consensus_correct"] = sum(complete[i]["consensus"] for i in order)
        out["consensus_differs_from_majority"] = sum(1 for i in order if complete[i]["consensus"] != majority_label([complete[i]["labels"][n] for n in labellers]))
    return out


def print_inter(inter: dict) -> None:
    fk = inter["fleiss_kappa"]
    print("inter-labeller (pre-discussion): " + ", ".join(f"{k} {v:.3f}" for k, v in inter["pairwise_agreement"].items())
          + f"; Fleiss kappa {fk if fk is None else round(fk, 3)}; unanimous {inter['unanimous']}/{inter['n_items']}; "
          + "correct per labeller " + ", ".join(f"{n} {c}" for n, c in inter["labelled_correct_per_labeller"].items()) + f"; majority correct {inter['majority_correct']}"
          + (f"; consensus correct {inter['consensus_correct']}, differs from majority on {inter['consensus_differs_from_majority']}" if "consensus_correct" in inter else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="the raw answer files the sheet was built from (judge sidecars beside them)")
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", help="the key written by 10 (required unless --labels-only; opened only after the sheet passes its checks)")
    ap.add_argument("--labels-only", action="store_true", help="pre-discussion report from the label columns alone: no key, no scores, no answer files; the consensus columns are not read")
    ap.add_argument("--labellers", nargs="*", help="label columns to use (default: every label_<name> column; with --key the key's list)")
    ap.add_argument("--config", help="default config/pilot.yaml, or the amendment's config with --amendment")
    ap.add_argument("--desc", default="data/visual_description.json", help="object visual descriptions (judge input, hash-checked; loaded only when a planning item is present)")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--no-provenance", action="store_true", help="sidecar-less synthetic files only (tests); the result is exploratory and nothing is frozen from it")
    ap.add_argument("--amendment", choices=sorted(AMENDMENTS), help="answer files produced under this amendment: its config (unless --config is given) and its freeze")
    args = ap.parse_args()
    frozen_kw = {}
    if args.amendment:
        am = amendment(args.amendment)
        args.config = args.config or am["config"]
        frozen_kw = {"frozen_path": am["frozen"], "amendment": am["name"]}
    args.config = args.config or "config/pilot.yaml"

    # 1. the sheet alone
    rows, cols = read_rows(Path(args.sheet))
    labellers = labellers_of(cols, args.labellers)
    try:
        complete, unfinished, progress = classify_sheet(rows, labellers, need_consensus=not args.labels_only)
    except CalibrationError as e:
        sys.exit(f"{args.sheet}: {e}")
    what = "all label columns" if args.labels_only else f"all {len(labellers)} labels and consensus"
    print(f"sheet: {len(rows)} items; {len(complete)} complete ({what}), {len(unfinished)} unfinished; labelled per labeller: " + ", ".join(f"{n} {k}" for n, k in progress.items()))
    partial = max(progress.values()) - min(progress.values()) if progress else 0
    if partial:
        print(f"  {partial} tail item(s) carry labels from some labellers only (kept on the sheet, not used)")
    if not complete:
        sys.exit("no complete item; nothing to report")

    if args.labels_only:
        inter = inter_labeller(complete, labellers)
        print_inter(inter)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sheet": str(args.sheet), "labellers": labellers, "mode": "labels-only",
                                                  "counts": {"sheet_items": len(rows), "complete": len(complete), "unfinished": len(unfinished), "partial_tail": partial, "labelled_per_labeller": progress}, "inter_labeller": inter}, indent=2) + "\n")
            print(f"wrote {args.out}")
        return

    # 2. the answer files and their provenance
    if not args.files or not args.key:
        sys.exit("the answer files and --key are required (or --labels-only for the pre-discussion report)")
    for f in args.files:
        if f.endswith(".judge.jsonl") or f.endswith(".meta.json"):
            sys.exit(f"{f} is a sidecar, not a result file; name the raw answer files")
    cfg = load_config(args.config)
    try:
        bars = acceptance_bars(cfg.get("judge"))
    except CalibrationError as e:
        sys.exit(str(e))
    try:
        prov = prepare_calibration_files(args.files, cfg, no_provenance=args.no_provenance, split_file=args.split_file, exclusions_file=args.exclusions_file, **frozen_kw)
        judge = judge_settings_for(prov["split"], cfg)
    except (ProvenanceError, FreezeError) as e:
        sys.exit(f"inputs refused: {e}")
    exploratory = prov["analysis_class"] != "calibration"
    if prov.get("note"):
        print(prov["note"])
    settings, backend, replicates = judge["settings"], judge["backend"], judge["production_replicates"]

    # 3. the key, opened now: the sheet must be the issued one
    key = json.loads(Path(args.key).read_text())
    if list(key["labellers"]) != labellers:
        sys.exit(f"the key was issued for labellers {key['labellers']}, the sheet is read with {labellers}")
    want = sorted(str(Path(p).resolve()) for p in key["source"])
    have = sorted(str(Path(p).resolve()) for p in args.files)
    if want != have:
        sys.exit(f"the key was built from {key['source']}, not from the given files")
    issued = list(key.get("order") or key["items"])
    present = [(r.get("item") or "").strip() for r in rows]
    if present != issued:
        missing, extra = sorted(set(issued) - set(present)), sorted(set(present) - set(issued))
        sys.exit(f"the sheet does not hold the issued items in the issued order: {len(missing)} missing {missing[:5]}, {len(extra)} unknown {extra[:5]}"
                 + ("; order differs" if not missing and not extra else ""))
    by_row = {(r["item"] or "").strip(): r for r in rows}

    # 4. production aggregates per precision
    by_precision = {}
    cells_present = {key["items"][i]["cell"] for i in complete}
    desc_data = None
    if any(needs_description(c) for c in cells_present):
        desc_data = load_visual_description(ensure_visual_description(args.desc))
    for f in args.files:
        recs = read_records(f)
        precs = {r.get("precision") for r in recs}
        if len(precs) != 1:
            sys.exit(f"{f}: {len(precs)} precisions in one file")
        prec = precs.pop()
        scores = load_scores(sidecar_path(f))
        if not scores:
            sys.exit(f"{f}: no judge sidecar; run scripts/08_judge_vqa.py (three replicates) first")
        sel, stale = production_scores(scores, recs, settings, desc_data, None, backend, replicates)
        sel0, _ = production_scores(scores, recs, settings, desc_data, None, backend, 1)
        by_precision[prec] = {"records": {r["question_id"]: r for r in recs}, "selected": sel, "stale": stale, "rep0": sel0}
        print(f"  {f}: {len(sel)} answers with the mean of {replicates} judgings, {len(stale)} stale")

    # 5. join: each complete item must show the issued text, and the current record must be that text
    items, no_aggregate, shown_differs, record_differs = [], [], [], []
    for item_id, lab in complete.items():
        k = key["items"][item_id]
        row = by_row[item_id]
        if display_sha256(row["cell"], row["question"], row["ground_truth"], row["answer"]) != k["display_sha256"]:
            shown_differs.append(item_id)
            continue
        side = by_precision.get(k["precision"])
        if side is None:
            sys.exit(f"{item_id}: precision {k['precision']} not among the given files")
        rec = side["records"].get(k["question_id"])
        if rec is None or display_sha256(rec.get("cell"), rec.get("question"), rec.get("gt"), rec.get("pred")) != k["display_sha256"]:
            record_differs.append(item_id)
            continue
        agg = side["selected"].get(k["question_id"])
        if agg is None:
            no_aggregate.append((item_id, side["stale"].get(k["question_id"], "not judged")))
            continue
        labels = [lab["labels"][n] for n in labellers]
        rep0 = side["rep0"].get(k["question_id"])
        items.append(Item(item_id, rec["scene_token"], float(agg["score"]), lab["consensus"], majority_label(labels), float(rep0["score"]) if rep0 else None))
    if shown_differs:
        sys.exit(f"{len(shown_differs)} item(s) whose sheet text (cell, question, ground truth, answer) differs from the issued one ({shown_differs[:5]}); the labels were given for other text")
    if record_differs:
        sys.exit(f"{len(record_differs)} item(s) whose answer in the given files differs from the issued one ({record_differs[:5]}); the sheet was built from other answers")
    for item_id, why in no_aggregate[:5]:
        print(f"  no aggregate for {item_id}: {why}")
    print(f"calibration items: {len(items)} (complete and with the mean of {replicates} judgings); {len(no_aggregate)} complete item(s) without an aggregate")

    inter = inter_labeller(complete, labellers)
    print_inter(inter)
    try:
        result = calibrate(items, bars)
    except CalibrationError as e:
        sys.exit(f"calibration refused: {e}")
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_class": prov["analysis_class"],
        "provenance": prov,
        "judge_settings": settings,
        "judge_backend": backend,
        "production_replicates": replicates,
        "sheet": str(args.sheet),
        "key": str(args.key),
        "labellers": labellers,
        "counts": {"sheet_items": len(rows), "complete": len(complete), "unfinished": len(unfinished), "partial_tail": partial, "without_aggregate": len(no_aggregate), "calibration_items": len(items), "labelled_per_labeller": progress},
        "inter_labeller": inter,
        "calibration": result,
        "items": [{**item_dict(it), **{k: key["items"][it.item][k] for k in ("question_id", "precision", "cell")},
                   "scores": by_precision[key["items"][it.item]["precision"]]["selected"][key["items"][it.item]["question_id"]].get("scores"),
                   "labels": complete[it.item]["labels"], "consensus_reason": complete[it.item]["reason"]} for it in items],
    }
    if not result["calibrated"]:
        print("NO CALIBRATION: " + "; ".join(result["reasons"]))
    else:
        f_ = result["fit"]
        print(f"\nfit on all {result['n_items']} items ({result['n_scenes']} scenes): threshold {f_['threshold']}, agreement {f_['agreement']:.3f}, kappa {f_['kappa'] if f_['kappa'] is None else round(f_['kappa'], 3)}, "
              f"recall correct {f_['recall_correct']}, recall incorrect {f_['recall_incorrect']}; judge-correct/human-incorrect {f_['judge_correct_human_incorrect']}, human-correct/judge-incorrect {f_['human_correct_judge_incorrect']}; "
              f"human-correct {f_['n_human_correct']}, human-incorrect {f_['n_human_incorrect']}")
        p = result["loso"]["pooled"]
        print(f"leave-one-scene-out, pooled held-out: agreement {p['agreement']:.3f}, kappa {p['kappa'] if p['kappa'] is None else round(p['kappa'], 3)}, recall correct {p['recall_correct']}, recall incorrect {p['recall_incorrect']}; "
              f"fold thresholds {sorted({fd['threshold'] for fd in result['loso']['folds']})}")
        for name, s in result["sensitivity"].items():
            print(f"sensitivity {name}: threshold {s['fit']['threshold']}, fit agreement {s['fit']['agreement']:.3f}, held-out agreement {s['heldout']['agreement']:.3f}, held-out kappa {s['heldout']['kappa'] if s['heldout']['kappa'] is None else round(s['heldout']['kappa'], 3)}, would be accepted: {s['accepted']}")
        if result["accepted"] and exploratory:
            print(f"\nACCEPTED on exploratory inputs: threshold {result['frozen_threshold']} (bars {bars}); nothing from a --no-provenance run is frozen")
        elif result["accepted"]:
            print(f"\nACCEPTED: threshold {result['frozen_threshold']} (bars {bars})")
            print(f"next: python scripts/freeze_judge.py --threshold {result['frozen_threshold']} --records " + " ".join(str(sidecar_path(f)) for f in args.files) + " results/stability/*.judge.jsonl --note \"...\"")
        else:
            print(f"\nNOT ACCEPTED (bars {bars}): " + "; ".join(result["reasons"]))
            print("no threshold is frozen; every VQA cell stays unassessed and the human labels are the VQA evidence (decision 4 fallback)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
