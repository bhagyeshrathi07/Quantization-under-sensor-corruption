"""Decision 4 of manifests/judge_decisions_2026-11-13.md on synthetic labels,
written before any human label exists: the threshold rule, the acceptance under
leave-one-scene-out cross-validation, the inter-labeller statistics, the blind
sheet and the calibration script end to end."""
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from fractions import Fraction

from qflip.calibrate import (
    DEFAULT_ACCEPTANCE,
    CalibrationError,
    Item,
    acceptance,
    acceptance_bars,
    agreement_curve,
    calibrate,
    choose_threshold,
    cohen_kappa,
    confusion,
    exact_bar,
    exact_rates,
    fit,
    fleiss_kappa,
    majority_label,
    metrics,
    pairwise_agreement,
    pooled_heldout,
)
from qflip.config import load_config
from qflip.guard import prepare_calibration_files
from qflip.judge import backend_description, judge_backend, judge_input_hash, judge_settings
from qflip.provenance import ProvenanceError
from qflip.freeze import FreezeError

ROOT = Path(__file__).resolve().parents[1]
ENV = {k: v for k, v in os.environ.items() if k not in ("JUDGE_API_KEY", "OPENAI_API_KEY")}


def _items(spec, scene_of=lambda i: f"s{i % 5}"):
    """spec: iterable of (score, label[, majority[, rep0]])."""
    out = []
    for i, row in enumerate(spec):
        score, label = row[0], row[1]
        maj = row[2] if len(row) > 2 else None
        rep0 = row[3] if len(row) > 3 else None
        out.append(Item(f"D{i:03d}", scene_of(i), score, label, maj, rep0))
    return out


def test_threshold_rule_maximizes_agreement_and_takes_the_lowest_tie():
    # correct answers score 60..100, incorrect 0..30: every t in 31..60 agrees perfectly; the rule picks 31
    pairs = [(s, 1) for s in range(60, 101, 10)] + [(s, 0) for s in range(0, 31, 10)]
    curve = agreement_curve(pairs)
    assert len(curve) == 101 and max(curve) == 1.0
    assert choose_threshold(curve) == 31
    assert curve[0] == 5 / 9  # t = 0: everything judged correct
    f = fit(pairs)
    assert f["threshold"] == 31 and f["agreement"] == 1.0 and f["kappa"] == 1.0
    assert f["judge_correct_human_incorrect"] == 0 and f["human_correct_judge_incorrect"] == 0
    with pytest.raises(CalibrationError):
        agreement_curve([(50, 2)])
    with pytest.raises(CalibrationError):
        agreement_curve([(101, 1)])
    with pytest.raises(CalibrationError):
        choose_threshold([1.0] * 100)


def test_confusion_kappa_and_recalls():
    pairs = [(80, 1), (70, 1), (20, 1), (90, 0), (10, 0), (5, 0)]
    c = confusion(pairs, 50)
    assert c == {"tp": 2, "fn": 1, "fp": 1, "tn": 2}
    m = metrics(pairs, 50)
    assert m["agreement"] == pytest.approx(4 / 6) and m["recall_correct"] == pytest.approx(2 / 3) and m["recall_incorrect"] == pytest.approx(2 / 3)
    assert m["kappa"] == pytest.approx((4 / 6 - 0.5) / 0.5)
    # undefined when both sides give one class only
    assert cohen_kappa({"tp": 0, "fp": 0, "fn": 0, "tn": 10}) is None
    assert cohen_kappa({"tp": 0, "fp": 0, "fn": 0, "tn": 0}) is None
    # chance-level: judge says incorrect for everything while humans are split -> kappa 0
    assert cohen_kappa({"tp": 0, "fp": 0, "fn": 5, "tn": 5}) == pytest.approx(0.0)


def test_uninformative_judge_passes_agreement_alone_but_fails_acceptance():
    """Review of November 13: a judge that calls everything incorrect agrees
    with humans 90% of the time when 90% of answers are incorrect. The class
    bars catch it."""
    rng = random.Random(1)
    spec = [(rng.uniform(0, 30), 0) for _ in range(108)] + [(rng.uniform(0, 30), 1) for _ in range(12)]
    # scores carry no information about the label; 12 human-correct items (< 15 anyway)
    items = _items(spec, scene_of=lambda i: f"s{i % 6}")
    res = calibrate(items)
    assert res["calibrated"] and not res["accepted"]
    assert res["fit"]["agreement"] >= 0.80  # the agreement bar alone would have approved it
    reasons = " ".join(res["reasons"])
    assert "only 12 human-correct items" in reasons
    assert "recall on human-correct" in reasons or "kappa" in reasons
    assert res["frozen_threshold"] is None


def test_informative_judge_is_accepted_and_the_frozen_threshold_is_the_full_fit():
    rng = random.Random(2)
    spec = []
    for i in range(120):
        label = 1 if i % 3 == 0 else 0  # 40 correct, 80 incorrect
        base = rng.uniform(55, 95) if label else rng.uniform(0, 40)
        spec.append((base, label, label, min(100, max(0, base + rng.uniform(-8, 8)))))
    items = _items(spec, scene_of=lambda i: f"s{i % 8}")
    res = calibrate(items)
    assert res["calibrated"] and res["accepted"], res["reasons"]
    assert res["frozen_threshold"] == res["fit"]["threshold"]
    assert 40 <= res["frozen_threshold"] <= 55
    assert res["loso"]["n_scenes"] == 8 and len(res["loso"]["folds"]) == 8
    pooled = res["loso"]["pooled"]
    assert pooled["n"] == 120 and pooled["threshold"] is None
    # sensitivity rows exist and never change the verdict fields
    assert set(res["sensitivity"]) == {"majority_label", "replicate_0_only"}
    assert res["sensitivity"]["majority_label"]["items_changed_by_discussion"] == 0
    assert "curve" not in res["sensitivity"]["replicate_0_only"]["fit"]


def test_leave_one_scene_out_fits_without_the_held_out_scene():
    """One scene carries incorrect answers scored 40; with them the full fit
    moves to 41, without them it is 28. The fold that holds that scene out
    must fit 28 (a fit that saw the held-out items would pick 41 and judge them
    all incorrect; review of November 14)."""
    others = [(s, 1) for s in range(70, 100, 3)] + [(s, 0) for s in range(0, 30, 3)]  # 10 + 10: every t in 28..70 is perfect -> 28
    odd = [(40, 0)] * 6  # scene "odd": incorrect answers at 40
    items = _items(others, scene_of=lambda i: f"s{i % 2}") + [Item(f"O{i}", "odd", s, y) for i, (s, y) in enumerate(odd)]
    full = fit([(it.score, it.label) for it in items])
    assert full["threshold"] == 41 and full["agreement"] == 1.0
    res = pooled_heldout(items)
    fold = next(f for f in res["folds"] if f["scene"] == "odd")
    assert fold["threshold"] == choose_threshold(agreement_curve(others)) == 28 != full["threshold"]
    assert fold["n_train"] == 20 and fold["n_held"] == 6
    assert fold["confusion"] == {"tp": 0, "fp": 6, "fn": 0, "tn": 0}  # judged correct at 28, humans say incorrect
    assert res["pooled"]["confusion"]["fp"] == 6 and res["pooled"]["n"] == 26
    with pytest.raises(CalibrationError, match="at least 2 scenes"):
        pooled_heldout([Item("a", "s", 50, 1), Item("b", "s", 10, 0)])


def test_kappa_exactly_at_the_bar_passes():
    """Review of November 14: TP 9, TN 78, FP 12, FN 6 gives kappa 2/5 exactly;
    floating point gave 0.3999999999999999 and failed the 0.40 bar."""
    c = {"tp": 9, "tn": 78, "fp": 12, "fn": 6}
    rates = exact_rates(c)
    assert rates["kappa"] == Fraction(2, 5) and rates["agreement"] == Fraction(87, 105)
    assert rates["recall_correct"] == Fraction(9, 15) and rates["recall_incorrect"] == Fraction(78, 90)
    ok, why = acceptance({"confusion": c}, 15, 90, DEFAULT_ACCEPTANCE)
    assert ok and why == [], why
    # one item less on the diagonal and it fails on kappa alone
    ok, why = acceptance({"confusion": {"tp": 8, "tn": 78, "fp": 12, "fn": 7}}, 15, 90, DEFAULT_ACCEPTANCE)
    assert not ok and len(why) == 2 and why[0].startswith("held-out Cohen's kappa") and "recall on human-correct" in why[1]
    assert cohen_kappa(c) == 0.4
    assert exact_bar(0.40) == Fraction(2, 5) and exact_bar(15) == 15


def test_acceptance_bars_each_fail_separately():
    def pooled(tp, fp, fn, tn):
        return {"threshold": None, "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}

    good = pooled(24, 5, 6, 85)  # agreement 0.908, recall 0.8 / 0.944
    assert acceptance(good, 30, 90, DEFAULT_ACCEPTANCE) == (True, [])
    ok, why = acceptance(pooled(19, 10, 11, 60), 30, 70, DEFAULT_ACCEPTANCE)  # agreement 0.79, kappa 0.495, recalls 0.633 / 0.857
    assert not ok and why == ["held-out agreement 0.790 below 0.80"]
    ok, why = acceptance(pooled(0, 0, 0, 100), 30, 90, DEFAULT_ACCEPTANCE)  # judge and humans both say incorrect for everything held out
    assert not ok and "held-out Cohen's kappa undefined" in why[0] and "recall on human-correct items undefined" in why[1]
    ok, why = acceptance(pooled(15, 5, 15, 85), 30, 90, DEFAULT_ACCEPTANCE)  # agreement 0.833, kappa 0.5, recall correct 0.5
    assert not ok and why == ["held-out recall on human-correct items 0.500 below 0.60"]
    ok, why = acceptance(good, 14, 90, DEFAULT_ACCEPTANCE)
    assert not ok and why == ["only 14 human-correct items, 15 required"]
    ok, why = acceptance(good, 90, 3, DEFAULT_ACCEPTANCE)
    assert not ok and why == ["only 3 human-incorrect items, 15 required"]


def test_too_few_items_means_no_calibration():
    items = _items([(80, 1), (10, 0)] * 40, scene_of=lambda i: f"s{i % 4}")  # 80 items
    res = calibrate(items)
    assert res == {"n_items": 80, "bars": DEFAULT_ACCEPTANCE, "calibrated": False, "accepted": False, "frozen_threshold": None,
                   "reasons": ["only 80 calibration items, 100 required (plan section 7); no calibration"]}
    with pytest.raises(CalibrationError, match="duplicate item"):
        calibrate([Item("x", f"s{i % 3}", 50, 1) for i in range(100)])


def test_acceptance_bars_come_from_config_as_a_unit():
    assert acceptance_bars(None) == DEFAULT_ACCEPTANCE
    assert acceptance_bars({}) == DEFAULT_ACCEPTANCE
    live = acceptance_bars(load_config(ROOT / "config/pilot.yaml")["judge"])
    assert live == DEFAULT_ACCEPTANCE  # the decision-4 numbers are the ones in config
    with pytest.raises(CalibrationError, match="exactly"):
        acceptance_bars({"acceptance": {"agreement": 0.8}})
    with pytest.raises(CalibrationError, match="positive integer"):
        acceptance_bars({"acceptance": {**DEFAULT_ACCEPTANCE, "min_per_class": 0}})
    with pytest.raises(CalibrationError, match=r"\[0, 1\]"):
        acceptance_bars({"acceptance": {**DEFAULT_ACCEPTANCE, "kappa": 1.4}})


def test_inter_labeller_statistics():
    rows = [[1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]]
    assert fleiss_kappa(rows) == pytest.approx(1 / 3)
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0]]) == pytest.approx(1.0)
    assert fleiss_kappa([[1, 1, 1], [1, 1, 1]]) is None
    by = {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "c": [r[2] for r in rows]}
    assert pairwise_agreement(by) == {"a/b": 0.75, "a/c": 0.75, "b/c": 0.5}
    assert [majority_label(r) for r in rows] == [1, 0, 1, 0]
    with pytest.raises(CalibrationError):
        majority_label([1, 0])
    with pytest.raises(CalibrationError):
        pairwise_agreement({"a": [1], "b": [1, 0]})


# ---- the scripts ----


def _answer(qid, scene, precision, pred):
    return {"question_id": qid, "scene_token": scene, "frame_token": "f", "cell": "perception/VQA", "condition": "clean", "precision": precision, "n_views": 6,
            "views": ["CAM_FRONT"] * 6, "question": f"Question {int(qid[1:])}: what is the state of the traffic light?", "gt": "red", "pred": pred, "error": None, "model": f"m-{precision}"}


def _world(tmp_path, n_questions=60, scenes=6):
    """Two sidecar-less answer files (one per precision) and judge sidecars
    with three distinct judgings per answer; the hidden truth is about 40%
    correct with scores well apart from the incorrect ones."""
    cfg = load_config(ROOT / "config/pilot.yaml")
    settings, backend = judge_settings(cfg["judge"]), backend_description(judge_backend(cfg["judge"]))
    rng = random.Random(7)
    truth = {}
    for prec in ("bf16", "int4"):
        recs, judged = [], []
        for i in range(n_questions):
            qid, scene = f"q{i:03d}", f"scene{i % scenes}"
            pred = f"{prec} answer {i}: the light is {'red' if rng.random() < 0.5 else 'green'}"
            r = _answer(qid, scene, prec, pred)
            recs.append(r)
            correct = (i * 7 + (0 if prec == "bf16" else 3)) % 10 < 4
            centre = rng.uniform(55, 90) if correct else rng.uniform(0, 30)
            truth[(qid, prec)] = correct
            h = judge_input_hash(r["cell"], r["question"], r["gt"], r["pred"], None, settings)
            for k in range(3):
                score = int(round(min(100, max(0, centre + rng.uniform(-6, 6)))))
                judged.append({"question_id": qid, "scene_token": scene, "cell": r["cell"], "precision": prec, "replicate": k, "input_sha256": h, "desc": None, "score": score,
                               "reply": f"Total Score: {score}\n\nBrief Summary: x", "score_rule": "released", "judge_backend": backend, "response_id": f"id-{prec}-{qid}-{k}"})
        (tmp_path / f"{prec}_clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (tmp_path / f"{prec}_clean.judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    return truth


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, env=ENV, capture_output=True, text=True)


def _build(tmp_path, labellers=("ronak", "bhagyesh", "akshay"), copies=False, seed=None):
    files = [tmp_path / "bf16_clean.jsonl", tmp_path / "int4_clean.jsonl"]
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    extra = (["--copies"] if copies else []) + (["--seed", seed] if seed is not None else [])
    r = _run("10_build_label_sheet.py", *files, "--labellers", *labellers, "--sheet", sheet, "--key", key, "--no-provenance", *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    return files, sheet, key, r


def _read(sheet):
    with Path(sheet).open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames


def _write(sheet, rows, cols):
    with Path(sheet).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_10_builds_a_blind_shuffled_sheet_and_a_separate_key(tmp_path):
    _world(tmp_path)
    files, sheet, key, r = _build(tmp_path, seed=5)
    assert "EXPLORATORY" in r.stdout
    rows, cols = _read(sheet)
    assert len(rows) == 120
    assert cols == ["item", "cell", "question", "ground_truth", "answer", "label_ronak", "label_bhagyesh", "label_akshay", "consensus", "consensus_reason"]
    assert all(row["label_ronak"] == "" and row["consensus"] == "" for row in rows)
    text = sheet.read_text()
    k = json.loads(key.read_text())
    for v in k["items"].values():
        assert v["scene_token"] not in text and v["question_id"] not in text  # no scene token, no question id
    assert "bf16" not in text.replace("bf16 answer", "") and "int4" not in text.replace("int4 answer", "")  # precision only inside the synthetic answer text
    assert k["n_items"] == 120 and set(k["items"]) == {row["item"] for row in rows} and k["labellers"] == ["ronak", "bhagyesh", "akshay"]
    assert k["order"] == [row["item"] for row in rows] and k["provenance"]["analysis_class"] == "exploratory"
    assert {(v["question_id"], v["precision"]) for v in k["items"].values()} == {(f"q{i:03d}", p) for i in range(60) for p in ("bf16", "int4")}
    # the key names what the sheet hides and binds what it shows; no score
    assert set(next(iter(k["items"].values()))) == {"question_id", "precision", "scene_token", "cell", "pred_sha256", "display_sha256"}
    # shuffled: not in question order; deterministic in the seed
    assert [k["items"][row["item"]]["question_id"] for row in rows] != sorted(k["items"][row["item"]]["question_id"] for row in rows)
    r2 = _run("10_build_label_sheet.py", *files, "--labellers", "ronak", "bhagyesh", "akshay", "--sheet", tmp_path / "s2.csv", "--key", tmp_path / "k2.json", "--seed", 5, "--no-provenance")
    assert r2.returncode == 0 and (tmp_path / "s2.csv").read_text() == text
    # never overwritten; sidecars refused; even labeller counts refused; sidecar-less files without --no-provenance refused
    r3 = _run("10_build_label_sheet.py", *files, "--labellers", "a", "b", "c", "--sheet", sheet, "--key", tmp_path / "k3.json", "--no-provenance")
    assert r3.returncode == 1 and "never overwritten" in r3.stderr
    r4 = _run("10_build_label_sheet.py", tmp_path / "bf16_clean.judge.jsonl", "--labellers", "a", "--sheet", tmp_path / "x.csv", "--key", tmp_path / "x.json", "--no-provenance")
    assert r4.returncode == 1 and "is a sidecar" in r4.stderr
    r5 = _run("10_build_label_sheet.py", *files, "--labellers", "a", "b", "--sheet", tmp_path / "y.csv", "--key", tmp_path / "y.json", "--no-provenance")
    assert r5.returncode == 1 and "odd number" in r5.stderr
    r6 = _run("10_build_label_sheet.py", *files, "--labellers", "a", "b", "c", "--sheet", tmp_path / "z.csv", "--key", tmp_path / "z.json")
    assert r6.returncode == 1 and "no .meta.json sidecar" in r6.stderr


def test_10_writes_independent_copies_and_12_merges_them(tmp_path):
    truth = _world(tmp_path)
    files, sheet, key, r = _build(tmp_path, copies=True)
    copies = {n: tmp_path / f"sheet.{n}.csv" for n in ("ronak", "bhagyesh", "akshay")}
    assert all(p.exists() for p in copies.values())
    rows, cols = _read(copies["ronak"])
    assert cols == ["item", "cell", "question", "ground_truth", "answer", "label_ronak"] and len(rows) == 120
    master_rows, _ = _read(sheet)
    assert [r["item"] for r in rows] == [r["item"] for r in master_rows]
    # each labeller fills their copy; the master stays empty until the merge
    k = json.loads(key.read_text())
    for n, p in copies.items():
        rows, cols = _read(p)
        for i, row in enumerate(rows):
            if n == "akshay" and i >= 110:
                continue  # akshay did not finish
            row[f"label_{n}"] = str(int(truth[(k["items"][row["item"]]["question_id"], k["items"][row["item"]]["precision"])]))
        _write(p, rows, cols)
    r = _run("12_merge_label_copies.py", "--sheet", sheet, *copies.values())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "label_akshay <-" in r.stdout and "110 label(s), 10 blank" in r.stdout
    rows, _ = _read(sheet)
    assert all(row["label_ronak"] in ("0", "1") and row["consensus"] == "" for row in rows)
    assert sum(1 for row in rows if row["label_akshay"] == "") == 10
    # labels-only report works before any consensus exists, and needs neither key nor files
    r = _run("11_calibrate_judge.py", "--labels-only", "--sheet", sheet)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "110 complete (all label columns), 10 unfinished" in r.stdout and "Fleiss kappa 1.0" in r.stdout and "akshay/bhagyesh 1.000" in r.stdout
    assert "akshay 110" in r.stdout and "10 tail item(s) carry labels from some labellers only" in r.stdout
    # the consensus columns are not read before the discussion: a note there does not stop the report
    rows, cols = _read(sheet)
    rows[0]["consensus"], rows[0]["consensus_reason"] = "discussion pending", "to discuss"
    _write(sheet, rows, cols)
    r = _run("11_calibrate_judge.py", "--labels-only", "--sheet", sheet)
    assert r.returncode == 0, r.stdout + r.stderr
    rows[0]["consensus"] = rows[0]["consensus_reason"] = ""
    _write(sheet, rows, cols)
    # a labeller who skipped a row did not label in sheet order
    rows, cols = _read(sheet)
    rows[5]["label_bhagyesh"] = ""
    _write(sheet, rows, cols)
    r = _run("11_calibrate_judge.py", "--labels-only", "--sheet", sheet)
    assert r.returncode == 1 and "bhagyesh labelled" in r.stderr and "labelled in sheet order" in r.stderr
    rows[5]["label_bhagyesh"] = "0"
    _write(sheet, rows, cols)
    # a copy edited in its displayed text, or re-merged with a different label, is refused
    rows, cols = _read(copies["ronak"])
    rows[3]["answer"] += "!"
    _write(copies["ronak"], rows, cols)
    r = _run("12_merge_label_copies.py", "--sheet", sheet, copies["ronak"])
    assert r.returncode == 1 and "answer differs from the master sheet" in r.stderr
    rows[3]["answer"] = rows[3]["answer"][:-1]
    rows[3]["label_ronak"] = "1" if rows[3]["label_ronak"] == "0" else "0"
    _write(copies["ronak"], rows, cols)
    r = _run("12_merge_label_copies.py", "--sheet", sheet, copies["ronak"])
    assert r.returncode == 1 and "already carry a different label_ronak" in r.stderr
    r = _run("12_merge_label_copies.py", "--sheet", sheet, copies["bhagyesh"], copies["bhagyesh"])
    assert r.returncode == 1 and "a second copy" in r.stderr


def _fill(sheet, key, truth, labellers=("ronak", "bhagyesh", "akshay"), noise=0.05, leave_unfinished=0, partial=0, blank_first=0):
    """Labels from the hidden truth with independent noise per labeller; the
    consensus is the majority. The last `leave_unfinished` rows stay empty, the
    `partial` rows before them get one label only, the first `blank_first` rows
    stay empty (a sheet not labelled in order)."""
    rng = random.Random(11)
    k = json.loads(key.read_text())
    rows, cols = _read(sheet)
    n = len(rows)
    for i, row in enumerate(rows):
        for c in [f"label_{name}" for name in labellers] + ["consensus", "consensus_reason"]:
            row[c] = ""
        t = truth[(k["items"][row["item"]]["question_id"], k["items"][row["item"]]["precision"])]
        if i >= n - leave_unfinished or i < blank_first:
            continue
        labels = [(1 - int(t)) if rng.random() < noise else int(t) for _ in labellers]
        if i >= n - leave_unfinished - partial:
            row[f"label_{labellers[0]}"] = str(labels[0])
            continue
        for name, lab in zip(labellers, labels):
            row[f"label_{name}"] = str(lab)
        row["consensus"] = str(majority_label(labels))
    _write(sheet, rows, cols)


def test_11_calibrates_from_sheet_key_and_sidecars(tmp_path):
    truth = _world(tmp_path)
    files, sheet, key, _ = _build(tmp_path)
    _fill(sheet, key, truth, leave_unfinished=6)
    out = tmp_path / "cal.json"
    r = _run("11_calibrate_judge.py", *files, "--sheet", sheet, "--key", key, "--out", out, "--no-provenance")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "114 complete" in r.stdout and "6 unfinished" in r.stdout and "EXPLORATORY" in r.stdout
    rep = json.loads(out.read_text())
    assert rep["counts"] == {"sheet_items": 120, "complete": 114, "unfinished": 6, "partial_tail": 0, "without_aggregate": 0, "calibration_items": 114, "labelled_per_labeller": {"ronak": 114, "bhagyesh": 114, "akshay": 114}}
    assert rep["production_replicates"] == 3 and rep["judge_backend"] and rep["analysis_class"] == "exploratory"
    cal = rep["calibration"]
    assert cal["calibrated"] and cal["accepted"], cal["reasons"]
    assert 30 <= cal["frozen_threshold"] <= 55 and cal["frozen_threshold"] == cal["fit"]["threshold"]
    assert cal["loso"]["n_scenes"] == 6 and cal["loso"]["pooled"]["agreement"] >= 0.8
    assert set(cal["sensitivity"]) == {"majority_label", "replicate_0_only"}
    inter = rep["inter_labeller"]
    assert inter["n_items"] == 114 and set(inter["pairwise_agreement"]) == {"bhagyesh/ronak", "akshay/bhagyesh", "akshay/ronak"} and inter["fleiss_kappa"] > 0.7
    # exploratory inputs: accepted is reported, the freeze command is not
    assert "ACCEPTED on exploratory inputs" in r.stdout and "freeze_judge.py" not in r.stdout
    it = rep["items"][0]
    assert set(it) >= {"item", "scene", "score", "label", "majority", "score_rep0", "question_id", "precision", "cell", "scores", "labels"} and len(it["scores"]) == 3
    assert it["score"] == pytest.approx(sum(it["scores"]) / 3)


def test_11_refuses_sheets_that_break_the_rules(tmp_path):
    truth = _world(tmp_path)
    files, sheet, key, _ = _build(tmp_path)
    base = ["11_calibrate_judge.py", *files, "--sheet", sheet, "--key", key, "--no-provenance"]
    # different stopping points (review of November 14, second pass): the common prefix is calibrated, the tail's extra labels are kept and counted, not used
    _fill(sheet, key, truth, leave_unfinished=4, partial=2)
    r = _run(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "114 complete (all 3 labels and consensus), 6 unfinished" in r.stdout and "2 tail item(s) carry labels from some labellers only" in r.stdout
    rows, cols = _read(sheet)
    assert sum(1 for r_ in rows if r_["label_ronak"] != "") == 116  # the extra labels are still on the sheet
    # an item every labeller reached but without a consensus keeps the key closed
    rows[50]["consensus"] = ""
    _write(sheet, rows, cols)
    r = _run(*base)
    assert r.returncode == 1 and "has every label but no consensus" in r.stderr
    # a consensus on a tail item (not every labeller reached it) is refused
    rows[50]["consensus"] = str(majority_label([int(rows[50][f"label_{n}"]) for n in ("ronak", "bhagyesh", "akshay")]))
    rows[115]["consensus"] = "1"
    _write(sheet, rows, cols)
    r = _run(*base)
    assert r.returncode == 1 and "has a consensus but not every labeller reached it" in r.stderr
    # blank first rows with complete later rows: not labelled in sheet order
    _fill(sheet, key, truth, blank_first=20)
    r = _run(*base)
    assert r.returncode == 1 and "labelled in sheet order" in r.stderr
    # rows removed: the sheet no longer holds the issued items
    _fill(sheet, key, truth)
    rows, cols = _read(sheet)
    _write(sheet, rows[20:], cols)
    r = _run(*base)
    assert r.returncode == 1 and "does not hold the issued items in the issued order: 20 missing" in r.stderr
    # rows reordered
    _write(sheet, rows[1:] + rows[:1], cols)
    r = _run(*base)
    assert r.returncode == 1 and "order differs" in r.stderr
    _write(sheet, rows, cols)
    # displayed text edited after issue: the labels were given for other text (review of November 14, item 1)
    _fill(sheet, key, truth)
    rows, cols = _read(sheet)
    for col in ("answer", "question", "ground_truth", "cell"):
        edited = [dict(r_) for r_ in rows]
        edited[7][col] = edited[7][col] + " "
        _write(sheet, edited, cols)
        r = _run(*base)
        assert r.returncode == 1 and "sheet text (cell, question, ground truth, answer) differs from the issued one" in r.stderr, col
    _write(sheet, rows, cols)
    # a display column removed
    _fill(sheet, key, truth)
    rows, cols = _read(sheet)
    _write(sheet, [{c: v for c, v in r_.items() if c != "ground_truth"} for r_ in rows], [c for c in cols if c != "ground_truth"])
    r = _run(*base)
    assert r.returncode == 1 and "blind column(s) ['ground_truth'] missing" in r.stderr
    _write(sheet, rows, cols)
    # an answer that changed in the files since the sheet was built
    _fill(sheet, key, truth)
    raw = [json.loads(l) for l in (tmp_path / "int4_clean.jsonl").read_text().splitlines()]
    raw[0]["pred"] += " (edited)"
    (tmp_path / "int4_clean.jsonl").write_text("".join(json.dumps(x) + "\n" for x in raw))
    r = _run(*base)
    assert r.returncode == 1 and "answer in the given files differs from the issued one" in r.stderr
    # the key must belong to the given files
    r = _run("11_calibrate_judge.py", files[0], "--sheet", sheet, "--key", key, "--no-provenance")
    assert r.returncode == 1 and "the key was built from" in r.stderr
    # no key, no files: only --labels-only may do without them
    r = _run("11_calibrate_judge.py", "--sheet", sheet)
    assert r.returncode == 1 and "--labels-only" in r.stderr


def test_11_reports_when_the_judge_is_uninformative(tmp_path):
    """Scores unrelated to the labels: the fit still finds its best t, the
    held-out metrics fail and nothing is frozen."""
    _world(tmp_path)
    files, sheet, key, _ = _build(tmp_path, labellers=("a", "b", "c"))
    k = json.loads(key.read_text())
    rng = random.Random(3)
    random_truth = {(v["question_id"], v["precision"]): rng.random() < 0.3 for v in k["items"].values()}
    _fill(sheet, key, random_truth, labellers=("a", "b", "c"), noise=0.0)
    out = tmp_path / "cal.json"
    r = _run("11_calibrate_judge.py", *files, "--sheet", sheet, "--key", key, "--out", out, "--no-provenance")
    assert r.returncode == 0, r.stdout + r.stderr
    cal = json.loads(out.read_text())["calibration"]
    assert cal["calibrated"] and not cal["accepted"] and cal["frozen_threshold"] is None
    assert "NOT ACCEPTED" in r.stdout and "stays unassessed" in r.stdout


def test_calibration_files_must_be_development_answers_under_the_core_freeze(monkeypatch, tmp_path):
    """Review of November 14, item 2: the builder took final-split files and
    the calibrator took development sidecars without the freeze check or with
    other serving settings."""
    import qflip.guard as guard

    frozen = {"frozen": {"date": "2026-11-08T00:06:06+00:00", "git_commit": "abc"}}
    good_meta = {"split": "development", "freeze_checked": {"date": "2026-11-08T00:06:06+00:00", "git_commit": "abc"}, "precision": "bf16"}
    state = {"prov": {"split": "development", "analysis_class": "exploratory", "files": {}}, "meta": dict(good_meta), "diffs": []}
    monkeypatch.setattr(guard, "validate_analysis_files", lambda files, cfg, **kw: dict(state["prov"]))
    monkeypatch.setattr(guard, "load_frozen", lambda path: frozen)
    monkeypatch.setattr(guard, "load_meta", lambda f: dict(state["meta"]))
    monkeypatch.setattr(guard, "check_meta_against_frozen", lambda m, fz: list(state["diffs"]))
    monkeypatch.setattr(guard, "check_config", lambda *a, **k: [])  # the live-config comparison is covered by tests/test_amendment.py on a real freeze
    f = tmp_path / "bf16_clean.jsonl"
    f.write_text("")
    prov = prepare_calibration_files([f], {})
    assert prov["analysis_class"] == "calibration" and prov["frozen"] == frozen["frozen"]
    state["prov"]["split"] = "final"
    with pytest.raises(ProvenanceError, match="must be development-split answers"):
        prepare_calibration_files([f], {})
    state["prov"]["split"] = "development"
    state["meta"].pop("freeze_checked")
    with pytest.raises(FreezeError, match="freeze_checked=None"):
        prepare_calibration_files([f], {})
    state["meta"] = dict(good_meta)
    state["diffs"] = ["max_pixels: file='401408' frozen='200704'"]
    with pytest.raises(FreezeError, match="max_pixels"):
        prepare_calibration_files([f], {})
    state["diffs"] = []
    monkeypatch.setattr(guard, "load_frozen", lambda path: None)
    with pytest.raises(FreezeError, match="missing"):
        prepare_calibration_files([f], {})
    # synthetic path: sidecar-less files only, and labelled so
    monkeypatch.setattr(guard, "legacy_files_check", lambda files, s, e: {"split": "unknown", "analysis_class": "exploratory", "files": {}})
    prov = prepare_calibration_files([f], {}, no_provenance=True)
    assert prov["analysis_class"] == "exploratory" and prov["note"].startswith("EXPLORATORY")
    with pytest.raises(ProvenanceError, match="no answer files"):
        prepare_calibration_files([], {})
