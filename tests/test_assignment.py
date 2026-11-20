"""The label assignment of November 19: one human labeller per item, in
contiguous blocks; the sheet classification, the labelling tool, the
assignment writer and the disagreement list follow it."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qflip.calibrate import CalibrationError, classify_sheet, inter_labeller_stats, load_assignment

from tests.test_label_ui import ITEMS, ui  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
NAMES = ("ronak", "bhagyesh", "akshay")


def _rows(labels=None, consensus=None):
    rows = []
    for i, (item, cell, q, gt, ans) in enumerate(ITEMS):
        r = {"item": item, "cell": cell, "question": q, "ground_truth": gt, "answer": ans, "consensus": "", "consensus_reason": ""}
        for n in NAMES:
            r[f"label_{n}"] = ""
        if labels:
            for n, v in labels.get(item, {}).items():
                r[f"label_{n}"] = str(v)
        if consensus and item in consensus:
            r["consensus"] = str(consensus[item])
        rows.append(r)
    return rows


def _assign(tmp_path, blocks):
    p = tmp_path / "sheet.assignment.json"
    p.write_text(json.dumps({"assignment": blocks}))
    return p


def test_classification_follows_the_assignment(tmp_path):
    rows = _rows({"D001": {"ronak": 1}, "D002": {"ronak": 0}, "D003": {"bhagyesh": 1}})
    a = load_assignment(_assign(tmp_path, {"ronak": ["D001", "D002"], "bhagyesh": ["D003", "D004"], "akshay": ["D005"]}), rows, list(NAMES))
    complete, unfinished, progress = classify_sheet(rows, list(NAMES), need_consensus=False, assignment=a)
    assert sorted(complete) == ["D001", "D002", "D003"] and unfinished == ["D004", "D005"] and progress == {"ronak": 2, "bhagyesh": 1, "akshay": 0}
    assert complete["D003"]["labels"] == {"bhagyesh": 1} and complete["D003"]["labellers"] == ["bhagyesh"]
    # calibration mode: every finished item needs a consensus, an unfinished one must not have one
    with pytest.raises(CalibrationError, match="has every label but no consensus"):
        classify_sheet(rows, list(NAMES), need_consensus=True, assignment=a)
    rows2 = _rows({"D001": {"ronak": 1}, "D002": {"ronak": 0}, "D003": {"bhagyesh": 1}}, consensus={"D001": 1, "D002": 1, "D003": 1})
    complete, unfinished, _ = classify_sheet(rows2, list(NAMES), need_consensus=True, assignment=a)
    assert complete["D002"]["consensus"] == 1 and unfinished == ["D004", "D005"]
    rows3 = _rows({"D001": {"ronak": 1}}, consensus={"D001": 1, "D003": 0})
    with pytest.raises(CalibrationError, match="not every labeller reached it"):
        classify_sheet(rows3, list(NAMES), need_consensus=True, assignment=a)
    # a label outside the assignment, and a skipped row inside it, are refused
    with pytest.raises(CalibrationError, match="not in akshay's assignment"):
        classify_sheet(_rows({"D001": {"akshay": 1}}), list(NAMES), need_consensus=False, assignment=a)
    with pytest.raises(CalibrationError, match="first rows of their assignment"):
        classify_sheet(_rows({"D004": {"bhagyesh": 1}}), list(NAMES), need_consensus=False, assignment=a)
    # an assignment must cover every item, name sheet labellers only, and list items in sheet order
    for bad, msg in (({"ronak": ["D001"], "bhagyesh": ["D002"], "akshay": ["D003"]}, "assigned to nobody"),
                     ({"ronak": ["D002", "D001", "D003", "D004", "D005"]}, "in sheet order"),
                     ({"ronak": ITEMS_IDS(), "zed": ["D001"]}, "not on the sheet"),
                     ({"ronak": ["D001", "D009"]}, "not on the sheet")):
        with pytest.raises(CalibrationError, match=msg):
            load_assignment(_assign(tmp_path, bad), rows, list(NAMES))
    # statistics with no shared items: pairwise none, Fleiss None, majority = the item's own label
    st = inter_labeller_stats(complete, list(NAMES))
    assert st["pairwise_agreement"]["bhagyesh/ronak"] == {"n": 0, "agreement": None} and st["fleiss_kappa"] is None and st["n_items_all_labellers"] == 0
    assert st["majority_correct"] == 2 and st["labelled_per_labeller"] == {"ronak": 2, "bhagyesh": 1, "akshay": 0}
    # and with an overlap: pairwise on the shared items only
    rows4 = _rows({"D001": {"ronak": 1, "bhagyesh": 1}, "D002": {"ronak": 0, "bhagyesh": 1}})
    a2 = load_assignment(_assign(tmp_path, {"ronak": ["D001", "D002", "D003"], "bhagyesh": ["D001", "D002", "D004"], "akshay": ["D005"]}), rows4, list(NAMES))
    complete, _, _ = classify_sheet(rows4, list(NAMES), need_consensus=False, assignment=a2)
    st = inter_labeller_stats(complete, list(NAMES))
    assert st["pairwise_agreement"]["bhagyesh/ronak"] == {"n": 2, "agreement": 0.5} and st["n_items_multi"] == 2 and st["unanimous"] == 1


def ITEMS_IDS():
    return [it[0] for it in ITEMS]


def _write(path, rows, cols):
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, env=ENV, capture_output=True, text=True)


def test_tool_assignment_writer_and_disagreement_list(tmp_path):
    sheet = tmp_path / "sheet.csv"
    cols = [*ui.BLIND_COLUMNS, *(f"label_{n}" for n in NAMES), "consensus", "consensus_reason"]
    _write(sheet, _rows(), cols)
    for n in NAMES:
        _write(tmp_path / f"sheet.{n}.csv", _rows(), [*ui.BLIND_COLUMNS, f"label_{n}"])
    # the assignment writer: contiguous blocks, remainder on the first, never overwritten, picked up by default
    r = _run("15_label_assignment.py", "--sheet", sheet, "--labellers", *NAMES, "--out", tmp_path / "equal.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "equal.json").read_text())["assignment"] == {"ronak": ["D001", "D002"], "bhagyesh": ["D003", "D004"], "akshay": ["D005"]}
    r = _run("15_label_assignment.py", "--sheet", sheet, "--labellers", *NAMES, "--sizes", "1", "1", "1", "--out", tmp_path / "x.json")
    assert r.returncode == 1 and "summing to 5" in r.stderr
    r = _run("15_label_assignment.py", "--sheet", sheet, "--labellers", *NAMES, "--sizes", "1", "3", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    a = json.loads((tmp_path / "sheet.assignment.json").read_text())["assignment"]
    assert a == {"ronak": ["D001"], "bhagyesh": ["D002", "D003", "D004"], "akshay": ["D005"]}
    assert _run("15_label_assignment.py", "--sheet", sheet, "--labellers", *NAMES).returncode == 1  # never overwritten
    # the tool shows and labels the labeller's block only, in order (the default assignment file beside the sheet)
    c = ui.Copy(tmp_path / "sheet.bhagyesh.csv", "bhagyesh", tmp_path / "sheet.assignment.json")
    st = c.state()
    assert [i["item"] for i in st["items"]] == ["D002", "D003", "D004"] and st["n"] == 3 and st["next"] == 0
    assert c.set_label(0, 1)["item"] == "D002"
    with pytest.raises(ui.CopyError, match="labelled in sheet order"):
        c.set_label(2, 1)
    c.set_label(1, 0)
    c.set_label(2, 1)
    c.close()
    with (tmp_path / "sheet.bhagyesh.csv").open(encoding="utf-8", newline="") as fh:
        assert [x["label_bhagyesh"] for x in csv.DictReader(fh)] == ["", "1", "0", "1", ""]
    r = _run("13_label_ui.py", "--labeller", "bhagyesh", "--sheet", sheet, "--check")
    assert r.returncode == 0 and "3 items for bhagyesh (D002 to D004" in r.stdout and "3 labelled" in r.stdout
    # a label outside the block is refused at load
    _write(tmp_path / "sheet.akshay.csv", _rows({"D001": {"akshay": 1}}), [*ui.BLIND_COLUMNS, "label_akshay"])
    with pytest.raises(ui.CopyError, match="not in akshay's assignment"):
        ui.Copy(tmp_path / "sheet.akshay.csv", "akshay", tmp_path / "sheet.assignment.json")
    _write(tmp_path / "sheet.akshay.csv", _rows(), [*ui.BLIND_COLUMNS, "label_akshay"])
    # model copies and records, then the disagreement list: refused until the block is done, then listed with reasons
    for m, labs in (("fable", [1, 1, 0, 0, 1]), ("astra", [1, 0, 1, 0, 0])):
        _write(tmp_path / f"sheet.{m}.csv", [dict(r, **{f"label_{m}": str(v)}) for r, v in zip(_rows(), labs)], [*ui.BLIND_COLUMNS, f"label_{m}"])
        d = tmp_path / "records" / f"llm_{m}"
        d.mkdir(parents=True)
        (d / "labels.json").write_text(json.dumps({"items": {it: {"label": v, "reason": f"{m} says {v}", "ambiguous": it == "D004"} for it, v in zip(ITEMS_IDS(), labs)}}))
    r = _run("16_disagreements.py", "--labeller", "ronak", "--sheet", sheet, "--records-dir", tmp_path / "records")
    assert r.returncode == 1 and "without a label" in r.stderr
    r = _run("16_disagreements.py", "--labeller", "bhagyesh", "--sheet", sheet, "--records-dir", tmp_path / "records")
    assert r.returncode == 0, r.stdout + r.stderr
    md = (tmp_path / "records" / "disagreements_bhagyesh.md").read_text()
    # bhagyesh D002=1 D003=0 D004=1; fable D002=1 D003=0 D004=0; astra D002=0 D003=1 D004=0 -> every item disagrees with at least one model
    assert "Disagreements: 3 of 3" in md and "## D002" in md and "## D004" in md and "fable (0, flagged ambiguous): fable says 0" in md and "fable 2/3, astra 0/3" in r.stdout
    assert "label_" not in md and "precision" not in md.lower()
