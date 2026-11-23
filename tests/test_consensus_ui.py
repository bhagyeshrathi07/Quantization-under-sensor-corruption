"""Consensus mode of scripts/13_label_ui.py: the master sheet with the human
and model labels side by side; only the consensus columns are written."""
import csv
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.test_label_ui import ITEMS, ui  # noqa: F401

NAMES = ("ronak", "bhagyesh", "akshay")


def _write(path, rows, cols):
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def _world(tmp_path):
    cols = [*ui.BLIND_COLUMNS, *(f"label_{n}" for n in NAMES), "consensus", "consensus_reason"]
    human = {"D001": ("ronak", 1), "D002": ("ronak", 0), "D003": ("bhagyesh", 1), "D004": ("bhagyesh", 0), "D005": ("akshay", 1)}
    rows = []
    for item, cell, q, gt, ans in ITEMS:
        r = {"item": item, "cell": cell, "question": q, "ground_truth": gt, "answer": ans}
        who, lab = human[item]
        r[f"label_{who}"] = str(lab)
        rows.append(r)
    sheet = tmp_path / "sheet.csv"
    _write(sheet, rows, cols)
    (tmp_path / "sheet.assignment.json").write_text(json.dumps({"assignment": {"ronak": ["D001", "D002"], "bhagyesh": ["D003", "D004"], "akshay": ["D005"]}}))
    for m, labs in (("fable", [1, 1, 1, 0, 0]), ("astra", [1, 0, 0, 0, 1])):
        _write(tmp_path / f"sheet.{m}.csv", [dict(r, **{f"label_{m}": str(v)}) for r, v in zip(rows, labs)], [*ui.BLIND_COLUMNS, f"label_{m}"])
        d = tmp_path / "records" / f"llm_{m}"
        d.mkdir(parents=True)
        (d / "labels.json").write_text(json.dumps({"items": {it[0]: {"label": v, "reason": f"{m} reason {it[0]}", "ambiguous": it[0] == "D002"} for it, v in zip(ITEMS, labs)}}))
    return sheet, cols


class Client:
    def __init__(self, server):
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(self):
        with urllib.request.urlopen(self.base + "/api/state") as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


def test_consensus_mode_shows_all_labels_and_writes_the_consensus_columns_only(tmp_path):
    sheet, cols = _world(tmp_path)
    original = sheet.read_bytes()
    m = ui.Master(sheet, tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    st = m.state()
    assert st["mode"] == "consensus" and st["n"] == 5 and st["filled"] == 0
    # D001: human 1, fable 1, astra 1 -> agreed; D002: human 0, fable 1 -> disagreement; D003: human 1, fable 1, astra 0 -> disagreement
    byid = {it["item"]: it for it in st["items"]}
    assert byid["D001"]["human"] == {"ronak": 1} and byid["D001"]["disagreement"] is False
    assert byid["D002"]["disagreement"] is True and byid["D002"]["models"]["fable"] == {"label": 1, "reason": "fable reason D002", "ambiguous": True}
    assert [it["item"] for it in st["items"] if it["disagreement"]] == ["D002", "D003", "D005"]
    try:
        server = ui.serve(m, tmp_path / "guide.md", port=0)
    except PermissionError:
        m.close()
        pytest.skip("cannot open local sockets in this environment")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cl = Client(server)
    assert cl.get()["items"][1]["consensus"] is None
    # consensus with a comment; the comment alone; clearing; a bad label; a stale item
    status, j = cl.post("/api/consensus", {"index": 1, "item": "D002", "label": 1, "reason": "the reference action is there"})
    assert status == 200 and j == {"item": "D002", "consensus": 1, "reason": "the reference action is there", "filled": 1, "n": 5}
    status, j = cl.post("/api/consensus", {"index": 1, "item": "D002", "label": 1, "reason": "edited\ncomment"})
    assert status == 200 and j["reason"] == "edited comment"
    assert cl.post("/api/consensus", {"index": 1, "item": "D002", "label": 2, "reason": ""})[0] == 409  # refused by the sheet, not by the type check
    assert cl.post("/api/consensus", {"index": 1, "item": "D009", "label": 1, "reason": ""})[0] == 409
    assert cl.post("/api/label", {"index": 0, "item": "D001", "label": 1})[0] == 404  # labelling endpoint is off in consensus mode
    # the file: consensus columns changed, nothing else
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["consensus"] for r in rows] == ["", "1", "", "", ""] and rows[1]["consensus_reason"] == "edited comment"
    with sheet.open(encoding="utf-8", newline="") as fh:
        fixed = [[r[c] for c in cols if c not in ("consensus", "consensus_reason")] for r in csv.DictReader(fh)]
    orig_rows = list(csv.DictReader(original.decode("utf-8").splitlines(keepends=True)))
    assert fixed == [[r[c] for c in cols if c not in ("consensus", "consensus_reason")] for r in orig_rows]
    # accept the human label on every agreed item: D001 and D004 (both models agree with the human); D002 already set; D003 and D005 stay open
    status, j = cl.post("/api/accept-agreed", {})
    assert status == 200 and j == {"accepted": 2, "filled": 3, "n": 5}
    st = cl.get()
    assert {it["item"]: it["consensus"] for it in st["items"]} == {"D001": 1, "D002": 1, "D003": None, "D004": 0, "D005": None}
    status, j = cl.post("/api/consensus", {"index": 1, "item": "D002", "label": None, "reason": None})
    assert status == 200 and j["consensus"] is None and j["filled"] == 2
    server.shutdown()
    server.server_close()
    m.close()
    # the finished sheet feeds the calibration classification: consensus required on every item
    from qflip.calibrate import CalibrationError, classify_sheet, load_assignment
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    a = load_assignment(tmp_path / "sheet.assignment.json", rows, list(NAMES))
    with pytest.raises(CalibrationError, match="has every label but no consensus"):
        classify_sheet(rows, list(NAMES), need_consensus=True, assignment=a)
    # a consensus that differs from the human label needs a comment (review of November 23); D003: human 1
    m0 = ui.Master(sheet, tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    with pytest.raises(ui.CopyError, match="a comment with the reason is required"):
        m0.set_consensus(2, 0, "")
    assert m0.set_consensus(2, 0, "the action is missing")["consensus"] == 0
    assert m0.set_consensus(2, 1, "")["consensus"] == 1  # equal to the human label: no comment needed
    m0.close()
    # consensus mode does not open while any assigned human label is missing
    with sheet.open(encoding="utf-8", newline="") as fh:
        rows_all = list(csv.DictReader(fh))
    rows_all[4]["label_akshay"] = ""
    _write(tmp_path / "unfinished.csv", rows_all, cols)
    for mm in ("fable", "astra"):
        (tmp_path / f"unfinished.{mm}.csv").write_text((tmp_path / f"sheet.{mm}.csv").read_text())
    with pytest.raises(ui.CopyError, match="consensus mode opens only when every block is fully labelled"):
        ui.Master(tmp_path / "unfinished.csv", tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    # a model copy whose text differs from the master, or a second instance, is refused
    bad = tmp_path / "sheet.astra.csv"
    text = bad.read_text(encoding="utf-8")
    bad.write_text(text.replace("Back up.", "Back up!"), encoding="utf-8", newline="")
    with pytest.raises(ui.CopyError, match="differ from the master sheet"):
        ui.Master(sheet, tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    bad.write_text(text, encoding="utf-8", newline="")
    m1 = ui.Master(sheet, tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    with pytest.raises(ui.CopyError, match="another instance"):
        ui.Master(sheet, tmp_path / "sheet.assignment.json", ["fable", "astra"], tmp_path / "records")
    m1.close()
