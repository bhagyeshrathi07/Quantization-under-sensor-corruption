"""The local labelling tool (scripts/13_label_ui.py): writes one person's
column only, in sheet order, through a backed-up atomic rewrite; the blind
text comes back byte for byte."""
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

spec = importlib.util.spec_from_file_location("label_ui", ROOT / "scripts" / "13_label_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

ITEMS = [
    ("D001", "perception/VQA", "What are the important objects?", 'There is a "white van", to the front.', "A white van\nand a black car, parked."),
    ("D002", "planning/VQA", "Safe actions?", "Keep going at the same speed, continue ahead.", "Maintain speed; stay in lane."),
    ("D003", "prediction/VQA", "Notice first, second, third?", "Firstly, the truck. Secondly, the pedestrians.", "Truck, then pedestrians, then a sedan ✓"),
    ("D004", "planning/VQA", "Collision with <c1,CAM_BACK,0.5,0.5>?", "Back up.", "Reversing would hit the vehicle behind."),
    ("D005", "perception/VQA", "What are the important objects?", "A black sedan to the back.", ""),
]


def write_copy(path: Path, name: str, labels=None):
    cols = [*ui.BLIND_COLUMNS, f"label_{name}"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, (item, cell, q, gt, ans) in enumerate(ITEMS):
            w.writerow({"item": item, "cell": cell, "question": q, "ground_truth": gt, "answer": ans, f"label_{name}": "" if not labels or labels[i] is None else str(labels[i])})


def blind_of(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return [[r[c] for c in ui.BLIND_COLUMNS] for r in csv.DictReader(fh)]


class Client:
    def __init__(self, server):
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def page(self):
        with urllib.request.urlopen(self.base + "/") as r:
            return r.read().decode()


@pytest.fixture
def served(tmp_path):
    copy = tmp_path / "sheet.ronak.csv"
    write_copy(copy, "ronak")
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n\n## Common rules\n\n- be strict\n\n## perception/VQA\n\nevery object the reference names\n\n## planning/VQA\n\nthe action\n")
    c = ui.Copy(copy, "ronak")
    try:
        server = ui.serve(c, guide, port=0)
    except PermissionError:
        c.close()
        pytest.skip("cannot open local sockets in this environment")
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield copy, c, Client(server)
    server.shutdown()
    server.server_close()
    c.close()


def test_labels_are_saved_in_order_with_backup_and_untouched_text(served):
    copy, c, cl = served
    before = blind_of(copy)
    original_bytes = copy.read_bytes()
    status, st = cl.get("/api/state")
    assert status == 200 and st["n"] == 5 and st["labelled"] == 0 and st["next"] == 0 and st["items"][0]["label"] is None
    assert st["items"][2]["answer"].endswith("✓") and "\n" in st["items"][0]["answer"]
    # a gap is refused
    status, j = cl.post("/api/label", {"index": 2, "item": "D003", "label": 1})
    assert status == 409 and "labelled in sheet order" in j["error"]
    assert copy.read_bytes() == original_bytes and not (copy.parent / "sheet.ronak.csv.bak").exists()
    # the first item, then the second
    status, j = cl.post("/api/label", {"index": 0, "item": "D001", "label": 1})
    assert status == 200 and j == {"item": "D001", "label": 1, "labelled": 1, "n": 5, "changed": False}
    status, j = cl.post("/api/label", {"index": 1, "item": "D002", "label": 0})
    assert status == 200 and j["labelled"] == 2
    bak = copy.parent / "sheet.ronak.csv.bak"
    assert bak.exists() and not (copy.parent / "sheet.ronak.csv.tmp").exists()
    with copy.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["label_ronak"] for r in rows] == ["1", "0", "", "", ""]
    assert blind_of(copy) == before
    # the backup holds the previous version (one label)
    with bak.open(encoding="utf-8", newline="") as fh:
        assert [r["label_ronak"] for r in csv.DictReader(fh)] == ["1", "", "", "", ""]
    # a revision of an earlier item is allowed and reported as a change
    status, j = cl.post("/api/label", {"index": 0, "item": "D001", "label": 0})
    assert status == 200 and j["changed"] is True and j["labelled"] == 2
    # the blind text is written back byte for byte: only the label column differs from the original file
    orig_lines = original_bytes.decode("utf-8").split("\r\n")
    new_lines = copy.read_bytes().decode("utf-8").split("\r\n")
    assert len(orig_lines) == len(new_lines)
    diff = [(a, b) for a, b in zip(orig_lines, new_lines) if a != b]
    assert all(a + "0" == b or a.rsplit(",", 1)[0] == b.rsplit(",", 1)[0] for a, b in diff)
    # bad requests
    assert cl.post("/api/label", {"index": 1, "item": "D002", "label": 2})[0] == 409
    assert cl.post("/api/label", {"index": "1", "item": "D002", "label": 1})[0] == 400
    assert cl.post("/api/label", {"index": 1, "item": "D009", "label": 1})[0] == 409  # stale page
    assert cl.post("/api/label", {"index": 7, "item": None, "label": 1})[0] == 409
    # the page carries the guide sections and the labeller, nothing from a key
    page = cl.page()
    assert "every object the reference names" in page and ">ronak<" in page
    assert 'split("\\n")' in page and "/^(#{1,6})\\s+(.*)$/" in page  # the template is a raw string: the renderer's escapes reach the browser intact and "precision" not in page.lower().replace("precision_", "")
    # a second instance on the same copy is refused while the first is open (review of November 16: it would overwrite the first one's labels)
    with pytest.raises(ui.CopyError, match="another instance"):
        ui.Copy(copy, "ronak")
    # a file changed on disk by something else is not overwritten
    with copy.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label_ronak"] = "1"
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=[*ui.BLIND_COLUMNS, "label_ronak"])
    w.writeheader(); w.writerows(rows)
    copy.write_text(buf.getvalue(), encoding="utf-8", newline="")
    status, j = cl.post("/api/label", {"index": 2, "item": "D003", "label": 1})
    assert status == 409 and "changed on disk" in j["error"]
    with copy.open(encoding="utf-8", newline="") as fh:
        assert [r["label_ronak"] for r in csv.DictReader(fh)] == ["1", "0", "", "", ""]  # the external edit stands, nothing of ours was written
    # after the first instance closes, a fresh load resumes at the first unlabelled item and reads the labels back
    c.close()
    lock = copy.parent / "sheet.ronak.csv.lock"
    inode = lock.stat().st_ino
    c2 = ui.Copy(copy, "ronak")
    assert c2.labels == [1, 0, None, None, None] and c2.state()["next"] == 2
    # shutdown and restart keep one lock file: the file stays in place (deleting it would let two later instances lock different inodes)
    assert lock.exists() and lock.stat().st_ino == inode
    with pytest.raises(ui.CopyError, match="another instance"):
        ui.Copy(copy, "ronak")
    c2.close()
    assert lock.exists() and lock.stat().st_ino == inode
    c3 = ui.Copy(copy, "ronak")
    c3.close()


def test_copy_is_validated_before_serving(tmp_path):
    p = tmp_path / "sheet.ronak.csv"
    # a labelled row after a blank one
    write_copy(p, "ronak", labels=[1, None, 1, None, None])
    with pytest.raises(ui.CopyError, match="labelled in sheet order"):
        ui.Copy(p, "ronak")
    # somebody else's copy
    write_copy(p, "akshay")
    with pytest.raises(ui.CopyError, match="not ronak's copy"):
        ui.Copy(p, "ronak")
    # a label that is not 0, 1 or empty
    write_copy(p, "ronak", labels=[1, None, None, None, None])
    text = ui.read_raw(p)
    assert ",1\r\n" in text
    p.write_text(text.replace(",1\r\n", ",yes\r\n", 1), encoding="utf-8", newline="")
    with pytest.raises(ui.CalibrationError, match="not 0, 1 or empty"):
        ui.Copy(p, "ronak")
    # a file another program rewrote (quoting changed) is refused rather than rewritten
    write_copy(p, "ronak")
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=[*ui.BLIND_COLUMNS, "label_ronak"], quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(rows)
    p.write_text(buf.getvalue(), encoding="utf-8", newline="")
    with pytest.raises(ui.CopyError, match="does not round-trip"):
        ui.Copy(p, "ronak")
    # the real copies pass
    for name in ("ronak", "bhagyesh", "akshay"):
        real = ROOT / "manifests" / f"judge_label_sheet_2026-11-17.{name}.csv"
        if real.exists():
            try:
                c = ui.Copy(real, name, ui.default_assignment(real.with_name("judge_label_sheet_2026-11-17.csv")))  # the real copies follow the sheet's assignment
            except ui.CopyError as e:
                if "another instance" in str(e):
                    continue  # that labeller has the tool open right now; their copy is theirs
                raise
            assert len(c.rows) == 262 and c.cols == [*ui.BLIND_COLUMNS, f"label_{name}"]
            c.close()


def test_check_mode_and_guide_sections(tmp_path):
    p = tmp_path / "sheet.ronak.csv"
    write_copy(p, "ronak", labels=[1, 0, None, None, None])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "13_label_ui.py"), "--labeller", "ronak", "--copy", p, "--assignment", "none", "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "5 items for ronak (D001 to D005, every item), 2 labelled, next D003" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "13_label_ui.py"), "--labeller", "akshay", "--copy", p, "--assignment", "none", "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "not akshay's copy" in r.stderr
    sections = ui.guide_sections(ROOT / "manifests" / "labelling_guide.md")
    for cell in ("perception/VQA", "prediction/VQA", "planning/VQA"):
        assert cell in sections and sections[cell], cell
    assert ui.guide_sections(tmp_path / "missing.md")["full"].startswith("(guide")
