"""scripts/14_llm_label.py: a model labels its own copy under the guide, in
sheet order, through the labelling tool's writer; the prompts carry what a
human sees and nothing more; bad replies are refused whole."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

from tests.test_label_ui import ITEMS, ui  # noqa: E402


def _master(path: Path):
    cols = [*ui.BLIND_COLUMNS, "label_ronak", "label_bhagyesh", "label_akshay", "consensus", "consensus_reason"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for item, cell, q, gt, ans in ITEMS:
            w.writerow({"item": item, "cell": cell, "question": q, "ground_truth": gt, "answer": ans})


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "14_llm_label.py"), *map(str, args)], cwd=ROOT, env=ENV, capture_output=True, text=True)


def test_emit_and_ingest_in_order_and_refusals(tmp_path):
    sheet = tmp_path / "sheet.csv"
    _master(sheet)
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n\n## perception/VQA\n\nevery object\n")
    out = tmp_path / "llm_fable"
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--batch", "2", "--emit")
    assert r.returncode == 0, r.stdout + r.stderr
    copy = tmp_path / "sheet.fable.csv"
    assert copy.exists()
    with copy.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [*ui.BLIND_COLUMNS, "label_fable"] and len(list(reader)) == 5
    prompts = sorted(out.glob("batch_*.prompt.txt"))
    assert [p.name for p in prompts] == ["batch_01.prompt.txt", "batch_02.prompt.txt", "batch_03.prompt.txt"]
    p1 = prompts[0].read_text()
    assert "every object" in p1 and "--- ITEM D001 ---" in p1 and "--- ITEM D002 ---" in p1 and "D003" not in p1
    assert "Do not open any file" in p1 and "precision" not in p1.lower() and "score" not in p1.lower().replace("scores", "")
    assert json.loads((out / "batch_03.items.json").read_text()) == ["D005"]
    # a human column cannot be a model labeller
    r = _run("--labeller", "ronak", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--emit")
    assert r.returncode == 1 and "must not be a human column" in r.stderr
    # replies: batch 2 before batch 1 is refused (not the next unlabelled rows); then in order, with a fence and prose tolerated
    (out / "batch_02.reply.txt").write_text('[{"item": "D003", "label": 1, "reason": "r", "ambiguous": false}, {"item": "D004", "label": 0, "reason": "r"}]')
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--ingest", out / "batch_02.reply.txt")
    assert r.returncode == 1 and "not the next unlabelled rows" in r.stderr
    (out / "batch_01.reply.txt").write_text('Here you go:\n```json\n[{"item": "D001", "label": 1, "reason": "both objects", "ambiguous": false},\n {"item": "D002", "label": 0, "reason": "speed", "ambiguous": true}]\n```\n')
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--ingest", out / "batch_01.reply.txt", out / "batch_02.reply.txt")
    assert r.returncode == 0, r.stdout + r.stderr
    with copy.open(encoding="utf-8", newline="") as fh:
        assert [r_["label_fable"] for r_ in csv.DictReader(fh)] == ["1", "0", "1", "0", ""]
    rec = json.loads((out / "labels.json").read_text())
    assert rec["items"]["D002"] == {"label": 0, "reason": "speed", "ambiguous": True, "reply": "batch_01.reply.txt"} and rec["items"]["D004"]["ambiguous"] is False
    # the master sheet is untouched and 12 refuses to merge the model's copy
    with sheet.open(encoding="utf-8", newline="") as fh:
        assert all(r_["label_ronak"] == "" and r_["consensus"] == "" for r_ in csv.DictReader(fh))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "12_merge_label_copies.py"), "--sheet", str(sheet), str(copy)], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "not on the master sheet" in r.stderr
    # a wrong label, a wrong item set, and a reply already ingested
    (out / "batch_03.reply.txt").write_text('[{"item": "D005", "label": 2, "reason": "x"}]')
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--ingest", out / "batch_03.reply.txt")
    assert r.returncode == 1 and "is not 0 or 1" in r.stderr
    (out / "batch_03.reply.txt").write_text('[{"item": "D009", "label": 1, "reason": "x"}]')
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--ingest", out / "batch_03.reply.txt")
    assert r.returncode == 1 and "differ from the batch" in r.stderr
    with copy.open(encoding="utf-8", newline="") as fh:
        assert [r_["label_fable"] for r_ in csv.DictReader(fh)] == ["1", "0", "1", "0", ""]
    r = _run("--labeller", "fable", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--ingest", out / "batch_01.reply.txt")
    assert r.returncode == 1 and "nothing to ingest" not in r.stderr and "0 this run" in r.stdout or r.returncode == 0


def test_codex_backend_runs_each_batch_from_an_empty_folder(tmp_path):
    sheet = tmp_path / "sheet.csv"
    _master(sheet)
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n")
    fake = tmp_path / "fakecodex.py"
    # a stand-in for `codex exec ... -o reply -`: records its cwd and args, answers every item in the prompt with 1
    fake.write_text('''import sys, re, json, os
args = sys.argv[1:]
out = args[args.index("-o") + 1]
prompt = sys.stdin.read()
ids = re.findall(r"--- ITEM (\\S+) ---", prompt)
assert "read-only" in args and "--json" in args and any(a.startswith("model_reasoning_effort=") for a in args)
with open(out, "w") as fh:
    json.dump([{"item": i, "label": 1, "reason": "cwd=" + os.getcwd(), "ambiguous": False} for i in ids], fh)
''')
    out = tmp_path / "llm_astra"
    rel_out = os.path.relpath(out, ROOT)  # relative, as on the command line: the backend runs in another folder and must still find the reply path
    r = _run("--labeller", "astra", "--sheet", sheet, "--guide", guide, "--out-dir", rel_out, "--batch", "2", "--backend", "codex", "--effort", "high", "--codex-cmd", f"{sys.executable} {fake}", "--parallel", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 batch(es) to run" in r.stdout and "5 / 5 labelled" in r.stdout
    rec = json.loads((out / "labels.json").read_text())
    assert rec["backend"] == "codex" and rec["effort"] == "high" and len(rec["items"]) == 5
    assert all(str(ROOT) not in v["reason"] for v in rec["items"].values())  # the model ran outside the repository
    # a second run has nothing left to do and changes nothing
    r = _run("--labeller", "astra", "--sheet", sheet, "--guide", guide, "--out-dir", out, "--batch", "2", "--backend", "codex", "--codex-cmd", f"{sys.executable} {fake}")
    assert r.returncode == 0 and "0 batch(es) to run" in r.stdout and "5 / 5 labelled (0 this run)" in r.stdout
