"""scripts/17_vqa_tables.py and scripts/18_failure_examples.py format the
report's VQA evidence and the harmful-flip examples from the producers'
files; every number comes from those files."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
BLIND = ("item", "cell", "question", "ground_truth", "answer")


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, env=ENV, capture_output=True, text=True)


def _csv(path, rows, cols):
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def test_vqa_tables_count_consensus_per_cell_precision_and_paired_question(tmp_path):
    # four questions, two cells; D001/D002 are q1 bf16/int4 (1,0: harmful), D003/D004 q2 (0,1: beneficial), D005/D006 q3 (1,1), D007/D008 q4 (0,0)
    items = [("D001", "planning/VQA", "q1", "bf16", 1), ("D002", "planning/VQA", "q1", "int4", 0), ("D003", "planning/VQA", "q2", "bf16", 0), ("D004", "planning/VQA", "q2", "int4", 1),
             ("D005", "perception/VQA", "q3", "bf16", 1), ("D006", "perception/VQA", "q3", "int4", 1), ("D007", "perception/VQA", "q4", "bf16", 0), ("D008", "perception/VQA", "q4", "int4", 0)]
    cols = [*BLIND, "label_ronak", "label_bhagyesh", "consensus", "consensus_reason"]
    rows = [{"item": it, "cell": c, "question": "?", "ground_truth": "g", "answer": "a", "label_ronak": str(v) if i < 4 else "", "label_bhagyesh": str(v) if i >= 4 else "", "consensus": str(v)} for i, (it, c, q, p, v) in enumerate(items)]
    sheet = tmp_path / "sheet.csv"
    _csv(sheet, rows, cols)
    (tmp_path / "sheet.assignment.json").write_text(json.dumps({"assignment": {"ronak": ["D001", "D002", "D003", "D004"], "bhagyesh": ["D005", "D006", "D007", "D008"]}}))
    _csv(tmp_path / "sheet.fable.csv", [dict(r, label_fable="1") for r in rows], [*BLIND, "label_fable"])
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"items": {it: {"question_id": q, "precision": p, "scene_token": "s", "cell": c} for it, c, q, p, v in items}}))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"calibration": {"n_items": 8, "bars": {"agreement": 0.8, "kappa": 0.4, "recall": 0.6, "min_per_class": 15, "min_items": 100}, "accepted": False, "reasons": ["too few"],
                                               "fit": {"threshold": 50, "agreement": 0.75, "kappa": 0.5, "recall_correct": 0.5, "recall_incorrect": 1.0, "n_human_correct": 4, "n_human_incorrect": 4},
                                               "loso": {"pooled": {"agreement": 0.7, "kappa": 0.4, "recall_correct": 0.25, "recall_incorrect": 1.0}, "fold_thresholds": [50, 52]}}}))
    out = tmp_path / "vqa.md"
    r = _run("17_vqa_tables.py", "--sheet", sheet, "--key", key, "--calibration", cal, "--models", "fable", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    md = out.read_text()
    assert "| planning/VQA | 1/2 (50.0%) | 1/2 (50.0%) | 2/4 (50.0%) |" in md and "| all | 2/4 (50.0%) | 2/4 (50.0%) | 4/8 (50.0%) |" in md
    assert "| planning/VQA | 2 | 0 | 1 | 1 | 0 |" in md and "| perception/VQA | 2 | 1 | 0 | 0 | 1 |" in md and "| all | 4 | 1 | 1 | 1 | 1 |" in md
    assert "**NOT ACCEPTED**" in md and "| recall, human-correct | 0.500 | 0.250 | 0.6 |" in md and "fold thresholds 50, 52" in md
    assert "| human (pre-discussion) | 8 | 8 | 4 |" in md and "| fable | 4 | 8 | 8 |" in md and "| consensus | 8 | 8 | 4 |" in md
    # an item without a consensus is refused
    rows[0]["consensus"] = ""
    _csv(sheet, rows, cols)
    r = _run("17_vqa_tables.py", "--sheet", sheet, "--key", key, "--out", out)
    assert r.returncode == 1 and "has no consensus" in r.stderr


def test_failure_examples_pick_low_margin_first_with_caps(tmp_path):
    # a tiny annotation file and answers for three harmful flips
    qs = []
    for i, (qid, qtype, tag, view) in enumerate((("q1", "perception", [0], {"CAM_FRONT": "nuscenes/samples/CAM_FRONT/a.jpg"}),
                                                  ("q2", "behavior", [0], {"CAM_FRONT": "nuscenes/samples/CAM_FRONT/b.jpg", "CAM_BACK": "nuscenes/samples/CAM_BACK/b.jpg"}),
                                                  ("q3", "behavior", [0], {"CAM_FRONT": "nuscenes/samples/CAM_FRONT/c.jpg"}))):
        qs.append({"scene_token": "s" * 32, "frame_token": "f", "question_type": qtype, "tag": tag, "question": f"Stem {qid}? Please select the correct answer from the following options: A. Alpha. B. Beta. C. Gamma.", "answer": "A", "image_path": view})
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(qs))
    import yaml
    cfg = yaml.safe_load((ROOT / "config/pilot.yaml").read_text())
    cfg["data"]["annotations"] = str(ann)
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    sys.path.insert(0, str(ROOT / "src"))
    from qflip.drivebench import load_questions
    ids = [q.question_id for q in load_questions(ann)]
    res = tmp_path / "res"
    res.mkdir()
    def rec(qid, prec, cond, pred, probs):
        return {"question_id": qid, "scene_token": "s" * 32, "condition": cond, "precision": prec, "cell": "perception/MCQ" if qid == ids[0] else "behavior/MCQ", "question": "x", "gt": "A", "pred": pred + ". text", "error": None,
                "mcq_pred_strict": pred, "mcq_gt": "A", "option_probs": probs}
    for cond in ("clean", "Fog"):
        (res / f"bf16_{cond}.jsonl").write_text("".join(json.dumps(rec(q, "bf16", cond, "A", {"A": 0.5, "B": 0.4, "C": 0.1})) + "\n" for q in ids))
        (res / f"int4_{cond}.jsonl").write_text("".join(json.dumps(rec(q, "int4", cond, "B", {"A": 0.3, "B": 0.6, "C": 0.1})) + "\n" for q in ids))
    comp = tmp_path / "compare.json"
    comp.write_text(json.dumps({"harmful_examples": [
        {"question_id": ids[0], "scene_token": "s" * 32, "condition": "clean", "cell": "perception/MCQ", "n_views": 1, "margin": 0.30},
        {"question_id": ids[1], "scene_token": "s" * 32, "condition": "Fog", "cell": "behavior/MCQ", "n_views": 2, "margin": 0.05},
        {"question_id": ids[2], "scene_token": "s" * 32, "condition": "Fog", "cell": "behavior/MCQ", "n_views": 1, "margin": 0.10}]}))
    out, js = tmp_path / "ex.md", tmp_path / "ex.json"
    r = _run("18_failure_examples.py", "--compare", comp, "--results", res, "--config", cfgp, "--images-root", tmp_path / "img", "--limit", "2", "--per-condition", "1", "--out", out, "--json", js)
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads(js.read_text())
    # lowest margin first (q2 at 0.05), then the Fog cap of 1 skips q3 (0.10) and takes q1 (clean)
    assert [e["question_id"] for e in data["examples"]] == [ids[1], ids[0]] and data["n_harmful"] == 3
    e = data["examples"][0]
    assert e["correct"] == "A" and e["bf16"] == "A" and e["int4"] == "B" and e["options"] == {"A": "Alpha.", "B": "Beta.", "C": "Gamma."} and e["stem"].startswith("Stem")
    assert e["images"]["CAM_BACK"] == "corruption/Fog/CAM_BACK/b.jpg" and e["images_present"]["CAM_BACK"] is False
    assert sorted(data["missing_images"]) == ["corruption/Fog/CAM_BACK/b.jpg", "corruption/Fog/CAM_FRONT/b.jpg", "nuscenes/samples/CAM_FRONT/a.jpg"]
    md = out.read_text()
    assert "## 1. behavior/MCQ, Fog, margin 0.050" in md and "- A. Alpha. (correct)" in md and "BF16 answered **A**" in md and "INT4 answered **B**" in md
