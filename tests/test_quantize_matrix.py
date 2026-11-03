"""Offline checks for scripts/quantize_awq.py (dry run) and scripts/run_matrix.sh (dry run)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _annotation(scene, frame, qtype, tag, question, cams=("CAM_FRONT", "CAM_BACK")):
    return {
        "scene_token": scene, "frame_token": frame, "question_type": qtype, "tag": tag,
        "question": question, "answer": "A",
        "image_path": {c: f"data/nuscenes/samples/{c}/{scene}_{frame}.jpg" for c in cams},
    }


def _fixture(tmp_path):
    ann = [
        _annotation("s1" * 16, "f1" * 16, "behavior", [0], "Predict the behavior. A. go B. stop"),
        _annotation("s1" * 16, "f1" * 16, "planning", [1], "What should the ego do with <c1,CAM_FRONT,1,1>?"),
        _annotation("s2" * 16, "f2" * 16, "perception", [0], "What is <c1,CAM_BACK,1,1>? A. car B. bus"),
        _annotation("s3" * 16, "f3" * 16, "prediction", [0], "dropped cell"),
        _annotation("s4" * 16, "f4" * 16, "behavior", [0], "development scene; not calibration"),
    ]
    (tmp_path / "ann.json").write_text(json.dumps(ann))
    (tmp_path / "split.json").write_text(json.dumps({"split": {"calibration": ["s1" * 16, "s2" * 16, "s3" * 16], "development": ["s4" * 16], "final": []}}))
    cfg = yaml.safe_load((ROOT / "config/pilot.yaml").read_text())
    cfg["data"]["annotations"] = str(tmp_path / "ann.json")
    cfg["prompt"]["system_file"] = str(ROOT / "prompt.txt")
    cfg["quantize"]["out_dir"] = str(tmp_path / "ckpt")
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return cfg


def test_quantize_dry_run_manifest(tmp_path):
    cfg = _fixture(tmp_path)
    q = _load("quantize_awq.py")
    qs = q.calibration_questions(cfg, tmp_path / "split.json", 0)
    assert sorted(x.cell for x in qs) == ["behavior/MCQ", "perception/MCQ", "planning/VQA"]  # dropped cell and dev scene excluded
    messages, paths = q.calibration_sample(qs[0], cfg, (ROOT / "prompt.txt").read_text())
    assert sum(c["type"] == "image" for c in messages[0]["content"]) == len(paths) == qs[0].n_views
    text = messages[0]["content"][-1]["text"]
    assert text.endswith(cfg["prompt"]["mcq_suffix"]) if qs[0].is_mcq else not text.endswith(cfg["prompt"]["mcq_suffix"])
    recipe = q.build_recipe(cfg["quantize"], ["re:.*layers\\.3\\..*"])
    assert recipe["weights"]["group_size"] == 128 and recipe["weights"]["symmetric"] is False
    assert "lm_head" in recipe["ignore"] and recipe["ignore"][-1] == "re:.*layers\\.3\\..*"

    out = subprocess.run(
        [sys.executable, "scripts/quantize_awq.py", "--config", str(tmp_path / "cfg.yaml"), "--split-file", str(tmp_path / "split.json"),
         "--manifest-dir", str(tmp_path / "manifests"), "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    m = json.loads((tmp_path / "manifests" / "quantize_ckpt.json").read_text())
    assert m["dry_run"] is True and m["calibration"]["n_samples"] == 3 and len(m["calibration"]["scenes"]) == 2
    assert m["recipe"]["scheme"] == "W4A16_ASYM" and m["revision"].startswith("cc594898")
    assert "3 calibration questions" in out.stdout


def test_run_matrix_dry_run():
    env = {**os.environ, "DRY_RUN": "1", "CONDITIONS": "clean Fog", "SPLIT": "development", "OUT_DIR": "results/dev", "N": "20"}
    out = subprocess.run(["bash", "scripts/run_matrix.sh"], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    assert out.count("01_serve.sh") == 2
    runs = [l for l in out.splitlines() if "03_smoke_drivebench.py" in l]
    assert len(runs) == 4
    assert "--precision bf16 --condition clean --n 20 --split development --out-dir results/dev" in runs[0]
    assert "--precision int4 --condition Fog" in runs[-1]
    assert out.index("kill server (bf16)") < out.index("--precision int4")
    one = subprocess.run(["bash", "scripts/run_matrix.sh", "int4"], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    assert one.count("01_serve.sh") == 1 and "bf16" not in one.split("matrix start")[1].split("\n", 1)[1]
