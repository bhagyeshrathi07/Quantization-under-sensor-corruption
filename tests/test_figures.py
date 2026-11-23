"""scripts/19_figures.py draws the report figure from 05's JSON; the numbers on it are the JSON's."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fh_by_margin_figure_carries_the_json_numbers(tmp_path):
    comp = {"mcq": {"groups": {"margin": [
        {"label": "[0,0.25)", "n": 100, "ref_acc": 0.40, "harmful": 10, "harmful_rate": 0.25, "ci": {"harmful_rate": [0.15, 0.35]}},
        {"label": "[0.25,0.5)", "n": 50, "ref_acc": 0.50, "harmful": 1, "harmful_rate": 0.04, "ci": {"harmful_rate": [0.0, 0.1]}},
        {"label": "[0.5,1]", "n": 20, "ref_acc": 0.20, "harmful": 0, "harmful_rate": 0.0, "ci": {"harmful_rate": [0.0, 0.0]}}]}}}
    cj = tmp_path / "compare.json"
    cj.write_text(json.dumps(comp))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "19_figures.py"), "--compare", str(cj), "--out-dir", str(tmp_path / "fig")], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    svg = (tmp_path / "fig" / "fh_by_margin.svg").read_text()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "0.250 (10/40)" in svg and "0.040 (1/25)" in svg and "0.000 (0/4)" in svg
    assert "100 pairs" in svg and "exploratory, pooled" in svg and "[0, 0]" in svg
    # no margin groups: refused
    cj.write_text(json.dumps({"mcq": {"groups": {}}}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "19_figures.py"), "--compare", str(cj), "--out-dir", str(tmp_path / "fig")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "no margin groups" in r.stderr
