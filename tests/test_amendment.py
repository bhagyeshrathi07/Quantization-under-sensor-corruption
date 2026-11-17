"""The VQA amendment (November 16, 2026): a second freeze that differs from the
core freeze in generation.max_tokens alone, covers the VQA cells alone, and is
enforced by the run guard, the analysis validation and the calibration guard
exactly as the core freeze is."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qflip.freeze import FreezeError, amendment, build_amendment_frozen, build_frozen, check_amendment, check_amendment_config, freeze_identity, load_frozen
from qflip.guard import prepare_calibration_files, prepare_run, validate_analysis_files
from qflip.provenance import ProvenanceError, ensure_meta
from qflip.runner import filter_cells

from tests.test_provenance_freeze import S, _manifest, _meta, world  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _core(w, tmp_path):
    fp = tmp_path / "config_frozen.yaml"
    fp.write_text(yaml.safe_dump(build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], requirements_file=w["man"] / "requirements_frozen.txt", protocol_snapshot=w["man"] / "PROTOCOL_frozen.md"), sort_keys=False))
    return fp


def _cfg512(w, **over):
    cfg = yaml.safe_load(yaml.safe_dump(w["cfg"]))
    cfg["generation"]["max_tokens"] = 512
    for k, v in over.items():
        section, key = k.split(".")
        cfg[section][key] = v
    return cfg


def _amend(w, tmp_path, cfg=None):
    base = load_frozen(_core(w, tmp_path))
    fp = tmp_path / "config_frozen_vqa512.yaml"
    fp.write_text(yaml.safe_dump(build_amendment_frozen(cfg or _cfg512(w), "vqa512", base, w["split"], w["excl"], git_state=("deadbeef", False), base_path=tmp_path / "config_frozen.yaml"), sort_keys=False))
    return fp


def test_amendment_differs_from_the_core_freeze_in_the_allowed_setting_alone(world, tmp_path):
    w = world
    base = load_frozen(_core(w, tmp_path))
    am = amendment("vqa512")
    assert am["allowed_diffs"] == {("generation", "max_tokens"): 512} and am["cells"] == ("perception/VQA", "prediction/VQA", "planning/VQA")
    frozen = build_amendment_frozen(_cfg512(w), "vqa512", base, w["split"], w["excl"], git_state=("deadbeef", False))
    assert frozen["amendment"]["name"] == "vqa512" and frozen["amendment"]["differs"] == {"generation.max_tokens": {"base": 128, "amendment": 512}}
    assert frozen["amendment"]["base"]["git_commit"] == base["frozen"]["git_commit"] and frozen["amendment"]["base"]["core_config_sha256"] == base["frozen"]["core_config_sha256"]
    assert frozen["core"]["generation"]["max_tokens"] == 512 and frozen["frozen"]["core_config_sha256"] != base["frozen"]["core_config_sha256"]
    for k in ("versions", "builds", "models_frozen"):
        assert frozen[k] == base[k]  # the environment and the models are inherited, never re-frozen
    assert frozen["frozen"]["protocol_snapshot_sha256"] == base["frozen"]["protocol_snapshot_sha256"]
    assert freeze_identity(frozen) == {"date": frozen["frozen"]["date"], "git_commit": "deadbeef", "amendment": "vqa512"}
    assert freeze_identity(base) == {"date": base["frozen"]["date"], "git_commit": base["frozen"]["git_commit"]}
    assert check_amendment(frozen, base) == []
    # the allowed setting at another value, a second changed setting, or no change at all: refused
    assert any("fixes it at 512" in d for d in check_amendment_config(_cfg512(w, **{"generation.max_tokens": 256}), base, am))
    with pytest.raises(FreezeError, match="may change only"):
        build_amendment_frozen(_cfg512(w, **{"serving.max_pixels": 401408}), "vqa512", base, w["split"], w["excl"], git_state=("d", False))
    with pytest.raises(FreezeError, match="requires 512"):
        build_amendment_frozen(w["cfg"], "vqa512", base, w["split"], w["excl"], git_state=("d", False))
    with pytest.raises(FreezeError, match="unknown amendment"):
        amendment("vqa1024")
    # an amendment built on another core freeze does not fit this one
    other = yaml.safe_load(yaml.safe_dump(base))
    other["frozen"]["git_commit"] = "0000000"
    assert any("built on core freeze git_commit" in d for d in check_amendment(frozen, other))
    assert any("is missing" in d for d in check_amendment(frozen, None))
    # inherited fingerprints are compared one by one, and the base file is bound by digest
    for k in ("annotations_sha256", "exclusions_sha256", "prompt_sha256", "protocol_snapshot_sha256", "requirements_sha256", "split_sha256"):
        drifted = yaml.safe_load(yaml.safe_dump(base))
        drifted["frozen"][k] = "f" * 64
        assert any(k in d for d in check_amendment(frozen, drifted)), k
    assert frozen["amendment"]["base"]["sha256"] is None  # built from a dict here, no base file given
    assert any("changed since the amendment was built" in d for d in check_amendment(frozen, base, base_sha256="0" * 64))


def test_run_guard_records_the_amendment_and_refuses_the_core_config_under_it(world, tmp_path):
    w = world
    fp = _amend(w, tmp_path)
    cfg512 = _cfg512(w)
    cfg_path = tmp_path / "vqa512.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg512))
    kw = dict(precision="int4", condition="clean", split="development", split_file=w["split"], exclusions_file=w["excl"], manifests_dir=w["man"], check_live_server=False, frozen_path=fp, amendment="vqa512")
    meta = prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "dev/int4_clean.jsonl", require_freeze=True, **kw)
    assert meta["freeze_checked"] == {"date": load_frozen(fp)["frozen"]["date"], "git_commit": "deadbeef", "amendment": "vqa512"}
    # the core config (128) is refused against the amendment freeze, and the amendment config against the core freeze
    with pytest.raises(FreezeError, match="max_tokens"):
        prepare_run(w["cfg"], config_path=w["cfg_path"], result_file=tmp_path / "bad/int4_clean.jsonl", require_freeze=True, **kw)
    with pytest.raises(FreezeError, match="no amendment block"):
        prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "bad2/int4_clean.jsonl", require_freeze=True, **{**kw, "frozen_path": tmp_path / "config_frozen.yaml"})
    with pytest.raises(FreezeError, match="max_tokens"):
        prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "bad2b/int4_clean.jsonl", require_freeze=True, **{**kw, "frozen_path": tmp_path / "config_frozen.yaml", "amendment": None})
    # the amendment freeze used without naming the amendment, or without requiring the freeze, is refused
    with pytest.raises(FreezeError, match="select it with --amendment"):
        prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "bad2c/int4_clean.jsonl", require_freeze=True, **{**kw, "amendment": None})
    with pytest.raises(FreezeError, match="must require its freeze"):
        prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "bad2d/int4_clean.jsonl", **kw)
    # resume in the other direction (review of November 17, item 2): an unchecked development run does not continue the amendment file
    ok_file = tmp_path / "dev/int4_clean.jsonl"
    ok_file.write_text(json.dumps({"question_id": "q0", "scene_token": S[3], "precision": "int4", "condition": "clean", "cell": "planning/VQA"}) + "\n")
    with pytest.raises(FreezeError, match="resumed only by a run under the same freeze check"):
        prepare_run(cfg512, config_path=cfg_path, result_file=ok_file, **{**kw, "frozen_path": tmp_path / "config_frozen.yaml", "amendment": None})
    assert json.loads((tmp_path / "dev/int4_clean.jsonl.meta.json").read_text())["freeze_checked"]["amendment"] == "vqa512"
    # resume: a file written without the check (the exploratory 512 pilot, MCQ rows included) is not adopted by the amendment run
    pilot = tmp_path / "pilot/int4_clean.jsonl"
    pilot.parent.mkdir()
    ensure_meta(pilot, {k: v for k, v in _meta(w, precision="int4").items()} | {"core_config_sha256": __import__("qflip.provenance", fromlist=["core_config_sha256"]).core_config_sha256(cfg512)})
    pilot.write_text(json.dumps({"question_id": "q0", "scene_token": S[3], "precision": "int4", "condition": "clean", "cell": "behavior/MCQ"}) + "\n")
    with pytest.raises(FreezeError, match="resumed only by a run under the same freeze check"):
        prepare_run(cfg512, config_path=cfg_path, result_file=pilot, require_freeze=True, **kw)
    assert "freeze_checked" not in json.loads((pilot.parent / "int4_clean.jsonl.meta.json").read_text())
    # a file under the right identity but holding an MCQ row is refused as well
    ident = {"date": load_frozen(fp)["frozen"]["date"], "git_commit": "deadbeef", "amendment": "vqa512"}
    mixed = tmp_path / "mixed/int4_clean.jsonl"
    mixed.parent.mkdir()
    ensure_meta(mixed, _meta(w, precision="int4") | {"core_config_sha256": __import__("qflip.provenance", fromlist=["core_config_sha256"]).core_config_sha256(cfg512), "freeze_checked": ident})
    mixed.write_text(json.dumps({"question_id": "q0", "scene_token": S[3], "precision": "int4", "condition": "clean", "cell": "behavior/MCQ"}) + "\n")
    with pytest.raises(ProvenanceError, match="behavior/MCQ"):
        prepare_run(cfg512, config_path=cfg_path, result_file=mixed, require_freeze=True, **kw)
    # an amendment freeze whose base has moved on is refused
    am = yaml.safe_load(fp.read_text())
    am["amendment"]["base"]["git_commit"] = "0000000"
    fp2 = tmp_path / "stale_amendment.yaml"
    fp2.write_text(yaml.safe_dump(am))
    with pytest.raises(FreezeError, match="does not fit the core freeze"):
        prepare_run(cfg512, config_path=cfg_path, result_file=tmp_path / "bad3/int4_clean.jsonl", require_freeze=True, **{**kw, "frozen_path": fp2, "amendment": "vqa512"})


def _files(w, d, cfg_path, rows_by_precision, split="development", **meta_over):
    out = []
    for p, rows in rows_by_precision.items():
        m = _meta(w, precision=p, split=split, **meta_over)
        m["core_config_sha256"] = __import__("qflip.provenance", fromlist=["core_config_sha256"]).core_config_sha256(yaml.safe_load(cfg_path.read_text()))
        f = d / f"{p}_clean.jsonl"
        d.mkdir(parents=True, exist_ok=True)
        ensure_meta(f, m)
        f.write_text("".join(json.dumps({"question_id": f"q{i}", "scene_token": S[3], "precision": p, "condition": "clean", **r}) + "\n" for i, r in enumerate(rows)))
        out.append(f)
    return out


def test_analysis_and_calibration_accept_amendment_files_only_under_the_amendment(world, tmp_path, monkeypatch):
    w = world
    fp = _amend(w, tmp_path)
    core_fp = tmp_path / "config_frozen.yaml"
    cfg512 = _cfg512(w)
    cfg_path = tmp_path / "vqa512.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg512))
    ident = freeze_identity(load_frozen(fp))
    vqa_rows = [{"cell": "planning/VQA"}, {"cell": "perception/VQA"}]
    files = _files(w, tmp_path / "dev_vqa512", cfg_path, {"bf16": vqa_rows, "int4": vqa_rows}, freeze_checked=ident)
    monkeypatch.setattr("qflip.guard.check_records_against_meta", lambda *a, **k: 0)
    # under the amendment: accepted, and labelled with it
    prov = validate_analysis_files(files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    assert prov["amendment"] == "vqa512" and prov["analysis_class"] == "exploratory"
    cal = prepare_calibration_files(files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    assert cal["analysis_class"] == "calibration" and cal["amendment"] == "vqa512"
    # under the core freeze the same development files are not calibration material: they were produced under the amendment
    with pytest.raises(FreezeError, match="select it with --amendment vqa512"):
        prepare_calibration_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=core_fp)
    # and the amendment's files are never handled as plain core files, on the development split either
    with pytest.raises(FreezeError, match="select it with --amendment vqa512"):
        validate_analysis_files(files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=core_fp)
    # development paths enforce the selected amendment too: 128-token answers copied VQA-only (core identity, core hash), a wrong live
    # config, and an amendment whose base moved are refused on the development split, not labelled "amendment vqa512"
    core_dev = _files(w, tmp_path / "dev_core_copy", w["cfg_path"], {"bf16": vqa_rows, "int4": vqa_rows}, freeze_checked=freeze_identity(load_frozen(core_fp)))
    with pytest.raises(FreezeError, match="freeze_checked"):
        validate_analysis_files(core_dev, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    with pytest.raises(FreezeError, match="max_tokens"):
        validate_analysis_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    with pytest.raises(FreezeError, match="max_tokens"):
        prepare_calibration_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    stale = yaml.safe_load(fp.read_text())
    stale["amendment"]["base"]["git_commit"] = "0000000"
    fp_stale = tmp_path / "stale.yaml"
    fp_stale.write_text(yaml.safe_dump(stale))
    with pytest.raises(FreezeError, match="does not fit the core freeze"):
        validate_analysis_files(files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp_stale, amendment="vqa512")
    # a freeze file without the amendment block does not pass as the amendment (review of November 17, item 1)
    noblock = yaml.safe_load(fp.read_text())
    del noblock["amendment"]
    fp_noblock = tmp_path / "noblock.yaml"
    fp_noblock.write_text(yaml.safe_dump(noblock))
    with pytest.raises(FreezeError, match="no amendment block"):
        validate_analysis_files(fin_like := files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp_noblock, amendment="vqa512")
    with pytest.raises(FreezeError, match="no amendment block"):
        prepare_calibration_files(fin_like, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp_noblock, amendment="vqa512")
    # the core freeze file edited after the amendment was built: the digest binding catches it even when the compared fields agree
    core_text = core_fp.read_text()
    core_fp.write_text(core_text + "# edited\n")
    with pytest.raises(FreezeError, match="changed since the amendment was built"):
        validate_analysis_files(files, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    core_fp.write_text(core_text)
    # an MCQ row under the amendment is a breach
    mixed = _files(w, tmp_path / "mixed", cfg_path, {"bf16": vqa_rows + [{"cell": "behavior/MCQ"}], "int4": vqa_rows}, freeze_checked=ident)
    with pytest.raises(ProvenanceError, match="behavior/MCQ.*amendment vqa512"):
        validate_analysis_files(mixed, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    # final split: the amendment files pass under the amendment and fail under the core freeze (core hash), core files fail under the amendment
    fin = _files(w, tmp_path / "final_vqa512", cfg_path, {"bf16": vqa_rows, "int4": vqa_rows}, split="final", freeze_checked=ident)
    out = validate_analysis_files(fin, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")
    # final population, but the amendment is a secondary measure: never "confirmatory" (review of November 16, fourth pass)
    assert out["analysis_class"] == "exploratory" and out["amendment"] == "vqa512" and out["split"] == "final" and out["frozen"]["git_commit"] == "deadbeef"
    with pytest.raises(FreezeError, match="select it with --amendment vqa512"):
        validate_analysis_files(fin, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=core_fp)
    # a final amendment file whose sidecar lost its identity would still fail the core hash under the core freeze
    stripped = [json.loads(Path(str(f) + ".meta.json").read_text()) for f in fin]
    for f, m in zip(fin, stripped):
        m.pop("freeze_checked")
        Path(str(f) + ".meta.json").write_text(json.dumps(m))
    with pytest.raises(FreezeError, match="core_config_sha256"):
        validate_analysis_files(fin, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=core_fp)
    core_fin = _files(w, tmp_path / "final_core", w["cfg_path"], {"bf16": [{"cell": "planning/VQA"}], "int4": [{"cell": "planning/VQA"}]}, split="final")
    with pytest.raises(FreezeError, match="core_config_sha256"):
        validate_analysis_files(core_fin, cfg512, split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp, amendment="vqa512")


def test_cell_filter_and_the_freeze_script(world, tmp_path):
    from qflip.drivebench import Question

    def q(i, question_type, tag):
        return Question(question_id=f"q{i}", scene_token="s", frame_token="f", question_type=question_type, question="?", answer="A", tag=tag, image_path={}, raw={})

    qs = [q(0, "planning", [1]), q(1, "behavior", [0]), q(2, "perception", [2]), q(3, "prediction", [0])]
    assert [x.cell for x in qs] == ["planning/VQA", "behavior/MCQ", "perception/VQA", None]
    assert [x.question_id for x in filter_cells(qs, "vqa")] == ["q0", "q2"]
    assert [x.question_id for x in filter_cells(qs, "mcq")] == ["q1"]
    assert filter_cells(qs, "all") == qs and filter_cells(qs, None) == qs and filter_cells(qs, ("behavior/MCQ",))[0].question_id == "q1"
    # the real amendment config fits the real core freeze (written to a scratch path; the repo file is written from a clean commit)
    out = tmp_path / "config_frozen_vqa512.yaml"
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out), "--allow-dirty"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "amendment vqa512 on core freeze" in r.stdout and out.exists()
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out), "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "amendment vqa512" in r.stdout, r.stdout + r.stderr
    # the core config does not pass the amendment check
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out), "--config", "config/pilot.yaml", "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "max_tokens" in r.stdout
    # --check uses the runtime loader: a base freeze file edited after the build fails the digest, even with every compared field equal
    base_copy = tmp_path / "base_copy.yaml"
    base_copy.write_text((ROOT / "manifests/config_frozen.yaml").read_text())
    out2 = tmp_path / "amend_on_copy.yaml"
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out2), "--base-frozen", str(base_copy), "--allow-dirty"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out2), "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    base_copy.write_text(base_copy.read_text() + "# edited\n")
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", str(out2), "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "changed since the amendment was built" in r.stdout, r.stdout + r.stderr
    # a plain core freeze given as the amendment, and the amendment file checked as the core freeze, both fail
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--amendment", "vqa512", "--frozen", "manifests/config_frozen.yaml", "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "no amendment block" in r.stdout
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--frozen", str(out), "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 1 and "select it with --amendment" in r.stdout
    # the core check is untouched by the amendment config
    r = subprocess.run([sys.executable, "scripts/freeze_config.py", "--check"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
