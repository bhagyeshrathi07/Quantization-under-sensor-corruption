"""Sidecars, resume compatibility, checkpoint fingerprints, freeze checks,
final-run guard, analysis-side validation, exclusion list."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qflip.freeze import FreezeError, blocking_open_items, build_frozen, check_config, check_serve_manifest, core_config, protocol_open_items
from qflip.guard import final_guard, validate_analysis_files
from qflip.provenance import (
    ProvenanceError,
    build_run_meta,
    checkpoint_fingerprint,
    compat_diff,
    ensure_meta,
    load_meta,
    parse_serve_manifest,
)
from qflip.split import filter_split, load_exclusions, scenes_for

ROOT = Path(__file__).resolve().parents[1]
S = ["s%02d" % i for i in range(20)]


def _ckpt(d: Path, payload: bytes = b"w1") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"quantization_config": {"quant_method": "compressed-tensors"}}')
    (d / "model.safetensors").write_bytes(payload)
    return d


def _manifest(path: Path, precision: str, model: str, revision: str = "", extra: str | None = None, max_pixels=200704, kv=2147483648, date="2026-11-04T00:00:00+00:00",
              seed=0, max_num_seqs=1, prefix_caching="true", port=8000, pid=None):
    extra = extra if extra is not None else ("--dtype bfloat16 --revision abc123" if precision == "bf16" else "--dtype float16 --quantization compressed-tensors")
    path.write_text(
        f"date: {date}\nprecision: {precision}\nmodel: {model}\nrevision: {revision}\nserved_model_name: qwen-{precision}\n"
        f"max_pixels: {max_pixels}\nmax_model_len: 4096\nkv_args: --kv-cache-memory-bytes {kv}\nseed: {seed}\nmax_num_seqs: {max_num_seqs}\n"
        f"prefix_caching: {prefix_caching}\nextra_args: {extra}\nport: {port}\n" + (f"pid: {pid}\n" if pid else "")
        + f"power_limit_w: 450.00\ngpu: RTX 4090\nvllm: 0.33.0\ntransformers: 5.20.2\n"
    )
    return path


@pytest.fixture
def world(tmp_path):
    """A self-contained repo-like directory: config, inputs, split, exclusions, checkpoint, manifests."""
    cfg = yaml.safe_load((ROOT / "config/pilot.yaml").read_text())
    (tmp_path / "data").mkdir()
    ann = tmp_path / "data/ann.json"
    ann.write_text(json.dumps([{"scene_token": s, "frame_token": "f", "question_type": "behavior", "tag": [0], "question": f"q{s}", "answer": "A", "image_path": {"CAM_FRONT": "data/nuscenes/samples/CAM_FRONT/x.jpg"}} for s in S]))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("PROMPT")
    cfg["data"]["annotations"] = str(ann)
    cfg["prompt"]["system_file"] = str(prompt)
    ckpt = _ckpt(tmp_path / "ckpt")
    cfg["quantize"]["out_dir"] = str(ckpt)
    cfg["quantize"]["revision"] = "abc123"
    cfg["quantize"]["base_model"] = "Qwen/base"
    cfg["models"] = {"bf16": "qwen-bf16", "int4": "qwen-int4"}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"split": {"calibration": S[:2], "development": S[2:5], "final": S[5:]}}))
    excl = tmp_path / "exclusions.json"
    excl.write_text(json.dumps({"entries": [{"scene_token": S[5]}, {"scene_token": S[6]}]}))
    man = tmp_path / "manifests"
    man.mkdir()
    (man / "PROTOCOL_frozen.md").write_text("## 2. Serving\n\n| a | 1 | FROZEN |\n")
    _manifest(man / "serve_bf16.txt", "bf16", "Qwen/base", "abc123")
    _manifest(man / "serve_int4.txt", "int4", str(ckpt))
    return {"cfg": cfg, "cfg_path": cfg_path, "split": split, "excl": excl, "ckpt": ckpt, "man": man, "root": tmp_path}


def _meta(w, precision="bf16", condition="clean", split="development", **over):
    m = build_run_meta(split=split, precision=precision, condition=condition, config_path=w["cfg_path"], annotations_path=w["cfg"]["data"]["annotations"],
                       prompt_path=w["cfg"]["prompt"]["system_file"], split_file=w["split"], exclusions_file=w["excl"],
                       serve_manifest_path=w["man"] / f"serve_{precision}.txt", served_model_name=w["cfg"]["models"][precision], repo_root=w["root"])
    m.update(over)
    return m


# ---------------- exclusions / split ----------------

def test_exclusions_apply_to_final_only(world):
    w = world
    assert load_exclusions(w["excl"]) == {S[5], S[6]}
    assert scenes_for("final", w["split"], w["excl"]) == set(S[7:])
    assert scenes_for("final_exposed", w["split"], w["excl"]) == {S[5], S[6]}
    assert scenes_for("development", w["split"], w["excl"]) == set(S[2:5])
    assert scenes_for("final", w["split"], None) == set(S[5:])  # no exclusion file -> plain split

    class Q:
        def __init__(self, s):
            self.scene_token = s

    assert len(filter_split([Q(s) for s in S], "final", w["split"], w["excl"])) == 13
    with pytest.raises(ValueError):
        scenes_for("test", w["split"], w["excl"])


@pytest.mark.skipif(not (ROOT / "data/drivebench-test.json").exists(), reason="annotations not downloaded")
def test_real_population_after_exclusions():
    from qflip.drivebench import load_questions
    from qflip.split import load_split

    qs = load_questions(ROOT / "data/drivebench-test.json")
    split = load_split(ROOT / "manifests/scene_split.json")
    excl = load_exclusions(ROOT / "manifests/scene_exclusions.json")
    assert excl <= split["final"] and len(excl) == 15
    final = filter_split(qs, "final", ROOT / "manifests/scene_split.json", ROOT / "manifests/scene_exclusions.json")
    assert len({q.scene_token for q in final}) == 116
    assert len(final) == 971  # raw annotations, before eligibility routing
    assert sum(q.cell is not None for q in final) == 838


# ---------------- provenance ----------------

def test_checkpoint_fingerprint_changes_with_contents(tmp_path):
    a = checkpoint_fingerprint(str(_ckpt(tmp_path / "a", b"w1")))
    b = checkpoint_fingerprint(str(_ckpt(tmp_path / "b", b"w1")))
    c = checkpoint_fingerprint(str(_ckpt(tmp_path / "c", b"w2")))
    assert a == b and a != c and a.startswith("sha256:")
    assert checkpoint_fingerprint("Qwen/x", "deadbeef") == "hf:Qwen/x@deadbeef"
    with pytest.raises(ProvenanceError):
        checkpoint_fingerprint(str(tmp_path / "empty")) if (tmp_path / "empty").mkdir() is None else None


def test_meta_fields_and_manifest_parse(world):
    m = _meta(world)
    assert m["quantization"] == "none" and m["dtype"] == "bfloat16" and m["model_revision"] == "abc123"
    assert m["checkpoint_fingerprint"] == "hf:Qwen/base@abc123" and m["max_pixels"] == "200704"
    mi = _meta(world, "int4")
    assert mi["quantization"] == "compressed-tensors" and mi["checkpoint_fingerprint"].startswith("sha256:")
    assert parse_serve_manifest(world["man"] / "serve_bf16.txt")["kv_args"] == "--kv-cache-memory-bytes 2147483648"


def test_resume_allows_restart_but_not_changed_settings(world, tmp_path):
    rf = tmp_path / "r" / "bf16_clean.jsonl"
    m1 = _meta(world)
    ensure_meta(rf, m1)  # new file: sidecar written
    assert load_meta(rf)["precision"] == "bf16"
    rf.write_text('{"question_id": "x", "scene_token": "s02", "precision": "bf16", "condition": "clean"}\n')
    # unchanged server restarted: new timestamp, new serve manifest date -> still compatible
    _manifest(world["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123", date="2026-11-05T09:00:00+00:00")
    ensure_meta(rf, _meta(world, created="later"))
    # changed scientific setting -> refused, field named
    _manifest(world["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123", max_pixels=401408)
    with pytest.raises(ProvenanceError, match="max_pixels"):
        ensure_meta(rf, _meta(world))
    # different split or condition -> refused
    with pytest.raises(ProvenanceError, match="split"):
        _manifest(world["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123")
        ensure_meta(rf, _meta(world, split="final"))
    # existing file without sidecar -> refused
    orphan = tmp_path / "r" / "int4_clean.jsonl"
    orphan.write_text('{"question_id": "x"}\n')
    with pytest.raises(ProvenanceError, match="unknown provenance"):
        ensure_meta(orphan, _meta(world, "int4"))
    assert compat_diff(m1, _meta(world, "int4"), ignore=("precision", "served_model_name", "model_path_or_id", "model_revision", "checkpoint_fingerprint", "quantization", "dtype")) == []


def test_resume_inspects_existing_records(world, tmp_path):
    """Review item 6: a development sidecar on a file holding a final-scene record, or a
    record of another precision/condition, must not be appended to."""
    rf = tmp_path / "r" / "bf16_clean.jsonl"
    ensure_meta(rf, _meta(world))
    dev = set(S[2:5])
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "clean"}) + "\n")
    ensure_meta(rf, _meta(world), allowed_scenes=dev)
    rf.write_text(rf.read_text() + json.dumps({"question_id": "q2", "scene_token": S[9], "precision": "bf16", "condition": "clean"}) + "\n")
    with pytest.raises(ProvenanceError, match="outside the development population"):
        ensure_meta(rf, _meta(world), allowed_scenes=dev)
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "int4", "condition": "clean"}) + "\n")
    with pytest.raises(ProvenanceError, match="precision=.int4"):
        ensure_meta(rf, _meta(world), allowed_scenes=dev)
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "Fog"}) + "\n")
    with pytest.raises(ProvenanceError, match="condition=.Fog"):
        ensure_meta(rf, _meta(world), allowed_scenes=dev)


def test_judge_only_config_change_keeps_core_compatibility(world, tmp_path):
    """Review item 7: the compatibility hash covers the core sections only."""
    rf = tmp_path / "r" / "bf16_clean.jsonl"
    ensure_meta(rf, _meta(world))
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "clean"}) + "\n")
    cfg = yaml.safe_load(world["cfg_path"].read_text())
    cfg["judge"]["model"] = "some-other-judge"
    world["cfg_path"].write_text(yaml.safe_dump(cfg))
    m = _meta(world)
    assert m["config_sha256"] != load_meta(rf)["config_sha256"]
    ensure_meta(rf, m, allowed_scenes=set(S[2:5]))  # judge change: resume allowed
    cfg["generation"]["max_tokens"] = 256
    world["cfg_path"].write_text(yaml.safe_dump(cfg))
    with pytest.raises(ProvenanceError, match="core_config_sha256"):
        ensure_meta(rf, _meta(world), allowed_scenes=set(S[2:5]))


# ---------------- freeze ----------------

def test_protocol_open_items_parse_statuses_not_prose():
    text = (
        "# Protocol\n\nEach item is marked FROZEN or TO DECIDE.\n\n## 2. Serving\n\n| Flag | Value | Status |\n|---|---|---|\n"
        "| a | 1 | FROZEN |\n| b | 2 | TO DECIDE: later |\n\n## 7. Scoring\n\n| Item | Rule | Status |\n|---|---|---|\n| VQA | judge | TO DECIDE (team) |\n"
        "## 12. Mitigation\n\n| x | y | TO DECIDE |\n\n## Change log\n\n- 2026-10-19: TO DECIDE items remain\n"
    )
    assert protocol_open_items(text) == [(2, "b"), (7, "VQA"), (12, "x")]
    assert blocking_open_items(text) == [(2, "b")]


def test_freeze_build_check_and_serve_manifest(world):
    w = world
    frozen = build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], protocol_snapshot=w["man"] / "PROTOCOL_frozen.md")
    assert frozen["models_frozen"]["int4"]["checkpoint_fingerprint"].startswith("sha256:")
    assert "judge" not in frozen["core"] and set(core_config(w["cfg"])) >= {"serving", "analysis"}
    assert check_config(w["cfg"], frozen, w["split"], w["excl"]) == []
    changed = json.loads(json.dumps(w["cfg"]))
    changed["serving"]["max_pixels"] = 401408
    assert any("serving.max_pixels" in d for d in check_config(changed, frozen, w["split"], w["excl"]))
    changed = json.loads(json.dumps(w["cfg"]))
    changed["judge"]["model"] = "other"  # judge changes do not touch the core freeze
    assert check_config(changed, frozen, w["split"], w["excl"]) == []
    w["excl"].write_text(json.dumps({"entries": [{"scene_token": S[5]}]}))
    assert any("exclusions_sha256" in d for d in check_config(w["cfg"], frozen, w["split"], w["excl"]))
    w["excl"].write_text(json.dumps({"entries": [{"scene_token": S[5]}, {"scene_token": S[6]}]}))
    # serve manifests
    assert check_serve_manifest(parse_serve_manifest(w["man"] / "serve_bf16.txt"), "bf16", frozen) == []
    assert check_serve_manifest(parse_serve_manifest(w["man"] / "serve_int4.txt"), "int4", frozen) == []
    official = _manifest(w["man"] / "serve_int4_official.txt", "int4", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", extra="--dtype float16 --quantization awq_marlin")
    diffs = check_serve_manifest(parse_serve_manifest(official), "int4", frozen)
    assert any(d.startswith("model:") for d in diffs) and any(d.startswith("quantization:") for d in diffs)
    _ckpt(w["ckpt"], b"overwritten")
    assert any("checkpoint_fingerprint" in d for d in check_serve_manifest(parse_serve_manifest(w["man"] / "serve_int4.txt"), "int4", frozen))
    wrong_rev = _manifest(w["man"] / "serve_bf16_rev.txt", "bf16", "Qwen/base", "other")
    assert any(d.startswith("revision:") for d in check_serve_manifest(parse_serve_manifest(wrong_rev), "bf16", frozen))


def test_final_guard_and_analysis_validation(world, tmp_path):
    w = world
    frozen_path = tmp_path / "config_frozen.yaml"
    # no freeze -> refused
    with pytest.raises(FreezeError, match="missing"):
        final_guard(w["cfg"], "bf16", w["man"] / "serve_bf16.txt", w["split"], w["excl"], frozen_path)
    frozen_path.write_text(yaml.safe_dump(build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], protocol_snapshot=w["man"] / "PROTOCOL_frozen.md")))
    assert final_guard(w["cfg"], "bf16", w["man"] / "serve_bf16.txt", w["split"], w["excl"], frozen_path)["frozen"]
    # official-checkpoint server when the freeze names the own checkpoint -> refused
    official = _manifest(w["man"] / "serve_int4.txt", "int4", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", extra="--dtype float16 --quantization awq_marlin")
    with pytest.raises(FreezeError, match="model:"):
        final_guard(w["cfg"], "int4", official, w["split"], w["excl"], frozen_path)
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]))

    # analysis-side validation on development files
    d = tmp_path / "res"
    files = []
    for p in ("bf16", "int4"):
        f = d / f"{p}_clean.jsonl"
        ensure_meta(f, _meta(w, p))
        f.write_text("\n".join(json.dumps({"question_id": f"q{s}", "scene_token": s, "precision": p, "condition": "clean"}) for s in S[2:5]) + "\n")
        files.append(f)
    out = validate_analysis_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path)
    assert out["split"] == "development" and len(out["files"]) == 2
    # a record outside the population -> refused
    files[1].write_text(files[1].read_text() + json.dumps({"question_id": "qx", "scene_token": S[9], "precision": "int4", "condition": "clean"}) + "\n")
    with pytest.raises(ProvenanceError, match="outside the development population"):
        validate_analysis_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path)
    # missing sidecar -> refused
    orphan = d / "bf16_Fog.jsonl"
    orphan.write_text("{}\n")
    with pytest.raises(ProvenanceError, match="no .meta.json"):
        validate_analysis_files([orphan], w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path)
    # final-split files: require freeze, refuse CLI overrides
    ff = []
    for p in ("bf16", "int4"):
        f = d / "final" / f"{p}_clean.jsonl"
        ensure_meta(f, _meta(w, p, split="final"))
        f.write_text(json.dumps({"question_id": "q", "scene_token": S[8], "precision": p, "condition": "clean"}) + "\n")
        ff.append(f)
    with pytest.raises(FreezeError, match="missing"):
        validate_analysis_files(ff, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=tmp_path / "none.yaml")
    ok = validate_analysis_files(ff, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path, settings={"margin_edges": [0.25, 0.5]})
    assert "frozen" in ok
    with pytest.raises(FreezeError, match="margin_edges"):
        validate_analysis_files(ff, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path, settings={"margin_edges": [0.1, 0.5]})
    # excluded scene inside a final file -> refused
    ff[0].write_text(json.dumps({"question_id": "q", "scene_token": S[5], "precision": "bf16", "condition": "clean"}) + "\n")
    with pytest.raises(ProvenanceError, match="outside the final population"):
        validate_analysis_files(ff, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=frozen_path)


def test_freeze_config_cli(world, tmp_path):
    """Write refuses on a blocking TO DECIDE row and on a dirty tree; --check passes then fails after a config change."""
    w = world
    repo = w["root"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    protocol = repo / "PROTOCOL.md"
    protocol.write_text("## 2. Serving\n\n| a | 1 | TO DECIDE |\n\n## Change log\n")
    base = [sys.executable, str(ROOT / "scripts/freeze_config.py"), "--config", str(w["cfg_path"]), "--split-file", str(w["split"]), "--exclusions-file", str(w["excl"]),
            "--protocol", str(protocol), "--frozen", str(repo / "config_frozen.yaml"), "--snapshot", str(repo / "PROTOCOL_frozen.md"), "--int4-model", str(w["ckpt"])]
    r = subprocess.run(base, cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "section 2: a" in r.stdout
    protocol.write_text("## 2. Serving\n\n| a | 1 | FROZEN |\n\n## 7. Scoring\n\n| VQA | j | TO DECIDE |\n\n## Change log\n")
    r = subprocess.run(base, cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "dirty" in r.stdout + r.stderr
    r = subprocess.run(base + ["--allow-dirty"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (repo / "PROTOCOL_frozen.md").read_text() == protocol.read_text()
    r = subprocess.run(base + ["--check"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0 and "freeze check passed" in r.stdout
    r = subprocess.run(base + ["--check", "--serve-manifest", str(w["man"] / "serve_int4.txt"), "--precision", "int4"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    cfg = yaml.safe_load(w["cfg_path"].read_text())
    cfg["generation"]["max_tokens"] = 256
    w["cfg_path"].write_text(yaml.safe_dump(cfg))
    r = subprocess.run(base + ["--check"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "generation.max_tokens" in r.stdout


# ---------------- review-round regressions ----------------

def _frozen(w, tmp_path):
    fp = tmp_path / "config_frozen.yaml"
    fp.write_text(yaml.safe_dump(build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], protocol_snapshot=w["man"] / "PROTOCOL_frozen.md")))
    return fp


def _final_files(w, d, metas):
    ff = []
    for p, m in metas.items():
        f = d / f"{p}_clean.jsonl"
        ensure_meta(f, m)
        f.write_text(json.dumps({"question_id": "q", "scene_token": S[8], "precision": p, "condition": "clean"}) + "\n")
        ff.append(f)
    return ff


def test_final_analysis_checks_each_sidecar_against_freeze(world, tmp_path):
    """Review item 1: files produced under other serving/generation settings are refused
    even when the live config matches the freeze."""
    w = world
    fp = _frozen(w, tmp_path)
    kw = dict(split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp)
    ff = _final_files(w, tmp_path / "a", {"bf16": _meta(w, "bf16", split="final"), "int4": _meta(w, "int4", split="final")})
    assert validate_analysis_files(ff, w["cfg"], **kw)["analysis_class"] == "confirmatory"
    # both sidecars recorded max_pixels 401408 (live config still matches the freeze)
    _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123", max_pixels=401408)
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]), max_pixels=401408)
    bad = _final_files(w, tmp_path / "b", {"bf16": _meta(w, "bf16", split="final"), "int4": _meta(w, "int4", split="final")})
    with pytest.raises(FreezeError, match="max_pixels: file='401408'"):
        validate_analysis_files(bad, w["cfg"], **kw)
    _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123")
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]))
    # sidecar recorded a core config with max_tokens 256
    cfg2 = yaml.safe_load(w["cfg_path"].read_text())
    cfg2["generation"]["max_tokens"] = 256
    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump(cfg2))
    mb = _meta(w, "bf16", split="final")
    mb["core_config_sha256"] = build_run_meta(split="final", precision="bf16", condition="clean", config_path=alt, annotations_path=w["cfg"]["data"]["annotations"], prompt_path=w["cfg"]["prompt"]["system_file"],
                                            split_file=w["split"], exclusions_file=w["excl"], serve_manifest_path=w["man"] / "serve_bf16.txt", served_model_name="qwen-bf16", repo_root=w["root"])["core_config_sha256"]
    mi = _meta(w, "int4", split="final")
    mi["core_config_sha256"] = mb["core_config_sha256"]
    bad = _final_files(w, tmp_path / "c", {"bf16": mb, "int4": mi})
    with pytest.raises(FreezeError, match="core_config_sha256"):
        validate_analysis_files(bad, w["cfg"], **kw)
    # seed / max_num_seqs / prefix caching recorded differently (review item 5)
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]), seed=7, max_num_seqs=4, prefix_caching="false")
    _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123", seed=7, max_num_seqs=4, prefix_caching="false")
    bad = _final_files(w, tmp_path / "d", {"bf16": _meta(w, "bf16", split="final"), "int4": _meta(w, "int4", split="final")})
    with pytest.raises(FreezeError) as e:
        validate_analysis_files(bad, w["cfg"], **kw)
    assert "seed: file='7'" in str(e.value) and "max_num_seqs: file='4'" in str(e.value) and "prefix_caching: file='false'" in str(e.value)
    diffs = check_serve_manifest(parse_serve_manifest(w["man"] / "serve_int4.txt"), "int4", yaml.safe_load(fp.read_text()))
    assert {d.split(":")[0] for d in diffs} == {"seed", "max_num_seqs", "prefix_caching"}
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]))
    _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123")
    # official checkpoint recorded in the INT4 sidecar
    _manifest(w["man"] / "serve_int4.txt", "int4", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", extra="--dtype float16 --quantization awq_marlin")
    bad = _final_files(w, tmp_path / "e", {"bf16": _meta(w, "bf16", split="final"), "int4": _meta(w, "int4", split="final")})
    with pytest.raises(FreezeError, match="model: server='Qwen/Qwen2.5-VL-7B-Instruct-AWQ'"):
        validate_analysis_files(bad, w["cfg"], **kw)
    _manifest(w["man"] / "serve_int4.txt", "int4", str(w["ckpt"]))


def test_no_provenance_refuses_final_sidecars(world, tmp_path):
    """Review item 2: --no-provenance never analyzes a file whose sidecar declares final."""
    from qflip.guard import legacy_files_check

    w = world
    legacy = tmp_path / "pilot" / "bf16_clean.jsonl"
    legacy.parent.mkdir()
    legacy.write_text("{}\n")
    out = legacy_files_check([legacy])
    assert out["analysis_class"] == "exploratory" and out["files"][str(legacy)]["sidecar"] is False
    ff = _final_files(w, tmp_path / "f", {"bf16": _meta(w, "bf16", split="final")})
    with pytest.raises(ProvenanceError, match="not allowed for final-split files"):
        legacy_files_check([legacy, ff[0]])


def test_freeze_requires_and_checks_the_protocol_snapshot(world, tmp_path):
    """Review item 8: the snapshot path is stored, its hash required, and edits detected."""
    w = world
    snap = w["man"] / "PROTOCOL_frozen.md"
    frozen = build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], protocol_snapshot=snap)
    assert frozen["frozen"]["protocol_snapshot"] == str(snap) and len(frozen["frozen"]["protocol_snapshot_sha256"]) == 64
    assert check_config(w["cfg"], frozen, w["split"], w["excl"]) == []
    snap.write_text(snap.read_text() + "\n| b | 2 | FROZEN |\n")
    assert any(d.startswith("protocol_snapshot_sha256") for d in check_config(w["cfg"], frozen, w["split"], w["excl"]))
    with pytest.raises(FreezeError, match="snapshot .* missing"):
        build_frozen(w["cfg"], w["split"], w["excl"], str(w["ckpt"]), repo_root=w["root"], protocol_snapshot=tmp_path / "nope.md")
    frozen["frozen"].pop("protocol_snapshot")
    frozen["frozen"]["protocol_snapshot_sha256"] = None
    assert any(d.startswith("protocol_snapshot:") for d in check_config(w["cfg"], frozen, w["split"], w["excl"]))


def test_freeze_config_cli_custom_snapshot_is_fingerprinted(world, tmp_path):
    w = world
    repo = w["root"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    protocol = repo / "PROTOCOL.md"
    protocol.write_text("## 2. Serving\n\n| a | 1 | FROZEN |\n\n## Change log\n")
    snap = repo / "custom_snapshot.md"
    base = [sys.executable, str(ROOT / "scripts/freeze_config.py"), "--config", str(w["cfg_path"]), "--split-file", str(w["split"]), "--exclusions-file", str(w["excl"]),
            "--protocol", str(protocol), "--frozen", str(repo / "config_frozen.yaml"), "--snapshot", str(snap), "--int4-model", str(w["ckpt"]), "--allow-dirty"]
    r = subprocess.run(base, cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    frozen = yaml.safe_load((repo / "config_frozen.yaml").read_text())
    assert frozen["frozen"]["protocol_snapshot"] == str(snap) and frozen["frozen"]["protocol_snapshot_sha256"]
    r = subprocess.run(base + ["--check"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    snap.write_text(snap.read_text() + "edited\n")
    r = subprocess.run(base + ["--check"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "protocol_snapshot_sha256" in r.stdout


# ---------------- live server binding (review items 4, follow-up 1) ----------------

import http.server  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

_SERVER_CODE = """
import http.server, json, sys
root, port = sys.argv[2], int(sys.argv[4])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "qwen-int4", "object": "model", "root": root}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port():
    try:
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            return sk.getsockname()[1]
    except PermissionError:
        pytest.skip("cannot open local sockets in this environment")


def _serving_process(model: str, port: int):
    """A real listener on the port whose argv looks like `vllm serve MODEL --port N`.
    Skips only when the environment forbids local sockets; any other exit or a
    startup timeout fails with the server's stderr."""
    proc = subprocess.Popen([sys.executable, "-c", _SERVER_CODE, "serve", model, "--port", str(port)], stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).read()
            return proc
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                err = proc.stderr.read()
                if "PermissionError" in err:
                    pytest.skip("cannot open local sockets in this environment")
                pytest.fail(f"fake server exited with code {proc.returncode}:\n{err}")
            time.sleep(0.05)
    proc.kill()
    pytest.fail(f"fake server did not answer on port {port} within 5 s:\n{proc.stderr.read()}")


def _sleeping_process(model: str, port: int):
    """Matching argv, no socket: must not pass as the server."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "serve", model, "--port", str(port)])


class _FakeVllm:
    """In-process /v1/models endpoint (listener owned by the test process)."""

    def __init__(self, served: str, root: str | None):
        cards = [{"id": served, "object": "model", **({"root": root} if root else {})}]

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self_inner):
                body = json.dumps({"object": "list", "data": cards}).encode()
                self_inner.send_response(200)
                self_inner.send_header("Content-Type", "application/json")
                self_inner.end_headers()
                self_inner.wfile.write(body)

            def log_message(self_inner, *a):
                pass

        try:
            self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        except PermissionError:
            pytest.skip("cannot open local sockets in this environment")
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self):
        self.srv.shutdown()


def test_url_port_is_parsed_exactly():
    from qflip.provenance import url_port

    assert url_port("http://localhost:8100/v1") == 8100 and url_port("http://localhost:81/v1") == 81
    assert url_port("http://localhost/v1") == 80 and url_port("https://h/v1") == 443


def test_check_server_uses_model_root_not_only_alias(world):
    from qflip.provenance import check_server

    srv = _FakeVllm("qwen-int4", root="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    try:
        check_server(srv.base_url, "qwen-int4")  # alias alone passes
        with pytest.raises(ProvenanceError, match="loaded 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'"):
            check_server(srv.base_url, "qwen-int4", expect_model=str(world["ckpt"]))
        with pytest.raises(ProvenanceError, match="wrong server"):
            check_server(srv.base_url, "qwen-bf16")
    finally:
        srv.close()
    srv = _FakeVllm("qwen-int4", root=str(world["ckpt"]))
    try:
        assert check_server(srv.base_url, "qwen-int4", expect_model=str(world["ckpt"]))
    finally:
        srv.close()


def test_listener_pid_finds_the_owning_process():
    from qflip.provenance import listener_pid

    srv = _FakeVllm("x", None)
    try:
        assert listener_pid(srv.port, "http://127.0.0.1") == os.getpid()
        assert listener_pid(srv.port, "http://localhost") == os.getpid()
    finally:
        srv.close()
    assert listener_pid(_free_port(), "http://localhost") is None


def _fake_proc(root: Path, tcp: list[tuple[str, int, str]], tcp6: list[tuple[str, int, str]], fds: dict[int, list[str]], ppid: dict[int, int] | None = None):
    """A /proc fixture: LISTEN rows (text address, port, inode) and pid -> socket inodes."""
    import ipaddress

    def hexword(b):
        return "".join(f"{struct.unpack('<I', b[i:i + 4])[0]:08X}" for i in range(0, 16, 4))

    (root / "net").mkdir(parents=True)
    head = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    rows4 = [f"   0: {int.from_bytes(ipaddress.IPv4Address(a).packed, 'little'):08X}:{p:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 {i} 1 0 0 0 0\n" for a, p, i in tcp]
    rows6 = [f"   0: {hexword(ipaddress.IPv6Address(a).packed)}:{p:04X} {'0' * 32}:0000 0A 00000000:00000000 00:00000000 00000000     0        0 {i} 1 0 0 0 0\n" for a, p, i in tcp6]
    (root / "net/tcp").write_text(head + "".join(rows4))
    (root / "net/tcp6").write_text(head + "".join(rows6))
    for pid, inodes in fds.items():
        d = root / str(pid) / "fd"
        d.mkdir(parents=True)
        for n, i in enumerate(inodes):
            os.symlink(f"socket:[{i}]", d / str(n))
        (root / str(pid) / "status").write_text(f"Name:\tx\nPid:\t{pid}\nPPid:\t{(ppid or {}).get(pid, 1)}\n")
        (root / str(pid) / "cmdline").write_bytes(b"\0".join([b"python", b"serve", b"MODEL", b"--port", b"8000"]) + b"\0")


def test_listener_ownership_matches_endpoint_address(tmp_path):
    """Follow-up: sockets are matched on address and port, wildcard bindings count for
    every local address, non-loopback endpoints are refused, and several owners are ambiguous."""
    from qflip.provenance import check_server_process, listener_pid, url_local_addresses

    proc = tmp_path / "proc"
    _fake_proc(proc, tcp=[("127.0.0.1", 8000, "11"), ("127.0.0.2", 8000, "22")], tcp6=[("::1", 8001, "33")], fds={100: ["11"], 200: ["22"], 300: ["33"]})
    assert listener_pid(8000, "http://127.0.0.1:8000/v1", proc) == 100
    assert listener_pid(8000, "http://127.0.0.2:8000/v1", proc) == 200
    assert listener_pid(8000, "http://localhost:8000/v1", proc) == 100  # localhost = 127.0.0.1 or ::1; 127.0.0.2 is not it
    assert listener_pid(8001, "http://localhost:8001/v1", proc) == 300  # IPv6 loopback
    assert listener_pid(8001, "http://127.0.0.1:8001/v1", proc) is None
    with pytest.raises(ProvenanceError, match="not a loopback address"):
        listener_pid(8000, "http://remote.example:8000/v1", proc)
    with pytest.raises(ProvenanceError, match="not a loopback address"):
        url_local_addresses("http://10.0.0.5:8000/v1")
    # the full guard: manifest pid 100 serves 127.0.0.1 only
    man = {"pid": "100", "port": "8000", "model": "MODEL"}
    assert check_server_process(man, "http://127.0.0.1:8000/v1", proc) == 100
    with pytest.raises(ProvenanceError, match="served by pid 200, which is not the manifest server 100"):
        check_server_process(man, "http://127.0.0.2:8000/v1", proc)
    with pytest.raises(ProvenanceError, match="not a loopback address"):
        check_server_process(man, "http://remote.example:8000/v1", proc)
    # wildcard binding serves every local address; a child of the manifest pid may hold it
    proc2 = tmp_path / "proc2"
    _fake_proc(proc2, tcp=[("0.0.0.0", 8000, "44")], tcp6=[("::", 8001, "55")], fds={100: [], 101: ["44"], 102: ["55"]}, ppid={101: 100, 102: 100})
    assert listener_pid(8000, "http://127.0.0.1:8000/v1", proc2) == 101
    assert listener_pid(8001, "http://localhost:8001/v1", proc2) == 102
    assert check_server_process(man, "http://localhost:8000/v1", proc2) == 100
    # two owners for the same endpoint -> ambiguous, refused
    proc3 = tmp_path / "proc3"
    _fake_proc(proc3, tcp=[("127.0.0.1", 8000, "66"), ("0.0.0.0", 8000, "77")], tcp6=[], fds={100: ["66"], 200: ["77"]})
    with pytest.raises(ProvenanceError, match="ambiguous"):
        check_server_process(man, "http://127.0.0.1:8000/v1", proc3)
    # a matching socket whose owner is not visible -> unresolved, refused even though another owner is
    proc4 = tmp_path / "proc4"
    _fake_proc(proc4, tcp=[("127.0.0.1", 8000, "88")], tcp6=[("::1", 8000, "99")], fds={100: ["88"]})
    with pytest.raises(ProvenanceError, match="owning process is not visible .*::1.*unresolved"):
        check_server_process(man, "http://localhost:8000/v1", proc4)
    assert listener_pid(8000, "http://127.0.0.1:8000/v1", proc4) == 100  # the IPv4 endpoint alone is fully owned
    # an IPv4 wildcard does not serve an IPv6 endpoint; the real ::1 listener's owner is invisible -> refused
    proc5 = tmp_path / "proc5"
    _fake_proc(proc5, tcp=[("0.0.0.0", 8001, "12")], tcp6=[("::1", 8001, "13")], fds={100: ["12"]})
    with pytest.raises(ProvenanceError, match="not visible"):
        listener_pid(8001, "http://[::1]:8001/v1", proc5)
    assert listener_pid(8001, "http://127.0.0.1:8001/v1", proc5) == 100


def test_wildcard_matching_respects_address_family():
    from qflip.provenance import _addr_matches

    assert _addr_matches("0.0.0.0", ("127.0.0.1",)) and not _addr_matches("0.0.0.0", ("::1",))
    assert _addr_matches("0.0.0.0", ("127.0.0.1", "::1"))
    assert _addr_matches("::", ("::1",)) and _addr_matches("::", ("127.0.0.1",))  # dual-stack wildcard
    assert _addr_matches("::ffff:127.0.0.1", ("127.0.0.1",)) and not _addr_matches("::1", ("127.0.0.1",))


def test_prepare_run_requires_listener_ownership(world, tmp_path, monkeypatch):
    """Follow-up item 1: the manifest pid must own the LISTEN socket the client
    talks to; a process with matching arguments but no socket is refused, as are
    a missing port line, a mismatched port, a stale pid, and a foreign model."""
    from qflip.guard import prepare_run

    w = world
    model = str(w["ckpt"])
    port = _free_port()
    server = _serving_process(model, port)
    sleeper = _sleeping_process(model, port)
    base = f"http://127.0.0.1:{port}/v1"
    kw = dict(config_path=w["cfg_path"], precision="int4", condition="clean", split="development", split_file=w["split"], exclusions_file=w["excl"], manifests_dir=w["man"])
    try:
        monkeypatch.setenv("QFLIP_BASE_URL", base)
        # no pid/port lines (old 01_serve.sh) -> refused
        with pytest.raises(ProvenanceError, match="no pid/port"):
            prepare_run(w["cfg"], result_file=tmp_path / "r0/int4_clean.jsonl", **kw)
        # the real listener -> ok
        _manifest(w["man"] / "serve_int4.txt", "int4", model, port=port, pid=server.pid)
        assert prepare_run(w["cfg"], result_file=tmp_path / "r1/int4_clean.jsonl", **kw)["server_pid"] == str(server.pid)
        # matching argv, no socket -> refused
        _manifest(w["man"] / "serve_int4.txt", "int4", model, port=port, pid=sleeper.pid)
        with pytest.raises(ProvenanceError, match=f"served by pid {server.pid}, which is not the manifest server {sleeper.pid}"):
            prepare_run(w["cfg"], result_file=tmp_path / "r2/int4_clean.jsonl", **kw)
        # the guard follows QFLIP_BASE_URL like the client
        _manifest(w["man"] / "serve_int4.txt", "int4", model, port=port, pid=server.pid)
        monkeypatch.setenv("QFLIP_BASE_URL", "http://127.0.0.1:9/v1")
        with pytest.raises(ProvenanceError, match="cannot query http://127.0.0.1:9/v1/models"):
            prepare_run(w["cfg"], result_file=tmp_path / "r3/int4_clean.jsonl", **kw)
        monkeypatch.setenv("QFLIP_BASE_URL", base)
        # manifest port differs from the client's (and a prefix of it is not a match)
        _manifest(w["man"] / "serve_int4.txt", "int4", model, port=int(str(port)[:2]), pid=server.pid)
        with pytest.raises(ProvenanceError, match="the manifest server is on port"):
            prepare_run(w["cfg"], result_file=tmp_path / "r4/int4_clean.jsonl", **kw)
        # manifest names a model the server did not load -> refused by the model card
        _manifest(w["man"] / "serve_int4.txt", "int4", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", extra="--dtype float16 --quantization awq_marlin", port=port, pid=server.pid)
        with pytest.raises(ProvenanceError, match="does not describe the running server"):
            prepare_run(w["cfg"], result_file=tmp_path / "r5/int4_clean.jsonl", **kw)
        # stale pid -> refused
        sleeper.terminate()
        sleeper.wait()
        _manifest(w["man"] / "serve_int4.txt", "int4", model, port=port, pid=sleeper.pid)
        with pytest.raises(ProvenanceError, match="not running on this host"):
            prepare_run(w["cfg"], result_file=tmp_path / "r6/int4_clean.jsonl", **kw)
    finally:
        for pr in (server, sleeper):
            if pr.poll() is None:
                pr.kill()
                pr.wait()
    os.environ.pop("QFLIP_BASE_URL", None)


# ---------------- follow-up items 2 and 3 ----------------

def test_reference_dtype_is_frozen_and_enforced(world, tmp_path):
    """Follow-up item 2: a BF16-labelled server running --dtype float16 is refused by the
    serve-manifest check and by final analysis."""
    w = world
    fp = _frozen(w, tmp_path)
    frozen = yaml.safe_load(fp.read_text())
    assert frozen["models_frozen"]["bf16"]["dtype"] == "bfloat16" and frozen["models_frozen"]["int4"]["dtype"] == "float16"
    half = _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123", extra="--dtype float16 --revision abc123")
    assert any(d.startswith("dtype: server='float16'") for d in check_serve_manifest(parse_serve_manifest(half), "bf16", frozen))
    with pytest.raises(FreezeError, match="dtype: server='float16'"):
        final_guard(w["cfg"], "bf16", half, w["split"], w["excl"], fp)
    ff = _final_files(w, tmp_path / "g", {"bf16": _meta(w, "bf16", split="final"), "int4": _meta(w, "int4", split="final")})
    with pytest.raises(FreezeError, match="dtype: server='float16'"):
        validate_analysis_files(ff, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=fp)
    _manifest(w["man"] / "serve_bf16.txt", "bf16", "Qwen/base", "abc123")
    # an INT4 server in bfloat16 is refused the same way
    frozen["models_frozen"]["int4"].pop("dtype")
    assert any("records no dtype" in d for d in check_serve_manifest(parse_serve_manifest(w["man"] / "serve_int4.txt"), "int4", frozen))


def test_raw_rows_validated_before_dedup_on_resume_and_analysis(world, tmp_path):
    """Follow-up item 3: an INT4/Fog row hidden behind a later same-id BF16/clean row is
    refused on resume; analysis refuses rows whose condition differs from the sidecar."""
    w = world
    rf = tmp_path / "r" / "bf16_clean.jsonl"
    ensure_meta(rf, _meta(w))
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "int4", "condition": "Fog"}) + "\n"
                  + json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "clean"}) + "\n")
    with pytest.raises(ProvenanceError, match="row 1 \\('q1'\\) has precision='int4'"):
        ensure_meta(rf, _meta(w), allowed_scenes=set(S[2:5]))
    # a consistent retry (same precision/condition, same id twice) still resumes
    rf.write_text(json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "clean", "error": "timeout"}) + "\n"
                  + json.dumps({"question_id": "q1", "scene_token": S[2], "precision": "bf16", "condition": "clean"}) + "\n")
    ensure_meta(rf, _meta(w), allowed_scenes=set(S[2:5]))
    # analysis: Fog rows under a clean sidecar
    d = tmp_path / "a"
    files = []
    for p in ("bf16", "int4"):
        f = d / f"{p}_clean.jsonl"
        ensure_meta(f, _meta(w, p))
        f.write_text(json.dumps({"question_id": "q", "scene_token": S[2], "precision": p, "condition": "Fog"}) + "\n")
        files.append(f)
    with pytest.raises(ProvenanceError, match="has condition='Fog', sidecar says 'clean'"):
        validate_analysis_files(files, w["cfg"], split_file=w["split"], exclusions_file=w["excl"], frozen_path=tmp_path / "none.yaml")
