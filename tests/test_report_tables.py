"""Step 3: coverage and harmful-example producers, profile pooling, and the
seven report tables (formatting only)."""

import copy
from pathlib import Path

import pytest

from qflip.flips import Pair, pair_records
from qflip.provenance import COMPAT_FIELDS, PRECISION_FIELDS
from qflip.report import ProfileError, condition_rows, coverage_by_cell, coverage_report, harmful_examples, pool_profiles, profile_summary
from qflip.tables import render_all, t1_coverage, t2_headline, t3_contrasts, t4_breakdowns, t5_stability, t6_deployment, t7_harmful


def _rec(qid, scene, cell="behavior/MCQ", letter="A", correct=True, precision="bf16", error=None, margin=0.6):
    return {"question_id": qid, "scene_token": scene, "frame_token": "f", "cell": cell, "n_views": 6, "views": ["CAM_FRONT"] * 6, "condition": "clean",
            "precision": precision, "question": f"Q{qid}", "gt": "A", "pred": letter, "error": error, "mcq_pred_strict": letter, "mcq_correct": correct, "margin": margin}


def test_coverage_and_harmful_examples_from_records():
    ref = [_rec("1", "s1"), _rec("2", "s1"), _rec("3", "s2", error="timeout"), _rec("4", "s2"), _rec("5", "s3", cell="perception/VQA")]
    qnt = [_rec("1", "s1", letter="B", correct=False, precision="int4"), _rec("2", "s1", precision="int4"), _rec("3", "s2", precision="int4"), _rec("6", "s3", precision="int4")]
    pairs = pair_records(ref, qnt)
    assert [p.question_id for p in pairs] == ["1", "2"] and pairs[0].ref_answer == "A" and pairs[0].quant_answer == "B"
    cov = coverage_by_cell(ref, qnt, pairs, {"behavior/MCQ": 40, "perception/VQA": 10}, min_n=2, min_scenes=1)
    b = cov["behavior/MCQ"]
    assert b == {"eligible": 40, "present_ref": 3, "present_quant": 4, "errored_ref": 1, "errored_quant": 0, "paired": 2, "unpaired": 3, "scenes": 1, "assessed": True}
    v = cov["perception/VQA"]
    assert v["eligible"] == 10 and v["present_ref"] == 1 and v["paired"] == 0 and v["unpaired"] == 1 and not v["assessed"]
    # unknown population: eligible is None, not 0
    assert coverage_by_cell(ref, qnt, pairs, None, 2, 1)["behavior/MCQ"]["eligible"] is None
    ex = harmful_examples(pairs)
    assert len(ex) == 1 and ex[0]["question_id"] == "1" and ex[0]["ref_answer"] == "A" and ex[0]["quant_answer"] == "B" and ex[0]["margin"] == 0.6 and ex[0]["n_views"] == 6
    vq = Pair("9", "s9", "planning/VQA", 6, "Fog", None, True, False, "harmful", ref_score=80, quant_score=20)
    e = harmful_examples([vq])[0]
    assert e["ref_answer"] is None and e["ref_score"] == 80 and e["condition"] == "Fog"


def test_coverage_report_keys_conditions_by_file_identity_not_question_id():
    """Step-3 review item 1: question ids repeat across conditions; a condition
    whose INT4 answers all errored reports 0 present and 0 paired."""
    clean_ref, clean_qnt = [_rec("1", "s1"), _rec("2", "s2")], [_rec("1", "s1", precision="int4"), _rec("2", "s2", precision="int4")]
    fog_ref = [{**r, "condition": "Fog"} for r in clean_ref]
    fog_qnt = [{**r, "condition": "Fog", "error": "timeout"} for r in clean_qnt]
    pairs = pair_records(clean_ref, clean_qnt) + pair_records(fog_ref, fog_qnt)
    assert len(pairs) == 2 and {p.condition for p in pairs} == {"clean"}
    cov = coverage_report({"clean": (clean_ref, clean_qnt), "Fog": (fog_ref, fog_qnt)}, pairs, None, 2, 2)
    assert set(cov) == {"clean", "Fog"}
    assert cov["clean"]["behavior/MCQ"]["paired"] == 2 and cov["clean"]["behavior/MCQ"]["assessed"]
    f = cov["Fog"]["behavior/MCQ"]
    assert f == {"eligible": None, "present_ref": 2, "present_quant": 0, "errored_ref": 0, "errored_quant": 2, "paired": 0, "unpaired": 2, "scenes": 0, "assessed": False}
    # follow-up item 3: the headline keeps a condition without pairs as an n = 0 unassessed row
    rows = condition_rows([{"label": "clean", "n": 2, "scenes": 2, "assessed": False, "reason": "coverage: 2 questions from 2 scenes, below 30 / 10"}], ["clean", "Fog"], 30, 10)
    assert [r["label"] for r in rows] == ["Fog", "clean"] and rows[0] == {"label": "Fog", "n": 0, "scenes": 0, "assessed": False, "reason": "coverage: 0 questions from 0 scenes, below 30 / 10"}
    assert "| Fog | 0 | 0 | UNASSESSED (coverage: 0 questions from 0 scenes, below 30 / 10) |" in t2_headline({"mcq": {"groups": {"condition": rows}}})


# ---------------------------------------------------------------- pooling


def _session(prec, k, ttft, peak, host="h1", fingerprint="sha256:own", file=None, provenance=True, kv="--kv-cache-memory-bytes 2147483648", n=None, views=6, warmup=3, study=True, errors=0):
    reqs = [{"question_id": f"q{i}", "error": None, "ttft_s": t, "latency_s": 4 * t, "decode_tok_s": 30.0, "prompt_tokens": 1000, "completion_tokens": 60} for i, t in enumerate(ttft)]
    for r in reqs[:errors]:
        r.update({"error": "boom", "ttft_s": None, "decode_tok_s": None})
    s = {"file": file or f"{prec}_clean_s{k}.json", "precision": prec, "condition": "clean", "session": k, "gpu_peak_mib": peak, "split": "development",
         "n": n if n is not None else len(reqs), "views": views, "warmup": warmup, "study": study,
         "summary": profile_summary(reqs), "requests": reqs, "host": {"hostname": host, "gpu_name": "RTX 4090"}}
    if provenance:
        prov = {k_: f"v-{k_}" for k_ in COMPAT_FIELDS}
        prov.update({"split": "development", "precision": prec, "condition": "clean", "served_model_name": f"qwen-{prec}", "model_path_or_id": f"/ckpt/{prec}",
                     "checkpoint_fingerprint": fingerprint, "quantization": "none" if prec == "bf16" else "compressed-tensors", "dtype": "bfloat16" if prec == "bf16" else "float16",
                     "max_pixels": "200704", "kv_args": kv, "vllm_version": "0.33.0", "torch_version": "2.14.0", "transformers_version": "5.20.2", "compressed_tensors_version": "0.17.0"})
        s["provenance"] = prov
    return s


def test_pool_profiles_concatenates_requests_and_keeps_sessions():
    ss = [_session("bf16", 1, [0.5, 0.6, 0.7], 20000), _session("bf16", 2, [1.0, 1.1, 1.2], 20500), _session("bf16", 3, [0.9, 0.8, 0.4], 20100),
          _session("int4", 1, [0.3, 0.3, 0.3], 12000), _session("int4", 2, [0.2, 0.4, 0.3], 12500)]
    out = pool_profiles(ss)
    b = out["precisions"]["bf16"]
    assert b["n_sessions"] == 3 and [s["session"] for s in b["sessions"]] == [1, 2, 3]
    assert b["pooled"]["requests"] == 9 and b["pooled"]["ttft_s"]["p50"] == 0.8  # nearest rank over the 9 concatenated requests
    assert b["pooled"]["gpu_peak_mib_max"] == 20500 and b["pooled"]["gpu_peak_mib_per_session"] == [20000, 20500, 20100]
    assert b["provenance"]["checkpoint_fingerprint"] == "sha256:own" and b["host"]["hostname"] == "h1"
    assert abs(out["memory_reduction"] - (1 - 12500 / 20500)) < 1e-9
    assert out["budgets"]["memory_reduction_ge_25pct"] is True and out["budgets"]["ttft_p95_le_2s"] == {"bf16": True, "int4": True}
    assert out["budgets"]["decode_p50_ge_15"]["int4"] is True
    assert out["reference"] == []


def test_pool_profiles_refuses_mixed_setups_duplicates_and_unprovenanced_files():
    base = [_session("bf16", 1, [0.5], 20000), _session("bf16", 2, [0.6], 20000)]
    with pytest.raises(ProfileError, match="session numbers must be present and distinct"):
        pool_profiles([base[0], _session("bf16", 1, [0.7], 20000, file="dup.json")])
    with pytest.raises(ProfileError, match="checkpoint_fingerprint"):
        pool_profiles([base[0], _session("bf16", 2, [0.6], 20000, fingerprint="hf:official")])
    with pytest.raises(ProfileError, match="host.hostname"):
        pool_profiles([base[0], _session("bf16", 2, [0.6], 20000, host="h2")])
    other = copy.deepcopy(base[1])
    other["provenance"]["max_pixels"] = "401408"
    with pytest.raises(ProfileError, match="max_pixels"):
        pool_profiles([base[0], other])
    legacy = _session("bf16", 1, [0.5], 21000, provenance=False, file="nov3_bf16_s1.json")
    with pytest.raises(ProfileError, match="no provenance block"):
        pool_profiles([legacy])
    # step-3 review item 4: placeholder provenance, other workload, exploratory sessions, incomparable precisions
    placeholder = copy.deepcopy(base[0])
    placeholder["provenance"] = {"precision": "bf16", "checkpoint_fingerprint": "sha256:own"}
    with pytest.raises(ProfileError, match="provenance incomplete"):
        pool_profiles([placeholder])
    with pytest.raises(ProfileError, match="workload.views"):
        pool_profiles([base[0], _session("bf16", 2, [0.6], 20000, views=0)])
    with pytest.raises(ProfileError, match="workload.warmup"):
        pool_profiles([base[0], _session("bf16", 2, [0.6], 20000, warmup=0)])
    with pytest.raises(ProfileError, match="workload.question_ids"):
        pool_profiles([base[0], _session("bf16", 2, [0.6, 0.7], 20000)])
    with pytest.raises(ProfileError, match="exploratory session"):
        pool_profiles([base[0], _session("bf16", 2, [0.6], 20000, study=False)])
    # follow-up item 2: a session without the study flag (no evidence of the freeze check) is refused too
    unmarked = copy.deepcopy(base[1])
    del unmarked["study"]
    with pytest.raises(ProfileError, match="not marked as a study session"):
        pool_profiles([base[0], unmarked])
    # follow-up item 1: a local checkpoint has no Hub revision; a Hub model must have one
    local = copy.deepcopy(base[0])
    local["provenance"].update({"checkpoint_fingerprint": "sha256:abc", "model_revision": None})
    assert pool_profiles([local])["precisions"]["bf16"]["n_sessions"] == 1
    hub = copy.deepcopy(base[0])
    hub["provenance"].update({"checkpoint_fingerprint": "hf:Qwen/x@unknown", "model_revision": None})
    with pytest.raises(ProfileError, match="missing \\['model_revision'\\]"):
        pool_profiles([hub])
    with pytest.raises(ProfileError, match="not comparable.*kv_args"):
        pool_profiles(base + [_session("int4", 1, [0.3], 10000, kv="--kv-cache-memory-bytes 8589934592")])
    with pytest.raises(ProfileError, match="not comparable.*host.hostname"):
        pool_profiles(base + [_session("int4", 1, [0.3], 10000, host="h2")])
    with pytest.raises(ProfileError, match="not comparable.*workload.question_ids"):
        pool_profiles(base + [_session("int4", 1, [0.3, 0.3], 10000)])
    # runtime versions are compatibility fields: each one, drifted alone, blocks pooling within
    # a precision and the comparison across precisions
    for field in ("vllm_version", "torch_version", "transformers_version", "compressed_tensors_version"):
        drifted = copy.deepcopy(base[1])
        drifted["provenance"][field] = "99.0.0"
        with pytest.raises(ProfileError, match=field):
            pool_profiles([base[0], drifted])
        other = _session("int4", 1, [0.3], 10000)
        other["provenance"][field] = "99.0.0"
        with pytest.raises(ProfileError, match=f"not comparable.*{field}"):
            pool_profiles(base + [other])
    # step-3 review item 5: an all-error session keeps its counts, verdicts become n/a, nothing raises
    out = pool_profiles([_session("bf16", 1, [0.5, 0.6], 20000, errors=2), _session("int4", 1, [0.3, 0.3], 10000)])
    b = out["precisions"]["bf16"]["pooled"]
    assert b["requests"] == 2 and b["errors"] == 2 and b["ttft_s"]["p95"] is None
    assert out["budgets"]["ttft_p95_le_2s"] == {"bf16": None, "int4": True} and out["budgets"]["decode_p50_ge_15"]["bf16"] is None
    assert out["memory_reduction"] == 0.5 and out["budgets"]["memory_reduction_ge_25pct"] is True
    md = t6_deployment(out)
    assert "| **bf16** (pooled) | 1 | 2 | 2 | n/a / n/a / n/a | n/a / n/a / n/a | n/a | 20000 |" in md and "bf16: n/a, int4: pass" in md
    # listed unpooled as reference; the pooled numbers ignore it
    out = pool_profiles(base, reference=[legacy])
    assert out["precisions"]["bf16"]["pooled"]["gpu_peak_mib_max"] == 20000
    assert len(out["reference"]) == 1 and out["reference"][0]["file"] == "nov3_bf16_s1.json" and out["reference"][0]["gpu_peak_mib"] == 21000
    assert out["memory_reduction"] is None and out["budgets"]["memory_reduction_ge_25pct"] is None


# ---------------------------------------------------------------- tables


def _group(label, n, scenes, assessed=True, harmful=2, fh=0.1):
    g = {"label": label, "n": n, "scenes": scenes, "ref_acc": 0.8, "quant_acc": 0.78, "acc_delta": -0.02, "harmful": harmful, "beneficial": 1, "neutral_change": 1,
         "unchanged": n - harmful - 2, "harmful_rate": fh, "change_rate": 0.1, "ref_score_mean": None, "quant_score_mean": None, "assessed": assessed}
    if assessed:
        g["ci"] = {"harmful_rate": [0.02, 0.2], "acc_delta": [-0.05, 0.01], "change_rate": [0.05, 0.15]}
    else:
        g["reason"] = f"coverage: {n} questions from {scenes} scenes, below 30 / 10"
    return g


def _side(n, rc, sc, fh):
    return {"n": n, "scenes": sc, "ref_correct": rc, "ref_correct_scenes": sc, "harmful": int(rc * fh), "harmful_rate": fh}


def _contrast(name, a, b, assessed=True, reason=None, detail=None, population="clean", **extra):
    r = {"name": name, "a_label": a, "b_label": b, "population": population, "a": _side(60, 50, 20, 0.2), "b": _side(70, 60, 22, 0.05), "delta": 0.15,
         "ci95": [0.02, 0.28], "ci99_bonferroni": [-0.01, 0.31], "excludes_zero_raw": True, "excludes_zero_adjusted": False, "n_boot_contrast": 200, "valid_draws": 200,
         "intervals_available": True, "assessed": assessed, "reason": reason, "reason_detail": detail}
    if not assessed:
        r.update({"ci95": None, "ci99_bonferroni": None, "excludes_zero_raw": False, "excludes_zero_adjusted": False, "valid_draws": 0, "intervals_available": False, "a": _side(8, 5, 4, 0.2)})
    r.update(extra)
    return r


def _compare():
    return {
        "analysis_class": "confirmatory", "min_questions": 30, "min_scenes": 10, "n_boot_group": 2000, "seed": 0,
        "provenance": {"split": "final", "frozen": {"date": "2026-11-10"}},
        "coverage": {"clean": {"behavior/MCQ": {"eligible": 133, "present_ref": 130, "present_quant": 131, "errored_ref": 3, "errored_quant": 2, "paired": 129, "unpaired": 3, "scenes": 110, "assessed": True},
                               "planning/VQA": {"eligible": 399, "present_ref": 399, "present_quant": 399, "errored_ref": 0, "errored_quant": 0, "paired": 0, "unpaired": 399, "scenes": 0, "assessed": False}},
                     "Fog": {"behavior/MCQ": {"eligible": 133, "present_ref": 20, "present_quant": 20, "errored_ref": 0, "errored_quant": 0, "paired": 20, "unpaired": 0, "scenes": 8, "assessed": False}}},
        "mcq": {"overall": _group("all", 300, 116), "groups": {"condition": [_group("clean", 260, 116), _group("Fog", 20, 8, assessed=False)],
                                                              "cell": [_group("behavior/MCQ", 129, 110), _group("perception/MCQ", 131, 112)],
                                                              "n_views": [_group("1", 25, 9, assessed=False), _group("6", 275, 116)],
                                                              "margin": [_group("[0,0.25)", 40, 30, fh=0.3), _group("[0.25,0.5)", 60, 45), _group("[0.5,1]", 200, 116, fh=0.02)]}},
        "contrasts": [_contrast("C1", "margin [0,0.25)", "other margin bins", population="clean, pairs with a margin", excludes_zero_adjusted=True),
                      _contrast("C2", "behavior/MCQ", "perception/MCQ"),
                      _contrast("C4a", "Dark", "clean", assessed=False, reason="coverage", detail="A: 8 questions from 4 scenes", population="common support", support_dropped={"clean/bf16": 0, "clean/int4": 1, "Dark/bf16": 2, "Dark/int4": 0}),
                      _contrast("C4b", "Fog", "clean"), _contrast("C4c", "NoImage", "clean")],
        "descriptive": [_contrast("views", "1 view", "6 views", confirmatory=False, ci99_bonferroni=None, excludes_zero_adjusted=None, note="Δ_views = −Δ_task by construction; no decision.")],
        "contrast_settings": {"n_boot_contrast": 200, "seed": 0, "support_min_ref_correct": 20, "support_min_ref_scenes": 10, "family_size": 5, "alpha_adjusted": 0.01},
        "harmful_examples": [{"question_id": "q1", "scene_token": "s", "condition": "clean", "cell": "behavior/MCQ", "n_views": 6, "margin": 0.12, "ref_answer": "A", "quant_answer": "C", "ref_score": None, "quant_score": None},
                             {"question_id": "q2", "scene_token": "s", "condition": "Fog", "cell": "planning/VQA", "n_views": 6, "margin": None, "ref_answer": None, "quant_answer": None, "ref_score": 85, "quant_score": 40}],
    }


def _stability():
    return [{"precision": "bf16", "condition": "clean", "model": "m", "repeats": 3, "mcq_common": 25, "errors": 0, "self_change_rate": 0.0, "self_correctness_change_rate": 0.0, "self_text_change_rate": 0.08, "margin_range_max": 0.039},
            {"precision": "int4", "condition": "clean", "model": "m4", "repeats": 3, "mcq_common": 25, "errors": 0, "self_change_rate": 0.0, "self_correctness_change_rate": 0.0, "self_text_change_rate": 0.01, "margin_range_max": 0.004,
             "vqa": {"n": 35, "self_correctness_change_rate": 0.05}}]


def test_all_seven_tables_from_fixtures():
    profile = pool_profiles([_session("bf16", 1, [0.5, 0.6], 20000), _session("bf16", 2, [0.7, 0.8], 20500), _session("int4", 1, [0.3, 0.2], 12000)],
                            reference=[_session("int4", 1, [0.4], 13000, provenance=False, file="ref.json")])
    md = render_all(_compare(), profile, _stability())
    for t in ("### T1.", "### T2.", "### T3.", "### T4.", "### T5.", "### T6.", "### T7."):
        assert t in md
    assert md.startswith("Analysis class: **confirmatory**; core freeze 2026-11-10.")
    # T1 keeps n/a for unknown eligibility and the assessed verdict
    t1 = t1_coverage({"coverage": {"clean": {"behavior/MCQ": {"eligible": None, "present_ref": 8, "present_quant": 8, "errored_ref": 0, "errored_quant": 0, "paired": 8, "unpaired": 0, "scenes": 8, "assessed": False}}}, "min_questions": 30, "min_scenes": 10})
    assert "| clean | behavior/MCQ | n/a | 8 | 8 | 0 / 0 | 8 | 0 | 8 | no |" in t1
    # T2/T4: unassessed rows keep n, scenes and the reason
    t2 = t2_headline(_compare())
    assert "| Fog | 20 | 8 | UNASSESSED (coverage: 20 questions from 8 scenes, below 30 / 10) |" in t2
    assert "| clean | 260 | 116 | 0.800 / 0.780 | -0.020 [-0.050, +0.010] | 2 | 1 | 1 | 0.100 [0.020, 0.200] | 0.100 [0.050, 0.150] |" in t2
    # step-3 review item 2: missing per-condition groups are reported, never read as one condition
    one = _compare()
    del one["mcq"]["groups"]["condition"]
    assert "_05 wrote no per-condition groups" in t2_headline(one) and "one condition" not in t2_headline(one)
    t4 = t4_breakdowns(_compare())
    assert "| 1 | 25 | 9 | UNASSESSED (coverage: 25 questions from 9 scenes, below 30 / 10) |" in t4 and "by margin" in t4 and "by condition" not in t4
    # T3: both intervals, assessed marker, reason with detail, descriptive row outside the family, C4 audit
    t3 = t3_contrasts(_compare())
    assert "family of 5, Bonferroni α = 0.010" in t3
    meta = _compare()
    meta["contrast_settings"].update({"family_size": 7, "alpha_adjusted": 0.00714})
    assert "family of 7, Bonferroni α = 0.007" in t3_contrasts(meta)  # producer metadata, not len(rows)
    assert "| C1 | margin [0,0.25) vs other margin bins | clean, pairs with a margin |" in t3 and "| +0.150 | [+0.020, +0.280] | [-0.010, +0.310] | 200/200 | assessed * |" in t3
    assert "| C4a | Dark vs clean | common support | 0.200 (n=8, ref-correct 5/4 sc.) |" in t3 and "UNASSESSED (coverage: A: 8 questions from 4 scenes)" in t3
    assert "| views | 1 view vs 6 views |" in t3 and "descriptive (no decision)" in t3 and "Δ_views = −Δ_task" in t3
    assert "C4a {'clean/bf16': 0, 'clean/int4': 1, 'Dark/bf16': 2, 'Dark/int4': 0}" in t3
    # T5: VQA columns n/a when absent
    t5 = t5_stability(_stability())
    assert "| bf16 | clean | m | 3 | 25 | 0.000 | 0.000 | 0.080 | 0.039 | n/a | n/a | 0 |" in t5 and "| int4 | clean | m4 | 3 | 25 | 0.000 | 0.000 | 0.010 | 0.004 | 35 | 0.050 | 0 |" in t5
    # T6: pooled row, per-session rows, reference row, budgets
    t6 = t6_deployment(profile)
    assert "| **bf16** (pooled) | 2 | 4 | 0 |" in t6 and "| bf16 s1 | 1 | 2 | 0 |" in t6 and "| bf16 s2 | 1 | 2 | 0 |" in t6
    assert "(reference, unpooled)" in t6 and "reduction 41.5%" in t6 and "≥ 25%: pass" in t6  # 1 - 12000/20500
    # reference-only input still lists its rows, with a neutral label plus the identity when the file carries one
    ref_only = pool_profiles([], reference=[_session("int4", 1, [0.4], 13000, provenance=False, file="ref.json"), _session("int4", 4, [0.4], 12000, study=False, file="explore.json")])
    t6r = t6_deployment(ref_only)
    assert "| int4 s1 (reference, unpooled) |" in t6r and "| int4 s4 (reference, unpooled: /ckpt/int4) |" in t6r and "official" not in t6r and "≥ 25%: n/a" in t6r
    # T7: letters for MCQ, scores for VQA; limit respected
    t7 = t7_harmful(_compare())
    assert "| q1 | clean | behavior/MCQ | 6 | 0.12 | A → C |" in t7 and "| q2 | Fog | planning/VQA | 6 | n/a | score 85 → 40 |" in t7
    assert "showing 1 of 2" in t7_harmful(_compare(), limit=1)


def test_tables_degrade_without_inputs():
    md = render_all(None, None, None)
    assert md.count("_no comparison JSON_") == 5 and "_no stability summaries_" in md and "_no pooled profile" in md
    assert "run 05 --contrasts" in t3_contrasts({"analysis_class": "exploratory"})


# ---------------------------------------------------------------- integration


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/drivebench-test.json").exists() or not (ROOT / "data/visual_description.json").exists(), reason="data files not downloaded")
def test_cli_vqa_section_written_when_every_judge_record_is_stale(tmp_path):
    """Step-3 follow-up: --vqa with zero usable pairs still writes the VQA
    section with an n = 0 unassessed row per condition, and T2 shows it."""
    import json
    import subprocess
    import sys

    import yaml

    from qflip.judge import sidecar_path
    from qflip.records import read_records
    from qflip.tables import t2_headline

    d = tmp_path / "pilot"
    d.mkdir()
    for name in ("bf16_clean.jsonl", "int4_clean.jsonl"):
        src = ROOT / "results/pilot" / name
        (d / name).write_text(src.read_text())
        stale = [{"question_id": r["question_id"], "scene_token": r["scene_token"], "cell": r["cell"], "replicate": 0, "input_sha256": "0" * 64, "desc": None, "score": 50}
                 for r in read_records(src) if r["cell"].endswith("VQA")]
        sidecar_path(d / name).write_text("".join(json.dumps(r) + "\n" for r in stale))
    cfg = yaml.safe_load((ROOT / "config/pilot.yaml").read_text())
    cfg["judge"]["model"] = "probe-model"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "compare.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts/05_compare_flips.py"), str(d / "bf16_clean.jsonl"), str(d / "int4_clean.jsonl"), "--no-provenance", "--vqa",
                          "--judge-threshold", "60", "--config", str(cfg_path), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "0 usable judge scores, 12 stale" in res.stdout and "0 paired VQA" in res.stdout
    j = json.loads(out.read_text())
    assert j["vqa"]["threshold"] == 60.0 and j["vqa"]["overall"]["n"] == 0 and not j["vqa"]["overall"]["assessed"]
    assert j["vqa"]["groups"]["condition"] == [{"label": "clean", "n": 0, "scenes": 0, "assessed": False, "reason": "coverage: 0 questions from 0 scenes, below 30 / 10"}]
    t2 = t2_headline(j)
    assert "VQA (judged, threshold 60.0)" in t2 and "| clean | 0 | 0 | UNASSESSED (coverage: 0 questions from 0 scenes, below 30 / 10) |" in t2
    assert j["coverage"]["clean"]["planning/VQA"]["paired"] == 0 and j["coverage"]["clean"]["planning/VQA"]["unpaired"] == 4
