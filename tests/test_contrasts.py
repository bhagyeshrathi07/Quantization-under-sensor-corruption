"""Prespecified contrasts: planted effect, null, reproducibility, support rule,
family size, descriptive views row."""

import pytest

from qflip.contrasts import FAMILY_SIZE, bootstrap_contrast, common_support, format_contrast, prespecified_contrasts
from qflip.flips import Pair, PairingError, merge_pairs

SETTINGS = {"margin_edges": [0.25, 0.5], "min_questions": 30, "min_scenes": 10, "n_boot_contrast": 2000, "support_min_ref_correct": 20, "support_min_ref_scenes": 10}


def pair(qid, scene, flip, cell="behavior/MCQ", n_views=6, margin=0.4, condition="clean"):
    ref_ok = flip in ("harmful", "unchanged", "neutral_change")
    return Pair(question_id=qid, scene_token=scene, cell=cell, n_views=n_views, condition=condition, margin=margin,
                ref_correct=ref_ok, quant_correct=(flip in ("beneficial", "unchanged", "neutral_change")), flip=flip)


def side(prefix, scenes, harmful_frac, **kw):
    """Per scene: 4 reference-correct pairs with the given harmful fraction."""
    out = []
    for s in range(scenes):
        for k in range(4):
            flip = "harmful" if k < 4 * harmful_frac else "unchanged"
            out.append(pair(f"{prefix}{s}-{k}", f"s{s}", flip, **kw))
    return out


def test_planted_effect_and_null():
    a = side("a", 20, 0.5, cell="behavior/MCQ")
    b = side("b", 20, 0.0, cell="perception/MCQ", n_views=1)
    r = bootstrap_contrast(a, b, 2000, seed=0)
    assert r["delta"] == 0.5 and r["valid_draws"] == 2000
    assert r["excludes_zero_raw"] and r["excludes_zero_adjusted"]
    assert r["ci95"][0] > 0 and r["ci99_bonferroni"][0] > 0 and r["ci99_bonferroni"][0] <= r["ci95"][0]
    null = bootstrap_contrast(side("a", 20, 0.25), side("b", 20, 0.25, cell="perception/MCQ"), 2000)
    assert null["delta"] == 0.0 and null["ci95"][0] <= 0 <= null["ci95"][1] and not null["excludes_zero_adjusted"]


def test_same_seed_reproduces_and_seed_changes_draws():
    # heterogeneous scenes (F_h differs between scenes) so the interval is not degenerate
    a = [pair(f"a{s}-{k}", f"s{s}", "harmful" if k < s % 4 else "unchanged") for s in range(15) for k in range(4)]
    b = [pair(f"b{s}-{k}", f"s{s}", "harmful" if k < (s + 1) % 3 else "unchanged", cell="perception/MCQ") for s in range(15) for k in range(4)]
    r1, r2, r3 = bootstrap_contrast(a, b, 500, 0), bootstrap_contrast(a, b, 500, 0), bootstrap_contrast(a, b, 500, 1)
    assert r1 == r2 and r1["ci95"] != r3["ci95"]


def test_support_rule_and_family_size():
    clean = side("a", 20, 0.5, cell="behavior/MCQ") + [pair(f"b{i}", f"s{i % 20}", "beneficial", cell="perception/MCQ", n_views=1) for i in range(80)]
    res = prespecified_contrasts({"clean": clean}, SETTINGS)
    fam = {r["name"]: r for r in res["family"]}
    assert [r["name"] for r in res["family"]] == ["C1", "C2", "C4a", "C4b", "C4c"] and len(res["family"]) == FAMILY_SIZE
    c2 = fam["C2"]
    assert c2["b"]["ref_correct"] == 0 and c2["assessed"] is False and c2["reason"] == "support" and c2["valid_draws"] == 0
    assert c2["ci95"] is None and c2["delta"] is None
    # C1: every pair has margin 0.4 -> lowest bin empty -> coverage
    assert fam["C1"]["assessed"] is False and fam["C1"]["reason"] == "coverage"
    # C4: no stress files -> coverage, sides empty, dropped counts present
    for k in ("C4a", "C4b", "C4c"):
        assert fam[k]["assessed"] is False and fam[k]["reason"] == "coverage" and fam[k]["a"]["n"] == 0 and fam[k]["support_dropped"] == {"clean": 160, "condition": 0}
    # views absent from the family, present under descriptive with the note
    assert "views" not in fam and res["descriptive"][0]["name"] == "views" and res["descriptive"][0]["confirmatory"] is False
    assert "Δ_views = −Δ_task" in res["descriptive"][0]["note"]
    assert res["alpha_adjusted"] == 0.01


def test_condition_contrast_uses_common_support_and_assesses():
    clean = side("c", 20, 0.0, cell="behavior/MCQ")
    fog = [pair(p.question_id, p.scene_token, "harmful" if int(p.question_id.split("-")[1]) < 2 else "unchanged", condition="Fog") for p in clean]
    fog_extra = fog + [pair("extra", "s0", "unchanged", condition="Fog")]
    a, b, dropped = common_support(clean, fog_extra)
    assert len(a) == len(b) == 80 and dropped == {"clean": 0, "condition": 1}
    res = prespecified_contrasts({"clean": clean, "Fog": fog_extra}, SETTINGS)
    c4c = next(r for r in res["family"] if r["name"] == "C4c")
    assert c4c["assessed"] and c4c["delta"] == 0.5 and c4c["excludes_zero_adjusted"] and c4c["support_dropped"] == {"clean": 0, "condition": 1}
    assert c4c["a_label"] == "Fog" and c4c["b_label"] == "clean"


def test_valid_draws_rule():
    # each side has reference-correct pairs in one scene only (A: s1, B: s0); a draw is valid
    # only when both scenes are drawn, about 40% of draws -> valid_draws < 50%
    a = [pair(f"a{k}", "s1", "harmful" if k < 5 else "unchanged") for k in range(25)] + [pair(f"ax{i}", f"s{i}", "beneficial") for i in range(2, 21)]
    b = [pair(f"b{k}", "s0", "unchanged", cell="perception/MCQ") for k in range(25)] + [pair(f"bx{i}", f"s{i}", "beneficial", cell="perception/MCQ") for i in range(2, 21)]
    s = {**SETTINGS, "support_min_ref_scenes": 1}
    res = prespecified_contrasts({"clean": a + b}, s)
    c2 = next(r for r in res["family"] if r["name"] == "C2")
    assert c2["reason"] == "valid_draws" and 0 < c2["valid_draws"] < 0.5 * s["n_boot_contrast"]
    # below half valid draws the intervals and both flags are unavailable, the point estimate stays
    assert c2["ci95"] is None and c2["ci99_bonferroni"] is None and c2["excludes_zero_raw"] is False and c2["excludes_zero_adjusted"] is False
    assert c2["delta"] == 0.2
    assert "UNASSESSED (valid_draws" in format_contrast(c2) and "*" not in format_contrast(c2)


def test_intervals_cleared_below_half_valid_draws_whatever_fails_first():
    """Follow-up: the interval rule applies by valid_draws alone, also when coverage or
    support is the reported reason."""
    a = [pair(f"a{k}", "s1", "harmful" if k < 5 else "unchanged") for k in range(25)] + [pair(f"ax{i}", f"s{i}", "beneficial") for i in range(2, 21)]
    b = [pair(f"b{k}", "s0", "unchanged", cell="perception/MCQ") for k in range(25)] + [pair(f"bx{i}", f"s{i}", "beneficial", cell="perception/MCQ") for i in range(2, 21)]
    boot = bootstrap_contrast(a, b, 2000)
    assert 0 < boot["valid_draws"] < 1000 and boot["intervals_available"] is False
    assert boot["ci95"] is None and boot["ci99_bonferroni"] is None and boot["excludes_zero_raw"] is False and boot["excludes_zero_adjusted"] is False
    assert boot["delta"] == 0.2
    res = prespecified_contrasts({"clean": a + b}, SETTINGS)  # support_min_ref_scenes 10 -> reason is support, not valid_draws
    c2 = next(r for r in res["family"] if r["name"] == "C2")
    assert c2["reason"] == "support" and c2["ci95"] is None and c2["ci99_bonferroni"] is None and c2["excludes_zero_adjusted"] is False and c2["delta"] == 0.2
    small = prespecified_contrasts({"clean": a[:5] + a[25:] + b[:5] + b[25:]}, SETTINGS)  # 24 per side -> coverage; ~40% valid draws
    c2 = next(r for r in small["family"] if r["name"] == "C2")
    assert c2["reason"] == "coverage" and c2["ci95"] is None and c2["excludes_zero_raw"] is False


def test_nonpositive_draw_counts_are_rejected():
    a, b = side("a", 20, 0.5), side("b", 20, 0.0, cell="perception/MCQ")
    for n in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            bootstrap_contrast(a, b, n)
        with pytest.raises(ValueError):
            prespecified_contrasts({"clean": a + b}, {**SETTINGS, "n_boot_contrast": n})


def test_descriptive_views_row_carries_no_decision():
    clean = side("a", 20, 0.5, cell="behavior/MCQ", n_views=6) + side("b", 20, 0.0, cell="perception/MCQ", n_views=1)
    res = prespecified_contrasts({"clean": clean}, SETTINGS)
    v = res["descriptive"][0]
    assert v["assessed"] and v["delta"] == -0.5 and v["ci95"] is not None
    assert v["ci99_bonferroni"] is None and v["excludes_zero_adjusted"] is None and v["confirmatory"] is False
    line = format_contrast(v)
    assert "DESCRIPTIVE (no decision)" in line and "*" not in line
    c2 = next(r for r in res["family"] if r["name"] == "C2")
    assert c2["delta"] == 0.5 and "ASSESSED *" in format_contrast(c2)


def test_duplicate_inputs_are_refused():
    clean = side("a", 20, 0.5)
    with pytest.raises(PairingError, match="appears in file pair 1 and again in file pair 2"):
        merge_pairs([clean, clean])
    merge_pairs([clean, [pair(p.question_id, p.scene_token, p.flip, condition="Fog") for p in clean]])  # same ids, other condition: fine


def test_common_support_counts_all_four_files():
    clean = side("c", 20, 0.0)
    fog = [pair(p.question_id, p.scene_token, "unchanged", condition="Fog") for p in clean]
    ids = {p.question_id for p in clean}
    file_ids = {"clean/bf16": ids | {"e1", "e2"}, "clean/int4": ids, "Fog/bf16": ids | {"e3"}, "Fog/int4": ids - {"c0-0"}}
    a, b, dropped = common_support(clean, fog, file_ids)
    assert dropped == {"clean/bf16": 3, "clean/int4": 1, "Fog/bf16": 2, "Fog/int4": 0} and len(a) == len(b) == 79
    res = prespecified_contrasts({"clean": clean, "Fog": fog, "NoImage": fog}, SETTINGS, file_ids=file_ids)
    fam = {r["name"]: r for r in res["family"]}
    assert fam["C4c"]["support_dropped"] == dropped and fam["C4c"]["a"]["n"] == 79
    assert fam["C4a"]["support_dropped"] == {"clean": 0, "condition": 0}  # NoImage has no file ids: pair-level fallback
