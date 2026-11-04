from qflip.flips import bootstrap_scenes, classify, group_by, margin_bin, pair_records, summarize


def rec(qid, scene, correct, letter, cell="behavior/MCQ", n_views=6, margin=0.4, error=None, precision="bf16"):
    return {"question_id": qid, "scene_token": scene, "frame_token": "f", "cell": cell, "n_views": n_views, "views": ["CAM_FRONT"] * n_views,
            "condition": "clean", "precision": precision, "question": f"Q{qid}", "mcq_gt": "A",
            "margin": margin, "mcq_correct": correct, "mcq_pred_strict": letter, "error": error}


def qrec(*a, **k):
    return rec(*a, precision="int4", **k)


def test_classify():
    assert classify(rec("q", "s", True, "A"), rec("q", "s", False, "B")) == "harmful"
    assert classify(rec("q", "s", False, "B"), rec("q", "s", True, "A")) == "beneficial"
    assert classify(rec("q", "s", False, "B"), rec("q", "s", False, "C")) == "neutral_change"
    assert classify(rec("q", "s", True, "A"), rec("q", "s", True, "A")) == "unchanged"


def test_pair_records_drops_errors_vqa_and_unmatched():
    ref = [rec("1", "s", True, "A"), rec("2", "s", True, "A", error="boom"), rec("3", "s", True, "A", cell="planning/VQA"), rec("4", "s", True, "A")]
    qnt = [qrec("1", "s", False, "B"), qrec("2", "s", True, "A"), qrec("3", "s", True, "A", cell="planning/VQA")]
    pairs = pair_records(ref, qnt)
    assert [p.question_id for p in pairs] == ["1"]
    assert pairs[0].flip == "harmful"


def test_summarize_rates():
    ref = [rec("1", "a", True, "A"), rec("2", "a", True, "A"), rec("3", "b", False, "B"), rec("4", "b", False, "C")]
    qnt = [qrec("1", "a", False, "B"), qrec("2", "a", True, "A"), qrec("3", "b", True, "A"), qrec("4", "b", False, "C")]
    s = summarize(pair_records(ref, qnt))
    assert s["n"] == 4 and s["scenes"] == 2
    assert s["harmful"] == 1 and s["beneficial"] == 1 and s["unchanged"] == 2
    assert s["harmful_rate"] == 0.5          # 1 of 2 reference-correct lost
    assert s["acc_delta"] == 0.0
    assert s["change_rate"] == 0.5


def test_bootstrap_is_deterministic_and_bracketed():
    ref, qnt = [], []
    for i in range(40):
        scene = f"s{i % 10}"
        ref.append(rec(str(i), scene, True, "A"))
        qnt.append(qrec(str(i), scene, i % 4 != 0, "A" if i % 4 else "B"))  # 25% harmful
    pairs = pair_records(ref, qnt)
    ci1 = bootstrap_scenes(pairs, n_boot=300, seed=0)
    ci2 = bootstrap_scenes(pairs, n_boot=300, seed=0)
    assert ci1 == ci2
    lo, hi = ci1["harmful_rate"]
    assert lo <= 0.25 <= hi
    assert bootstrap_scenes([p for p in pairs if p.scene_token == "s0"], n_boot=10) == {}  # single scene -> no CI


def test_margin_bin_and_group_by():
    assert margin_bin(None, (0.25, 0.5)) == "no_margin"
    assert margin_bin(0.1, (0.25, 0.5)) == "[0,0.25)"
    assert margin_bin(0.25, (0.25, 0.5)) == "[0.25,0.5)"
    assert margin_bin(0.9, (0.25, 0.5)) == "[0.5,1]"
    pairs = pair_records(
        [rec("1", "s", True, "A", n_views=1, margin=0.1), rec("2", "s", True, "A", n_views=6, margin=0.7)],
        [qrec("1", "s", True, "A", n_views=1), qrec("2", "s", True, "A", n_views=6)],
    )
    assert list(group_by(pairs, "n_views")) == ["1", "6"]
    assert list(group_by(pairs, "margin")) == ["[0,0.25)", "[0.5,1]"]


def test_pairing_errors_name_the_field():
    import pytest

    from qflip.flips import PairingError, check_pair

    base = {"question_id": "q", "precision": "bf16", "condition": "clean", "mcq_gt": "A", "question": "Q?", "scene_token": "s", "frame_token": "f",
            "cell": "behavior/MCQ", "views": ["CAM_FRONT"], "mcq_correct": True, "mcq_pred_strict": "A"}
    other = {**base, "precision": "int4"}
    check_pair(base, other)  # identical apart from precision: fine
    for field, bad in [("condition", "NoImage"), ("mcq_gt", "B"), ("question", "Other?"), ("scene_token", "t"), ("frame_token", "g"), ("cell", "perception/MCQ"), ("views", ["CAM_BACK"])]:
        with pytest.raises(PairingError, match=field.replace("mcq_", "")):
            check_pair(base, {**other, field: bad})
    with pytest.raises(PairingError, match="reference=bf16, quantized=int4"):
        check_pair(base, {**base})  # both bf16
    with pytest.raises(PairingError, match="reference=bf16, quantized=int4"):
        check_pair(base, {**other, "precision": "fp8"})
    # legacy records without question text: annotation lookup
    legacy_a, legacy_b = {k: v for k, v in base.items() if k != "question"}, {k: v for k, v in other.items() if k != "question"}
    with pytest.raises(PairingError, match="question text missing"):
        check_pair(legacy_a, legacy_b)
    check_pair(legacy_a, legacy_b, question_text={"q": "Q?"})
    # pair_records refuses a mixed-condition comparison instead of producing flips
    with pytest.raises(PairingError, match="condition"):
        pair_records([base], [{**other, "condition": "NoImage", "mcq_correct": False, "mcq_pred_strict": "B"}])


def test_pairing_requires_reference_then_quantized_order():
    """Review item 3: reversed inputs would swap harmful and beneficial; refused."""
    import pytest

    from qflip.flips import PairingError

    ref = rec("q1", "s", True, "A", precision="bf16")
    quant = rec("q1", "s", False, "B", precision="int4")
    assert [p.flip for p in pair_records([ref], [quant])] == ["harmful"]
    with pytest.raises(PairingError, match="reference=bf16, quantized=int4"):
        pair_records([quant], [ref])
    with pytest.raises(PairingError, match="reference=bf16"):
        pair_records([ref], [rec("q1", "s", False, "B", precision="bf16")])
