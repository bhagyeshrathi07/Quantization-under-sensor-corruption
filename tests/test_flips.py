from qflip.flips import bootstrap_scenes, classify, group_by, margin_bin, pair_records, summarize


def rec(qid, scene, correct, letter, cell="behavior/MCQ", n_views=6, margin=0.4, error=None):
    return {"question_id": qid, "scene_token": scene, "cell": cell, "n_views": n_views, "condition": "clean",
            "margin": margin, "mcq_correct": correct, "mcq_pred_strict": letter, "error": error}


def test_classify():
    assert classify(rec("q", "s", True, "A"), rec("q", "s", False, "B")) == "harmful"
    assert classify(rec("q", "s", False, "B"), rec("q", "s", True, "A")) == "beneficial"
    assert classify(rec("q", "s", False, "B"), rec("q", "s", False, "C")) == "neutral_change"
    assert classify(rec("q", "s", True, "A"), rec("q", "s", True, "A")) == "unchanged"


def test_pair_records_drops_errors_vqa_and_unmatched():
    ref = [rec("1", "s", True, "A"), rec("2", "s", True, "A", error="boom"), rec("3", "s", True, "A", cell="planning/VQA"), rec("4", "s", True, "A")]
    qnt = [rec("1", "s", False, "B"), rec("2", "s", True, "A"), rec("3", "s", True, "A", cell="planning/VQA")]
    pairs = pair_records(ref, qnt)
    assert [p.question_id for p in pairs] == ["1"]
    assert pairs[0].flip == "harmful"


def test_summarize_rates():
    ref = [rec("1", "a", True, "A"), rec("2", "a", True, "A"), rec("3", "b", False, "B"), rec("4", "b", False, "C")]
    qnt = [rec("1", "a", False, "B"), rec("2", "a", True, "A"), rec("3", "b", True, "A"), rec("4", "b", False, "C")]
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
        qnt.append(rec(str(i), scene, i % 4 != 0, "A" if i % 4 else "B"))  # 25% harmful
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
        [rec("1", "s", True, "A"), rec("2", "s", True, "A")],
    )
    assert list(group_by(pairs, "n_views")) == ["1", "6"]
    assert list(group_by(pairs, "margin")) == ["[0,0.25)", "[0.5,1]"]
