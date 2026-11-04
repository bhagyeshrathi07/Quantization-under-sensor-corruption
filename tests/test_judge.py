import json

import pytest

from qflip.flips import coverage, pair_vqa_records, summarize
from qflip.judge import (
    VQA_CELLS,
    build_prompt,
    extract_score,
    judge_messages,
    load_scores,
    needs_description,
    scored_ids,
    sidecar_path,
    template_sha,
    visual_description,
)

DESC = {
    "scene": {
        "key_frames": {
            "frame": {
                "key_object_infos": {
                    "<c1,CAM_FRONT,767.5,513.3>": {"Visual_description": "White sedan car."},
                    "<c2,CAM_BACK,10.0,20.0>": {"Visual_description": "Red truck."},
                }
            }
        }
    }
}


def test_visual_description_lookup_matches_upstream():
    assert visual_description(DESC, "scene", "frame", "What is <c1,CAM_FRONT,767.5,513.3> doing?") == "white sedan car."
    assert visual_description(DESC, "scene", "frame", "Where is <c2,CAM_BACK,10.0,20.0>?") == "red truck."
    assert visual_description(DESC, "scene", "frame", "No object tag here") is None
    assert visual_description(DESC, "scene", "frame", "In this scenario, what are safe actions to take for the ego vehicle?").startswith("No visual description")
    with pytest.raises(ValueError):
        visual_description(DESC, "scene", "frame", "What about <c9,CAM_FRONT,1,1>?")


def test_prompt_build_and_description_requirement():
    assert needs_description("planning/VQA")
    assert not needs_description("prediction/VQA")
    assert not needs_description("perception/VQA")
    p = build_prompt("planning/VQA", "Q?", "turn left", "go straight", "white sedan car.")
    assert '"white sedan car."' in p and '"turn left"' in p and '"go straight"' in p and '"Q?"' in p
    with pytest.raises(KeyError):
        build_prompt("planning/VQA", "Q?", "gt", "pred", None)  # planning template needs DESC
    p2 = build_prompt("prediction/VQA", "Q?", "gt", "pred", None)
    assert p2.endswith("Brief Summary: ")
    assert len(template_sha("planning/VQA")) == 12
    assert set(VQA_CELLS) == {"perception/VQA", "prediction/VQA", "planning/VQA"}


def test_extract_score_and_messages():
    assert extract_score("...\nTotal Score: 85\n\nBrief Summary: fine") == 85
    assert extract_score("Total Score:100") == 100
    assert extract_score("no score here") is None
    assert extract_score("") is None
    m = judge_messages("PROMPT")
    assert len(m) == 1 and m[0]["role"] == "user" and m[0]["content"].startswith("You are an evaluator") and m[0]["content"].endswith("\nPROMPT")


def test_sidecar_resume(tmp_path):
    res = tmp_path / "bf16_clean.jsonl"
    side = sidecar_path(res)
    assert side.name == "bf16_clean.judge.jsonl"
    side.write_text("\n".join(json.dumps(r) for r in [
        {"question_id": "a", "score": 70},
        {"question_id": "b", "score": None, "error": "no Total Score in reply"},
        {"question_id": "b", "score": 40},
        {"question_id": "c", "score": None, "error": "timeout"},
    ]) + "\n")
    scores = load_scores(side)
    assert scores["b"]["score"] == 40  # last record wins
    assert scored_ids(scores) == {"a", "b"}
    assert load_scores(tmp_path / "missing.jsonl") == {}


def _rec(qid, scene, cell="planning/VQA", error=None, precision="bf16"):
    return {"question_id": qid, "scene_token": scene, "frame_token": "f", "cell": cell, "n_views": 6, "views": ["CAM_FRONT"] * 6,
            "condition": "clean", "precision": precision, "question": f"Q{qid}", "gt": "turn left", "error": error}


def test_pair_vqa_records_threshold_and_classes():
    ref = [_rec("1", "s1"), _rec("2", "s1"), _rec("3", "s2"), _rec("4", "s2"), _rec("5", "s2", cell="behavior/MCQ"), _rec("6", "s3")]
    qnt = [{**r, "precision": "int4"} for r in ref]
    rs = {"1": {"score": 80}, "2": {"score": 30}, "3": {"score": 70}, "4": {"score": 20}, "5": {"score": 90}, "6": {"score": 50}}
    qs = {"1": {"score": 40}, "2": {"score": 75}, "3": {"score": 65}, "4": {"score": 10}, "5": {"score": 90}, "6": {"score": None}}
    pairs = pair_vqa_records(ref, qnt, rs, qs, threshold=60)
    assert [p.question_id for p in pairs] == ["1", "2", "3", "4"]  # MCQ and unscored dropped
    assert [p.flip for p in pairs] == ["harmful", "beneficial", "unchanged", "unchanged"]
    assert pairs[0].margin is None and pairs[0].ref_score == 80
    s = summarize(pairs)
    assert s["harmful"] == 1 and s["beneficial"] == 1 and s["neutral_change"] == 0 and s["change_rate"] == 0.5
    assert s["harmful_rate"] == 0.5 and s["ref_score_mean"] == 50.0 and s["quant_score_mean"] == 47.5


def test_coverage_rule():
    assert coverage({"n": 30, "scenes": 10}, 30, 10)
    assert not coverage({"n": 29, "scenes": 10}, 30, 10)
    assert not coverage({"n": 30, "scenes": 9}, 30, 10)
    assert not coverage({"n": 0}, 1, 1)
