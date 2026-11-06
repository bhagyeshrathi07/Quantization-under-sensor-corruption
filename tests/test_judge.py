import json
from types import SimpleNamespace

import pytest

from qflip.flips import coverage, pair_vqa_records, summarize
from qflip.judge import (
    OPTIONAL_PARAMS,
    VISUAL_DESCRIPTION_SHA256,
    VQA_CELLS,
    JudgeInputError,
    JudgeSettingsError,
    build_prompt,
    compat_check,
    ensure_visual_description,
    expected_description,
    extract_score,
    input_sha256,
    invocation_usage,
    judge_fingerprints,
    judge_input_hash,
    judge_messages,
    judge_settings,
    load_scores,
    needs_description,
    production_scores,
    request_body,
    request_judge,
    score_key,
    scored_ids,
    scored_keys,
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

CFG = {"model": "judge-x", "temperature": 0.0, "top_p": None, "reasoning_effort": None, "seed": None, "max_tokens": 512, "max_tokens_param": "max_tokens"}


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


# ---------------------------------------------------------------- step 4: settings, identity, resume


def test_optional_settings_omitted_when_null():
    s = judge_settings(CFG)
    assert s == {"base_url": "https://api.openai.com/v1", "model": "judge-x", "max_tokens": 512, "max_tokens_param": "max_tokens", "temperature": 0.0}
    assert judge_settings({**CFG, "base_url": "http://judge.local:8001/v1/"})["base_url"] == "http://judge.local:8001/v1"
    body = request_body(s, "P")
    assert set(body) == {"model", "messages", "max_tokens", "temperature"}  # the endpoint is not a request field
    assert not ({"top_p", "reasoning_effort", "seed"} & set(body))
    s2 = judge_settings({**CFG, "temperature": None, "reasoning_effort": "none", "max_tokens_param": "max_completion_tokens"})
    body2 = request_body(s2, "P")
    assert "temperature" not in body2 and body2["reasoning_effort"] == "none"
    assert body2["max_completion_tokens"] == 512 and "max_tokens" not in body2
    with pytest.raises(JudgeSettingsError):
        judge_settings({**CFG, "model": None})
    with pytest.raises(JudgeSettingsError):
        judge_settings({**CFG, "max_tokens_param": "tokens"})
    assert set(OPTIONAL_PARAMS) == {"temperature", "top_p", "reasoning_effort", "seed"}


def test_input_sha256_changes_when_any_input_changes():
    s = judge_settings(CFG)
    base = dict(cell="planning/VQA", question="Q?", gt="turn left", pred="go straight", desc="white sedan car.", settings=s)
    h0 = judge_input_hash(**base)
    assert h0 == judge_input_hash(**base) and len(h0) == 64
    variants = [
        {"pred": "go straight."},
        {"question": "Q!"},
        {"gt": "turn right"},
        {"desc": "red truck."},
        {"cell": "perception/VQA", "desc": None},
        {"settings": {**s, "model": "judge-y"}},
        {"settings": {**s, "temperature": 0.5}},
        {"settings": {k: v for k, v in s.items() if k != "temperature"}},
        {"settings": {**s, "reasoning_effort": "low"}},
        {"settings": {**s, "max_tokens": 256}},
        {"settings": {**s, "max_tokens_param": "max_completion_tokens"}},
        {"settings": {**s, "seed": 1}},
        {"settings": {**s, "base_url": "http://judge.local:8001/v1"}},   # same request to another provider is another judge
    ]
    hashes = {judge_input_hash(**{**base, **v}) for v in variants}
    assert len(hashes) == len(variants) and h0 not in hashes
    # the hash is over the serialized request: key order does not matter, values do
    body = request_body(s, "P")
    assert input_sha256(dict(reversed(list(body.items()))), s["base_url"]) == input_sha256(body, s["base_url"])


def test_replicate_rows_coexist_and_resume_skips_only_exact_keys(tmp_path):
    res = tmp_path / "bf16_clean.jsonl"
    side = sidecar_path(res)
    assert side.name == "bf16_clean.judge.jsonl"
    side.write_text("\n".join(json.dumps(r) for r in [
        {"question_id": "a", "replicate": 0, "input_sha256": "h1", "score": 70},
        {"question_id": "a", "replicate": 1, "input_sha256": "h1", "score": 65},
        {"question_id": "a", "replicate": 2, "input_sha256": "h1", "score": None, "error": "timeout"},
        {"question_id": "b", "replicate": 0, "input_sha256": "h2", "score": None, "error": "no Total Score in reply"},
        {"question_id": "b", "replicate": 0, "input_sha256": "h2", "score": 40},
        {"question_id": "b", "replicate": 0, "input_sha256": "h3", "score": 90},
        {"question_id": "legacy", "score": 55},
    ]) + "\n")
    scores = load_scores(side)
    assert len(scores) == 6  # replicates and re-judged answers coexist; the retry of (b,0,h2) collapsed
    assert scores[("a", 0, "h1")]["score"] == 70 and scores[("a", 1, "h1")]["score"] == 65
    assert scores[("b", 0, "h2")]["score"] == 40  # last record of the same key wins
    assert scores[("b", 0, "h3")]["score"] == 90
    assert score_key({"question_id": "legacy", "score": 55}) == ("legacy", 0, None)
    assert scored_keys(scores) == {("a", 0, "h1"), ("a", 1, "h1"), ("b", 0, "h2"), ("b", 0, "h3"), ("legacy", 0, None)}
    assert scored_ids(scores) == {"a", "b", "legacy"}
    assert load_scores(tmp_path / "missing.jsonl") == {}
    # resume: only the exact key is skipped; replicate 2 of a (errored) and a new hash are judged
    done = scored_keys(scores)
    assert ("a", 2, "h1") not in done and ("a", 0, "h9") not in done and ("a", 0, "h1") in done


def _rec(qid, scene, cell="planning/VQA", error=None, precision="bf16", pred="go straight", question=None):
    return {"question_id": qid, "scene_token": scene, "frame_token": "frame", "cell": cell, "n_views": 6, "views": ["CAM_FRONT"] * 6,
            "condition": "clean", "precision": precision, "question": question or f"What is <c1,CAM_FRONT,767.5,513.3> doing? Q{qid}", "gt": "turn left", "pred": pred, "error": error}


# description data for _rec: every scene used in the tests carries the two objects of DESC
DESC_DATA = {s: DESC["scene"] for s in ("s1", "s2", "s3", "s4")}


def _judged(rec, settings, replicate=0, score=80, desc="white sedan car.", pred=None):
    h = judge_input_hash(rec["cell"], rec["question"], rec["gt"], pred if pred is not None else rec["pred"], desc, settings)
    return {"question_id": rec["question_id"], "replicate": replicate, "input_sha256": h, "desc": desc, "score": score}


def test_production_scores_use_replicate_zero_matching_current_answer_and_settings():
    s = judge_settings(CFG)
    recs = [_rec("a", "s1"), _rec("b", "s1"), _rec("c", "s2"), _rec("d", "s2"), _rec("e", "s3"), _rec("f", "s3", error="timeout"), _rec("g", "s4"),
            _rec("h", "s4", question="What about <c9,CAM_FRONT,1,1>? Qh")]
    rows = [
        _judged(recs[0], s, 0, 80), _judged(recs[0], s, 1, 60), _judged(recs[0], s, 2, 70),   # a: replicates; only 0 is production
        _judged(recs[1], s, 0, 90, pred="old answer"),                                        # b: judged before the answer changed
        _judged(recs[2], {**s, "model": "judge-old"}, 0, 50),                                 # c: other model
        _judged(recs[3], s, 1, 75),                                                           # d: replicate 0 missing
        {"question_id": "e", "score": 95, "desc": "white sedan car."},                        # e: pre-step-4 record
        _judged(recs[5], s, 0, 85),                                                           # f: answer errored now
        _judged(recs[7], s, 0, 85, desc="guess"),                                             # h: object tag not in the description file
    ]
    scores = {score_key(r): r for r in rows}
    selected, stale = production_scores(scores, recs, s, DESC_DATA)
    assert set(selected) == {"a"} and selected["a"]["score"] == 80
    assert set(stale) == {"b", "c", "d", "e", "h"}
    assert stale["d"] == "replicate 0 missing" and "pre-step-4" in stale["e"] and stale["h"].startswith("description lookup")
    assert "g" not in stale  # never judged: not stale, simply absent
    # a changed setting invalidates every production score
    selected2, stale2 = production_scores(scores, recs, {**s, "temperature": 0.3}, DESC_DATA)
    assert selected2 == {} and "a" in stale2
    # a sidecar from a run with a different max-tokens parameter is not reused either
    selected3, _ = production_scores(scores, recs, {**s, "max_tokens_param": "max_completion_tokens"}, DESC_DATA)
    assert selected3 == {}


def test_production_description_derived_from_the_checked_file_not_from_stored_rows():
    """Review item 3: two stored rows with different descriptions for the same
    answer; the one matching the description file wins, in every row order."""
    s = judge_settings(CFG)
    rec = _rec("a", "s1")
    good = _judged(rec, s, 0, 90, desc="white sedan car.")
    other = _judged(rec, s, 0, 10, desc="red truck.")
    for rows in ((good, other), (other, good)):
        scores = {score_key(r): r for r in rows}
        assert len(scores) == 2
        selected, stale = production_scores(scores, [rec], s, DESC_DATA)
        assert selected["a"]["score"] == 90 and stale == {}
    # description file says the object is something else: neither row matches, reported stale
    desc2 = {"s1": {"key_frames": {"frame": {"key_object_infos": {"<c1,CAM_FRONT,767.5,513.3>": {"Visual_description": "Blue van."}}}}}}
    selected, stale = production_scores({score_key(r): r for r in (good, other)}, [rec], s, desc2)
    assert selected == {} and "a" in stale
    # cells without a description placeholder never carry one
    assert expected_description(None, _rec("p", "s1", cell="prediction/VQA"), "Q") is None
    with pytest.raises(JudgeInputError):
        expected_description(None, rec, rec["question"])


def test_visual_description_file_must_match_the_recorded_hash(tmp_path):
    f = tmp_path / "visual_description.json"
    f.write_text("{}")
    with pytest.raises(JudgeInputError, match="differs from the recorded"):
        ensure_visual_description(f)
    assert len(VISUAL_DESCRIPTION_SHA256) == 64


def test_invocation_usage_counts_only_the_written_records():
    """Review item 4: the trial's measured usage covers this invocation's calls, not historical rows."""
    written = [
        {"score": 70, "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}},
        {"score": 60, "usage": {"prompt_tokens": 1100, "completion_tokens": 300, "total_tokens": 1400, "reasoning_tokens": 50}},
        {"score": None, "usage": None, "error": "timeout"},
    ]
    u = invocation_usage(written)
    assert u["calls"] == 3 and u["with_usage"] == 2 and u["errors"] == 1
    assert u["prompt_tokens_mean"] == 1050.0 and u["completion_tokens_mean"] == 250.0 and u["reasoning_tokens"] == 50
    assert invocation_usage([])["prompt_tokens_mean"] is None


def test_judge_fingerprints_cover_system_line_and_every_template():
    fp = judge_fingerprints()
    assert len(fp["system_line_sha256"]) == 64 and fp["visual_description_sha256"] == VISUAL_DESCRIPTION_SHA256
    assert set(fp["templates"]) == {"perception/MCQ", "perception/VQA", "prediction/VQA", "planning/VQA", "behavior/MCQ"}
    assert fp["templates"]["planning/VQA"] == template_sha("planning/VQA")


class _FakeClient:
    """Minimal openai-like client: `rejects` names parameters that raise a 400."""

    def __init__(self, rejects=(), reply="Total Score: 77\n\nBrief Summary: ok", usage=(900, 180)):
        self.rejects, self.reply, self.calls = set(rejects), reply, []
        self.usage = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage), completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        bad = self.rejects & set(kw)
        if bad:
            err = Exception(f"Unsupported parameter: {sorted(bad)[0]}")
            err.status_code = 400
            raise err
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))], usage=self.usage)


def test_request_judge_records_measured_usage_and_sends_only_set_params():
    s = judge_settings({**CFG, "reasoning_effort": "none"})
    c = _FakeClient()
    res = request_judge(c, request_body(s, "P"))
    assert res["score"] == 77 and res["error"] is None
    assert res["usage"] == {"prompt_tokens": 900, "completion_tokens": 180, "total_tokens": 1080, "reasoning_tokens": 0}
    assert set(c.calls[0]) == {"model", "messages", "max_tokens", "temperature", "reasoning_effort"}
    # a 4xx is not retried
    c2 = _FakeClient(rejects={"temperature"})
    res2 = request_judge(c2, request_body(s, "P"), retries=3)
    assert res2["score"] is None and "Unsupported parameter: temperature" in res2["error"] and len(c2.calls) == 1


def test_compat_check_names_rejected_settings():
    s = judge_settings({**CFG, "reasoning_effort": "low"})
    ok = compat_check(_FakeClient(), s, "P")
    assert len(ok) == 1 and ok[0]["accepted"] and ok[0]["usage"]["prompt_tokens"] == 900
    # a final-split prompt is sent only as the configured request: one call, no altered probes
    c = _FakeClient(rejects={"temperature"})
    rows = compat_check(c, s, "FINAL ANSWER", fallback=False)
    assert len(rows) == 1 and not rows[0]["accepted"] and len(c.calls) == 1
    rows = compat_check(_FakeClient(rejects={"temperature", "max_tokens"}), s, "P")
    by = {r["params"]: r for r in rows}
    assert not rows[0]["accepted"]
    assert not by["baseline: max_tokens only"]["accepted"]
    assert by["baseline: max_completion_tokens only"]["accepted"]
    assert not by["max_completion_tokens + temperature=0.0"]["accepted"]
    assert by["max_completion_tokens + reasoning_effort='low'"]["accepted"]


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
