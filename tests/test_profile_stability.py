import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

from qflip.client import consume_stream
from qflip.stats import percentile, summarize_latencies


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_percentiles():
    vals = [float(i) for i in range(1, 101)]
    assert percentile(vals, 50) == 50.0
    assert percentile(vals, 95) == 95.0
    assert percentile(vals, 99) == 99.0
    assert percentile([], 50) is None
    assert summarize_latencies([2.0, 1.0])["p50"] == 1.0


def _chunk(content=None, usage=None):
    choice = SimpleNamespace(delta=SimpleNamespace(content=content)) if content is not None else None
    return SimpleNamespace(choices=[choice] if choice else [], usage=usage)


def test_consume_stream_ttft_and_decode_rate():
    def chunks():
        time.sleep(0.05)
        yield _chunk("")            # empty delta must not count as first token
        yield _chunk("B")
        time.sleep(0.05)
        yield _chunk(". Turn")
        yield _chunk(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=3))

    c = consume_stream(chunks(), time.perf_counter())
    assert c.text == "B. Turn"
    assert c.ttft_s is not None and 0.04 < c.ttft_s < c.latency_s
    assert c.completion_tokens == 3
    assert c.decode_tok_s is not None and c.decode_tok_s > 0


def test_stability_summary_counts_self_disagreement():
    mod = _load("06_stability.py")

    def r(qid, letter, correct, margin, pred=None):
        return {"question_id": qid, "cell": "behavior/MCQ", "mcq_pred_strict": letter, "mcq_correct": correct,
                "margin": margin, "pred": pred or letter, "error": None}

    rep1 = [r("1", "A", True, 0.5), r("2", "B", False, 0.2), r("3", "C", True, 0.9)]
    rep2 = [r("1", "A", True, 0.4), r("2", "C", False, 0.2), r("3", "C", True, 0.9, pred="C. yes")]
    rep3 = [r("1", "B", False, 0.3), r("2", "B", False, 0.2), r("3", "C", True, 0.9)]
    s = mod.stability_summary([rep1, rep2, rep3])
    assert s["mcq_common"] == 3 and s["repeats"] == 3
    # pairs: (1,2) letters 1, (1,3) letters 1, (2,3) letters 2 -> 4 / (3 pairs * 3 q)
    assert abs(s["self_change_rate"] - 4 / 9) < 1e-9
    assert abs(s["self_correctness_change_rate"] - 2 / 9) < 1e-9
    assert s["margin_range_max"] == 0.2


def test_profile_summary_excludes_errors():
    mod = _load("07_profile.py")
    reqs = [
        {"error": None, "ttft_s": 0.5, "latency_s": 2.0, "decode_tok_s": 40.0, "prompt_tokens": 1000, "completion_tokens": 60},
        {"error": None, "ttft_s": 0.7, "latency_s": 2.4, "decode_tok_s": 35.0, "prompt_tokens": 1000, "completion_tokens": 60},
        {"error": "boom", "ttft_s": None, "latency_s": 9.0, "decode_tok_s": None, "prompt_tokens": None, "completion_tokens": None},
    ]
    s = mod.profile_summary(reqs)
    assert s["requests"] == 3 and s["errors"] == 1
    assert s["ttft_s"]["p95"] == 0.7 and s["decode_tok_s"]["p50"] == 35.0
