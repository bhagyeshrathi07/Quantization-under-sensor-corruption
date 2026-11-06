"""Report producers (plan section 8): coverage per cell and harmful examples
for 05, and the pooled deployment profile for 07 --report. 09_tables.py
formats what these produce and computes nothing."""

from collections import defaultdict

from qflip.flips import Pair
from qflip.flips import coverage as coverage_rule
from qflip.provenance import COMPAT_FIELDS, PRECISION_FIELDS, compat_diff
from qflip.stats import summarize_latencies


def fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


class ProfileError(RuntimeError):
    pass


# ---------------------------------------------------------------- 05 producers


def coverage_by_cell(ref_records: list[dict], quant_records: list[dict], pairs: list[Pair], eligible_by_cell: dict[str, int] | None, min_n: int, min_scenes: int) -> dict[str, dict]:
    """Per cell of one condition: eligible questions in the population (None
    when the population is unknown), present without error on each side,
    errored on each side, paired, unpaired (present on at least one side but
    not paired: missing on the other side, or VQA without a usable score),
    scenes among the pairs, and the coverage rule's verdict."""
    cells = set(eligible_by_cell or {})
    cells |= {r["cell"] for r in ref_records + quant_records if r.get("cell")}
    cells |= {p.cell for p in pairs}
    out = {}
    for cell in sorted(cells):
        ref_ok = {r["question_id"] for r in ref_records if r.get("cell") == cell and not r.get("error")}
        quant_ok = {r["question_id"] for r in quant_records if r.get("cell") == cell and not r.get("error")}
        ref_err = sum(1 for r in ref_records if r.get("cell") == cell and r.get("error"))
        quant_err = sum(1 for r in quant_records if r.get("cell") == cell and r.get("error"))
        cell_pairs = [p for p in pairs if p.cell == cell]
        paired_ids = {p.question_id for p in cell_pairs}
        scenes = len({p.scene_token for p in cell_pairs})
        out[cell] = {
            "eligible": None if eligible_by_cell is None else eligible_by_cell.get(cell, 0),
            "present_ref": len(ref_ok),
            "present_quant": len(quant_ok),
            "errored_ref": ref_err,
            "errored_quant": quant_err,
            "paired": len(paired_ids),
            "unpaired": len((ref_ok | quant_ok) - paired_ids),
            "scenes": scenes,
            "assessed": coverage_rule({"n": len(paired_ids), "scenes": scenes}, min_n, min_scenes),
        }
    return out


def coverage_report(files: dict[str, tuple[list[dict], list[dict]]], pairs: list[Pair], eligible_by_cell: dict[str, int] | None, min_n: int, min_scenes: int) -> dict[str, dict]:
    """Coverage per condition: `files` maps each condition (from the validated
    file identity, never from question ids, which repeat across conditions) to
    its (reference records, quantized records); the pairs of that condition
    are the ones whose condition field equals it."""
    return {cond: coverage_by_cell(ref, quant, [p for p in pairs if p.condition == cond], eligible_by_cell, min_n, min_scenes) for cond, (ref, quant) in files.items()}


def condition_rows(rows: list[dict], conditions: list[str], min_n: int, min_scenes: int) -> list[dict]:
    """The per-condition headline rows (T2): the groups 05 computed from the
    pairs, plus an n = 0 unassessed row for every validated input condition
    that produced no pair, so a condition never disappears from the table."""
    have = {r["label"] for r in rows}
    extra = [{"label": c, "n": 0, "scenes": 0, "assessed": False, "reason": f"coverage: 0 questions from 0 scenes, below {min_n} / {min_scenes}"} for c in conditions if c not in have]
    return sorted(rows + extra, key=lambda r: r["label"])


def harmful_examples(pairs: list[Pair]) -> list[dict]:
    """Every harmful pair: id, condition, cell, views, BF16 margin, the two
    answers (option letters for MCQ, judge scores for VQA)."""
    rows = []
    for p in sorted((p for p in pairs if p.flip == "harmful"), key=lambda p: (p.condition, p.cell, p.question_id)):
        rows.append({
            "question_id": p.question_id,
            "scene_token": p.scene_token,
            "condition": p.condition,
            "cell": p.cell,
            "n_views": p.n_views,
            "margin": p.margin,
            "ref_answer": p.ref_answer,
            "quant_answer": p.quant_answer,
            "ref_score": p.ref_score,
            "quant_score": p.quant_score,
        })
    return rows


# ---------------------------------------------------------------- 07 pooling

HOST_FIELDS = ("hostname", "gpu_name")
# Workload a session must declare; pooled and compared sessions must agree on all of it.
WORKLOAD_FIELDS = ("split", "condition", "n", "views", "warmup")


def profile_summary(requests: list[dict]) -> dict:
    """Percentile summary of one list of requests; errored requests counted, not summarized."""
    ok = [r for r in requests if not r.get("error") and r.get("ttft_s") is not None]
    tok_s = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s") is not None]
    return {
        "requests": len(requests),
        "errors": len(requests) - len(ok),
        "ttft_s": summarize_latencies([r["ttft_s"] for r in ok]),
        "latency_s": summarize_latencies([r["latency_s"] for r in ok]),
        "decode_tok_s": summarize_latencies(tok_s),
        "prompt_tokens_mean": sum(r.get("prompt_tokens") or 0 for r in ok) / len(ok) if ok else None,
        "completion_tokens_mean": sum(r.get("completion_tokens") or 0 for r in ok) / len(ok) if ok else None,
    }


def session_row(s: dict) -> dict:
    sm = s["summary"]
    prov = s.get("provenance") or {}
    return {
        "file": s.get("file"),
        "session": s.get("session"),
        "model": prov.get("model_path_or_id") or s.get("model"),
        "checkpoint_fingerprint": prov.get("checkpoint_fingerprint"),
        "study": s.get("study"),
        "precision": s.get("precision"),
        "condition": s.get("condition"),
        "n": sm["requests"],
        "errors": sm["errors"],
        "gpu_peak_mib": s.get("gpu_peak_mib"),
        "ttft_s": {k: sm["ttft_s"].get(k) for k in ("p50", "p95", "p99")},
        "latency_s": {k: sm["latency_s"].get(k) for k in ("p50", "p95", "p99")},
        "decode_tok_s": {k: sm["decode_tok_s"].get(k) for k in ("p50", "p95")},
        "host": s.get("host"),
    }


def workload(s: dict) -> dict:
    """The fixed workload of a session: split, condition, slice size, view
    filter, warm-up count, and the exact question ids measured."""
    w = {k: s.get(k) for k in WORKLOAD_FIELDS}
    w["question_ids"] = sorted(r.get("question_id") for r in s.get("requests", []))
    return w


def _session_check(s: dict) -> None:
    name = s.get("file", "?")
    prov = s.get("provenance")
    if not prov:
        raise ProfileError(f"{name} has no provenance block (written before plan step 3); it cannot be pooled. Pass it with --reference to list it unpooled")
    # A local checkpoint is identified by its sha256 fingerprint and has no Hub
    # revision (the producer writes null); a Hub model must carry one.
    optional = ("model_revision",) if str(prov.get("checkpoint_fingerprint", "")).startswith("sha256:") else ()
    missing = [k for k in COMPAT_FIELDS if k not in optional and prov.get(k) in (None, "")]
    if missing:
        raise ProfileError(f"{name}: provenance incomplete, missing {missing}; it cannot be pooled")
    if s.get("study") is not True:
        why = "an exploratory session (no freeze check at measurement time)" if s.get("study") is False else "not marked as a study session (no evidence of the freeze check at measurement time)"
        raise ProfileError(f"{name} is {why}; it cannot be pooled. Pass it with --reference to list it unpooled")
    host = s.get("host") or {}
    if any(not host.get(k) for k in HOST_FIELDS):
        raise ProfileError(f"{name} has no complete host block ({HOST_FIELDS}); it cannot be pooled")
    missing = [k for k in WORKLOAD_FIELDS if s.get(k) is None]
    if missing or not s.get("requests"):
        raise ProfileError(f"{name}: workload incomplete, missing {missing or 'requests'}; it cannot be pooled")
    for k in ("precision", "session", "summary"):
        if s.get(k) is None:
            raise ProfileError(f"{name}: {k} missing; it cannot be pooled")


def _setup_diff(a: dict, b: dict, ignore: tuple[str, ...] = ()) -> list[str]:
    diff = compat_diff(a["provenance"], b["provenance"], ignore=ignore)
    diff += [f"host.{k}" for k in HOST_FIELDS if a["host"].get(k) != b["host"].get(k)]
    diff += [f"workload.{k}" for k, v in workload(a).items() if workload(b).get(k) != v]
    return diff


def pool_profiles(sessions: list[dict], reference: list[dict] = ()) -> dict:
    """Pooling rule (plan section 6): sessions pool only when they share the
    precision, every compatibility field of a complete provenance block
    (checkpoint fingerprint, serve settings, inputs), the host and the fixed
    workload (slice, views, warm-up, the exact question ids); duplicate
    session numbers are refused; exploratory sessions (measured without the
    freeze check) are refused; pooled percentiles are computed over the
    concatenated requests; per-session rows and the session count stay in the
    output. The two precisions are compared (memory reduction, budgets) only
    when they share every non-precision setting, host and workload.
    `reference` sessions (official-checkpoint measurements, files without a
    provenance block) are listed unpooled."""
    by_prec: dict[str, list[dict]] = defaultdict(list)
    for s in sessions:
        _session_check(s)
        by_prec[s["precision"]].append(s)
    out = {"precisions": {}, "reference": [session_row(s) for s in reference]}
    firsts = {}
    for prec, ss in sorted(by_prec.items()):
        ss = sorted(ss, key=lambda s: (s.get("session") or 0, s.get("file", "")))
        nums = [s.get("session") for s in ss]
        if None in nums or len(set(nums)) != len(nums):
            raise ProfileError(f"{prec}: session numbers must be present and distinct, got {nums}")
        first = ss[0]
        firsts[prec] = first
        for s in ss[1:]:
            diff = _setup_diff(first, s)
            if diff:
                raise ProfileError(f"{prec}: {s.get('file')} and {first.get('file')} come from different setups; differing fields: {diff}; sessions are pooled only across the same checkpoint, serve settings, host and workload")
        requests = [r for s in ss for r in s["requests"]]
        peaks = [s.get("gpu_peak_mib") for s in ss if s.get("gpu_peak_mib") is not None]
        out["precisions"][prec] = {
            "n_sessions": len(ss),
            "sessions": [session_row(s) for s in ss],
            "pooled": {**profile_summary(requests), "gpu_peak_mib_max": max(peaks) if peaks else None, "gpu_peak_mib_per_session": peaks},
            "provenance": {k: first["provenance"].get(k) for k in COMPAT_FIELDS},
            "host": first["host"],
            "workload": {k: v for k, v in workload(first).items() if k != "question_ids"} | {"n_question_ids": len(workload(first)["question_ids"])},
        }
    p = out["precisions"]
    reduction = None
    if {"bf16", "int4"} <= set(p):
        # the comparison across precisions needs the same KV cache, serving settings, inputs, host and workload
        diff = _setup_diff(firsts["bf16"], firsts["int4"], ignore=PRECISION_FIELDS)
        if diff:
            raise ProfileError(f"bf16 and int4 sessions are not comparable; differing fields: {diff}; memory reduction and budgets need equal serving settings, host and workload")
        b, i = p["bf16"]["pooled"]["gpu_peak_mib_max"], p["int4"]["pooled"]["gpu_peak_mib_max"]
        if b and i is not None:
            reduction = 1 - i / b
    out["memory_reduction"] = reduction

    def verdict(x, ok):
        return None if x is None else ok(x)

    out["budgets"] = {
        "ttft_p95_le_2s": {k: verdict(v["pooled"]["ttft_s"]["p95"], lambda x: x <= 2.0) for k, v in p.items()},
        "decode_p50_ge_15": {k: verdict(v["pooled"]["decode_tok_s"]["p50"], lambda x: x >= 15.0) for k, v in p.items()},
        "memory_reduction_ge_25pct": verdict(reduction, lambda x: x >= 0.25),
    }
    return out
