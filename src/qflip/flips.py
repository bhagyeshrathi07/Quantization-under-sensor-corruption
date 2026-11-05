"""Pairing of reference (BF16) and quantized (INT4) MCQ records, flip
classification, and a paired bootstrap over scenes.

Scenes are the resampling unit because questions within a scene share frames
and are not independent. All statistics are computed on the strict MCQ rule
(option letter at the start of the answer); the released DriveBench rule is
kept in the records for comparability but misreads explanations.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

FLIP_CLASSES = ("harmful", "beneficial", "neutral_change", "unchanged")
PRECISIONS = ("bf16", "int4")


class PairingError(ValueError):
    """Reference and quantized records do not describe the same question under the same condition."""


def check_pair(a: dict, b: dict, question_text: dict[str, str] | None = None) -> None:
    """Raise PairingError naming the first field on which the two records of one
    question_id disagree. Records written before the schema carried `question`
    take it from question_text (annotation lookup)."""
    qid = a["question_id"]
    if (a.get("precision"), b.get("precision")) != PRECISIONS:
        raise PairingError(f"{qid}: records must come as (reference=bf16, quantized=int4), got ({a.get('precision')!r}, {b.get('precision')!r}); a reversed order would swap harmful and beneficial")
    qa = a.get("question") or (question_text or {}).get(qid)
    qb = b.get("question") or (question_text or {}).get(qid)
    checks = [
        ("condition", a.get("condition"), b.get("condition")),
        ("gt", a.get("mcq_gt", a.get("gt")), b.get("mcq_gt", b.get("gt"))),
        ("question", qa, qb),
        ("scene_token", a.get("scene_token"), b.get("scene_token")),
        ("frame_token", a.get("frame_token"), b.get("frame_token")),
        ("cell", a.get("cell"), b.get("cell")),
        ("views", list(a.get("views") or []), list(b.get("views") or [])),
    ]
    for field, x, y in checks:
        if x is None and y is None and field == "question":
            raise PairingError(f"{qid}: question text missing on both sides and no annotation lookup given")
        if x != y:
            raise PairingError(f"{qid}: {field} differs between records: {x!r} vs {y!r}")


def classify(ref: dict, quant: dict) -> str:
    if ref["mcq_correct"] and not quant["mcq_correct"]:
        return "harmful"
    if not ref["mcq_correct"] and quant["mcq_correct"]:
        return "beneficial"
    if ref["mcq_pred_strict"] != quant["mcq_pred_strict"]:
        return "neutral_change"
    return "unchanged"


@dataclass
class Pair:
    question_id: str
    scene_token: str
    cell: str
    n_views: int
    condition: str
    margin: float | None
    ref_correct: bool
    quant_correct: bool
    flip: str
    ref_score: float | None = None    # VQA only: judge rubric score
    quant_score: float | None = None


def pair_records(ref_records: list[dict], quant_records: list[dict], question_text: dict[str, str] | None = None) -> list[Pair]:
    """Match MCQ records by question_id; records with errors on either side are
    dropped. Every matched pair is validated with check_pair (PairingError)."""
    ref = {r["question_id"]: r for r in ref_records if (r.get("cell") or "").endswith("MCQ") and not r.get("error")}
    qnt = {r["question_id"]: r for r in quant_records if (r.get("cell") or "").endswith("MCQ") and not r.get("error")}
    pairs = []
    for qid in sorted(set(ref) & set(qnt)):
        a, b = ref[qid], qnt[qid]
        check_pair(a, b, question_text)
        pairs.append(
            Pair(
                question_id=qid,
                scene_token=a["scene_token"],
                cell=a["cell"],
                n_views=a["n_views"],
                condition=a.get("condition", "?"),
                margin=a.get("margin"),
                ref_correct=bool(a["mcq_correct"]),
                quant_correct=bool(b["mcq_correct"]),
                flip=classify(a, b),
            )
        )
    return pairs


def pair_vqa_records(
    ref_records: list[dict],
    quant_records: list[dict],
    ref_scores: dict[str, dict],
    quant_scores: dict[str, dict],
    threshold: float,
    question_text: dict[str, str] | None = None,
) -> list[Pair]:
    """VQA pairing from judge sidecars (qflip.judge.load_scores). Correct =
    rubric score >= threshold. Only harmful / beneficial / unchanged apply:
    free-text answers almost always differ in wording, so "neutral_change" is
    not defined and change_rate for VQA means a correctness change. Questions
    without a score on either side are dropped; margin is not available."""
    ref = {r["question_id"]: r for r in ref_records if (r.get("cell") or "").endswith("VQA") and not r.get("error")}
    qnt = {r["question_id"]: r for r in quant_records if (r.get("cell") or "").endswith("VQA") and not r.get("error")}
    pairs = []
    for qid in sorted(set(ref) & set(qnt)):
        rs, qs = ref_scores.get(qid, {}).get("score"), quant_scores.get(qid, {}).get("score")
        if rs is None or qs is None:
            continue
        a = ref[qid]
        check_pair(a, qnt[qid], question_text)
        rc, qc = rs >= threshold, qs >= threshold
        flip = "harmful" if rc and not qc else "beneficial" if qc and not rc else "unchanged"
        pairs.append(
            Pair(
                question_id=qid,
                scene_token=a["scene_token"],
                cell=a["cell"],
                n_views=a["n_views"],
                condition=a.get("condition", "?"),
                margin=None,
                ref_correct=rc,
                quant_correct=qc,
                flip=flip,
                ref_score=rs,
                quant_score=qs,
            )
        )
    return pairs


def merge_pairs(groups: list[list[Pair]]) -> list[Pair]:
    """Concatenate pair lists from several file pairs, refusing any
    (condition, question_id) that appears twice: duplicated inputs would
    manufacture support for the contrasts."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for i, g in enumerate(groups):
        for p in g:
            key = (p.condition, p.question_id)
            if key in seen:
                raise PairingError(f"{p.question_id} under {p.condition} appears in file pair {seen[key] + 1} and again in file pair {i + 1}; duplicated inputs are refused")
            seen[key] = i
            out.append(p)
    return out


def margin_bin(margin: float | None, edges: tuple[float, ...]) -> str:
    """Label like '[0.25,0.5)'; None -> 'no_margin'."""
    if margin is None:
        return "no_margin"
    lo = 0.0
    for e in edges:
        if margin < e:
            return f"[{lo:g},{e:g})"
        lo = e
    return f"[{lo:g},1]"


def summarize(pairs: list[Pair]) -> dict:
    """Point estimates for one set of pairs."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    counts = Counter(p.flip for p in pairs)
    ref_correct = sum(p.ref_correct for p in pairs)
    quant_correct = sum(p.quant_correct for p in pairs)
    return {
        "n": n,
        "scenes": len({p.scene_token for p in pairs}),
        "ref_acc": ref_correct / n,
        "quant_acc": quant_correct / n,
        "acc_delta": (quant_correct - ref_correct) / n,
        "harmful": counts["harmful"],
        "beneficial": counts["beneficial"],
        "neutral_change": counts["neutral_change"],
        "unchanged": counts["unchanged"],
        # F_h: fraction of reference-correct answers the quantized model loses
        "harmful_rate": counts["harmful"] / ref_correct if ref_correct else None,
        # any answer change, right or wrong (VQA pairs: any correctness change)
        "change_rate": (n - counts["unchanged"]) / n,
        # VQA only: mean rubric score per side, None for MCQ pairs
        "ref_score_mean": _mean([p.ref_score for p in pairs if p.ref_score is not None]),
        "quant_score_mean": _mean([p.quant_score for p in pairs if p.quant_score is not None]),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def coverage(summary: dict, min_questions: int, min_scenes: int) -> bool:
    """PROTOCOL section 5: a group is assessed only with enough questions and scenes."""
    return summary.get("n", 0) >= min_questions and summary.get("scenes", 0) >= min_scenes


def bootstrap_scenes(pairs: list[Pair], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CIs from resampling scenes with replacement. Returns
    {stat: (lo, hi)} for harmful_rate, acc_delta, change_rate."""
    by_scene = defaultdict(list)
    for p in pairs:
        by_scene[p.scene_token].append(p)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return {}
    rng = random.Random(seed)
    stats = {"harmful_rate": [], "acc_delta": [], "change_rate": []}
    for _ in range(n_boot):
        sample = [p for s in rng.choices(scenes, k=len(scenes)) for p in by_scene[s]]
        s = summarize(sample)
        for k in stats:
            if s.get(k) is not None:
                stats[k].append(s[k])
    out = {}
    for k, vals in stats.items():
        if not vals:
            continue
        vals.sort()
        lo = vals[int((alpha / 2) * len(vals))]
        hi = vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)]
        out[k] = (lo, hi)
    return out


def group_by(pairs: list[Pair], factor: str, margin_edges: tuple[float, ...] = (0.25, 0.5)) -> dict[str, list[Pair]]:
    key = {
        "cell": lambda p: p.cell,
        "n_views": lambda p: str(p.n_views),
        "margin": lambda p: margin_bin(p.margin, margin_edges),
        "condition": lambda p: p.condition,
    }[factor]
    groups = defaultdict(list)
    for p in pairs:
        groups[key(p)].append(p)
    return dict(sorted(groups.items()))
