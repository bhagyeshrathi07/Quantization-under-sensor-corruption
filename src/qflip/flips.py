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


def pair_records(ref_records: list[dict], quant_records: list[dict]) -> list[Pair]:
    """Match MCQ records by question_id; records with errors on either side are dropped."""
    ref = {r["question_id"]: r for r in ref_records if (r.get("cell") or "").endswith("MCQ") and not r.get("error")}
    qnt = {r["question_id"]: r for r in quant_records if (r.get("cell") or "").endswith("MCQ") and not r.get("error")}
    pairs = []
    for qid in sorted(set(ref) & set(qnt)):
        a, b = ref[qid], qnt[qid]
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
        # any answer change, right or wrong
        "change_rate": (n - counts["unchanged"]) / n,
    }


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
