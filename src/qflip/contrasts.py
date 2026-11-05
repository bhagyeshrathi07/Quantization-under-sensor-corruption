"""Prespecified confirmatory contrasts on the MCQ harmful rate F_h
(WEEK9_ANALYSIS_PLAN.md section 4; PROTOCOL.md section 8).

Family of five, fixed before results from the remaining confirmatory population exist:
  C1 margin      BF16 margin in the lowest bin vs all other bins   (clean)
  C2 task        behavior/MCQ vs perception/MCQ                     (clean)
  C4a-c          NoImage / LowLight / Fog vs clean                  (common support)
Statistic Δ = F_h(A) − F_h(B). Scene-level paired bootstrap: scenes are drawn
with replacement from the union of both sides' scenes, and every pair of a
drawn scene enters on its own side. A draw is invalid when either side has no
reference-correct pair. Intervals: raw percentile 95% and Bonferroni 99%
(α = 0.05 / 5). The views comparison (1 vs 6 views) is reported under
`descriptive` only: in the final split every behavior MCQ has six views and
every perception MCQ has one, so Δ_views = −Δ_task in every draw.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from qflip.flips import Pair, margin_bin

FAMILY_SIZE = 5
STRESS_CONDITIONS = ("NoImage", "LowLight", "Fog")
VIEWS_NOTE = "descriptive only: in the final split every behavior MCQ has six views and every perception MCQ has one, so Δ_views = −Δ_task in every draw"


@dataclass(frozen=True)
class Contrast:
    name: str
    a_label: str
    b_label: str
    population: str
    select_a: Callable[[list[Pair]], list[Pair]]
    select_b: Callable[[list[Pair]], list[Pair]]


def side_stats(pairs: list[Pair]) -> dict:
    ref_ok = [p for p in pairs if p.ref_correct]
    harmful = sum(p.flip == "harmful" for p in pairs)
    return {
        "n": len(pairs),
        "scenes": len({p.scene_token for p in pairs}),
        "ref_correct": len(ref_ok),
        "ref_correct_scenes": len({p.scene_token for p in ref_ok}),
        "harmful": harmful,
        "harmful_rate": harmful / len(ref_ok) if ref_ok else None,
    }


def _harmful_rate(pairs: list[Pair]) -> float | None:
    ref_ok = sum(p.ref_correct for p in pairs)
    return sum(p.flip == "harmful" for p in pairs) / ref_ok if ref_ok else None


def percentile_interval(values: list[float], alpha: float) -> tuple[float, float] | None:
    if not values:
        return None
    vals = sorted(values)
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)]
    return (lo, hi)


def bootstrap_contrast(a: list[Pair], b: list[Pair], n_boot: int, seed: int = 0, family_size: int = FAMILY_SIZE) -> dict:
    """Δ = F_h(A) − F_h(B) with scene-level paired resampling. Returns the point
    estimate, both intervals (None when no draw was valid), and valid_draws."""
    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError(f"n_boot_contrast must be a positive integer, got {n_boot!r}")
    by_scene_a, by_scene_b = defaultdict(list), defaultdict(list)
    for p in a:
        by_scene_a[p.scene_token].append(p)
    for p in b:
        by_scene_b[p.scene_token].append(p)
    scenes = sorted(set(by_scene_a) | set(by_scene_b))
    fa, fb = _harmful_rate(a), _harmful_rate(b)
    delta = fa - fb if fa is not None and fb is not None else None
    rng = random.Random(seed)
    deltas = []
    if scenes:
        for _ in range(n_boot):
            drawn = rng.choices(scenes, k=len(scenes))
            sa = [p for s in drawn for p in by_scene_a.get(s, ())]
            sb = [p for s in drawn for p in by_scene_b.get(s, ())]
            ra, rb = _harmful_rate(sa), _harmful_rate(sb)
            if ra is None or rb is None:
                continue
            deltas.append(ra - rb)
    # Prespecified rule: intervals exist only when at least half the draws were
    # valid, whatever else the support check later reports first.
    available = len(deltas) >= 0.5 * n_boot
    ci95 = percentile_interval(deltas, 0.05) if available else None
    ci99 = percentile_interval(deltas, 0.05 / family_size) if available else None
    return {
        "delta": delta,
        "ci95": ci95,
        "ci99_bonferroni": ci99,
        "excludes_zero_raw": bool(ci95 and (ci95[0] > 0 or ci95[1] < 0)),
        "excludes_zero_adjusted": bool(ci99 and (ci99[0] > 0 or ci99[1] < 0)),
        "n_boot_contrast": n_boot,
        "valid_draws": len(deltas),
        "intervals_available": available,
    }


def support_check(a: dict, b: dict, valid_draws: int, n_boot: int, settings: dict) -> tuple[bool, str | None, str | None]:
    """The prespecified support rule: (assessed, reason, detail). reason is one
    of coverage, support, valid_draws; the first failed condition is named."""
    for label, s in (("a", a), ("b", b)):
        if s["n"] < settings["min_questions"] or s["scenes"] < settings["min_scenes"]:
            return False, "coverage", f"side {label}: {s['n']} questions from {s['scenes']} scenes (min {settings['min_questions']} / {settings['min_scenes']})"
    for label, s in (("a", a), ("b", b)):
        if s["ref_correct"] < settings["support_min_ref_correct"] or s["ref_correct_scenes"] < settings["support_min_ref_scenes"]:
            return False, "support", f"side {label}: {s['ref_correct']} reference-correct pairs from {s['ref_correct_scenes']} scenes (min {settings['support_min_ref_correct']} / {settings['support_min_ref_scenes']})"
    if valid_draws < 0.5 * n_boot:
        return False, "valid_draws", f"{valid_draws} of {n_boot} draws had reference-correct pairs on both sides (min 50%)"
    return True, None, None


def evaluate_contrast(c: Contrast, a: list[Pair], b: list[Pair], settings: dict, seed: int = 0, family_size: int = FAMILY_SIZE) -> dict:
    sa, sb = side_stats(a), side_stats(b)
    boot = bootstrap_contrast(a, b, settings["n_boot_contrast"], seed, family_size)
    assessed, reason, detail = support_check(sa, sb, boot["valid_draws"], settings["n_boot_contrast"], settings)
    return {
        "name": c.name,
        "a_label": c.a_label,
        "b_label": c.b_label,
        "population": c.population,
        "a": sa,
        "b": sb,
        **boot,
        "assessed": assessed,
        "reason": reason,
        "reason_detail": detail,
    }


def common_support(clean: list[Pair], cond: list[Pair], file_ids: dict[str, set[str]] | None = None) -> tuple[list[Pair], list[Pair], dict]:
    """C4: question ids present without error in all four input files (clean
    BF16, clean INT4, condition BF16, condition INT4). With file_ids (the
    non-error MCQ ids per file, keyed "clean/bf16", "<cond>/int4", ...) the
    dropped count is reported per file; without it, per side after pairing.
    Returns (condition pairs, clean pairs, dropped counts)."""
    if file_ids:
        ids = set.intersection(*file_ids.values()) if file_ids else set()
        dropped = {k: len(v) - len(ids) for k, v in file_ids.items()}
    else:
        ids = {p.question_id for p in clean} & {p.question_id for p in cond}
        dropped = {"clean": len(clean) - len(ids), "condition": len(cond) - len(ids)}
    a = [p for p in cond if p.question_id in ids]
    b = [p for p in clean if p.question_id in ids]
    return a, b, dropped


def family(margin_edges: tuple[float, ...]) -> list[Contrast]:
    low = margin_bin(0.0, margin_edges)
    return [
        Contrast("C1", f"margin {low}", "other margin bins", "clean, pairs with a margin",
                 lambda ps: [p for p in ps if p.margin is not None and margin_bin(p.margin, margin_edges) == low],
                 lambda ps: [p for p in ps if p.margin is not None and margin_bin(p.margin, margin_edges) != low]),
        Contrast("C2", "behavior/MCQ", "perception/MCQ", "clean",
                 lambda ps: [p for p in ps if p.cell == "behavior/MCQ"],
                 lambda ps: [p for p in ps if p.cell == "perception/MCQ"]),
    ]


def prespecified_contrasts(pairs_by_condition: dict[str, list[Pair]], settings: dict, seed: int = 0, file_ids: dict[str, set[str]] | None = None) -> dict:
    """Evaluate the family of five on MCQ pairs grouped by condition, plus the
    descriptive views row. The family always has five entries; a missing
    condition yields an unassessed entry with empty sides. file_ids, when
    given, maps "<condition>/<precision>" to the non-error MCQ ids of that
    input file and drives the C4 common-support audit."""
    edges = tuple(float(x) for x in settings["margin_edges"])
    clean = pairs_by_condition.get("clean", [])
    out = []
    for c in family(edges):
        out.append(evaluate_contrast(c, c.select_a(clean), c.select_b(clean), settings, seed))
    for i, cond in enumerate(STRESS_CONDITIONS):
        c = Contrast(f"C4{'abc'[i]}", cond, "clean", "common support (ids paired without error under both)", None, None)
        four = {k: v for k, v in (file_ids or {}).items() if k.split("/")[0] in ("clean", cond)}
        a, b, dropped = common_support(clean, pairs_by_condition.get(cond, []), four if len(four) == 4 else None)
        row = evaluate_contrast(c, a, b, settings, seed)
        row["support_dropped"] = dropped
        out.append(row)
    assert len(out) == FAMILY_SIZE
    views = Contrast("views", "1 view", "6 views", "clean",
                     lambda ps: [p for p in ps if p.n_views == 1],
                     lambda ps: [p for p in ps if p.n_views == 6])
    desc = evaluate_contrast(views, views.select_a(clean), views.select_b(clean), settings, seed)
    # no family correction and no decision for a descriptive row
    desc.update({"confirmatory": False, "note": VIEWS_NOTE, "ci99_bonferroni": None, "excludes_zero_adjusted": None})
    return {"family": out, "descriptive": [desc], "family_size": FAMILY_SIZE, "alpha_adjusted": 0.05 / FAMILY_SIZE, "seed": seed}


def format_contrast(r: dict) -> str:
    def ci(x):
        return f"[{x[0]:+.3f}, {x[1]:+.3f}]" if x else "n/a"

    d = f"{r['delta']:+.3f}" if r["delta"] is not None else "n/a"
    fa = f"{r['a']['harmful_rate']:.3f}" if r["a"]["harmful_rate"] is not None else "n/a"
    fb = f"{r['b']['harmful_rate']:.3f}" if r["b"]["harmful_rate"] is not None else "n/a"
    if r["assessed"]:
        status = "ASSESSED" + (" *" if r.get("excludes_zero_adjusted") else "") if r.get("confirmatory", True) else "DESCRIPTIVE (no decision)"
    else:
        status = f"UNASSESSED ({r['reason']}: {r['reason_detail']})"
    return (f"  {r['name']:5s} {r['a_label']} vs {r['b_label']}: F_h {fa} (n={r['a']['n']}, ref-correct {r['a']['ref_correct']}/{r['a']['ref_correct_scenes']} scenes) "
            f"vs {fb} (n={r['b']['n']}, ref-correct {r['b']['ref_correct']}/{r['b']['ref_correct_scenes']} scenes)  Δ={d} 95% {ci(r['ci95'])} 99%B {ci(r['ci99_bonferroni'])} "
            f"valid {r['valid_draws']}/{r['n_boot_contrast']}  {status}")
