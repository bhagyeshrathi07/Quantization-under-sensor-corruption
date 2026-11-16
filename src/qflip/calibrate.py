"""Judge threshold calibration against human labels, by the rule fixed before
any score met a label (manifests/judge_decisions_2026-11-13.md, decision 4).

Pure functions over (score, label) items; the file handling is in
scripts/10_build_label_sheet.py (the blind sheet and its key) and
scripts/11_calibrate_judge.py (joining sheet, key and judge sidecars).

Rule: candidates are every integer t in 0..100; judge-correct = score >= t;
agreement(t) = share of items where judge-correct equals the human label;
chosen t = highest agreement, lowest t on a tie. Acceptance is evaluated apart
from the fit by leave-one-scene-out cross-validation: for each scene the
threshold is fitted on the other scenes' items and applied to the held-out
items; the pooled held-out predictions must reach the bars (agreement, Cohen's
kappa, recall on human-correct and on human-incorrect items), and the
calibration items must hold at least `min_per_class` items of each human
class. When acceptance passes, the threshold frozen is the one fitted on all
calibration items. The bars are the team's choice (config `judge.acceptance`),
written down so they cannot be chosen after seeing the curves.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from itertools import combinations

THRESHOLDS = tuple(range(0, 101))

# Decision 4 bars; config judge.acceptance overrides nothing at run time
# unless it names every key (the script refuses a partial block).
ACCEPTANCE_KEYS = ("agreement", "kappa", "recall", "min_per_class", "min_items")
DEFAULT_ACCEPTANCE = {"agreement": 0.80, "kappa": 0.40, "recall": 0.60, "min_per_class": 15, "min_items": 100}


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    """One sheet item joined with its judge aggregate: `score` is the production
    aggregate (mean of the frozen number of judgings), `label` the consensus
    label (1 correct, 0 incorrect), `majority` the pre-discussion majority
    label, `score_rep0` replicate 0 alone (both for the sensitivity rows)."""

    item: str
    scene: str
    score: float
    label: int
    majority: int | None = None
    score_rep0: float | None = None


def acceptance_bars(jcfg: dict | None) -> dict:
    """The acceptance bars from config `judge.acceptance`, or the decision-4
    defaults when the block is absent. A block that names only some keys is
    refused: the bars are a unit, fixed in advance."""
    block = (jcfg or {}).get("acceptance")
    if block is None:
        return dict(DEFAULT_ACCEPTANCE)
    if not isinstance(block, dict) or set(block) != set(ACCEPTANCE_KEYS):
        raise CalibrationError(f"judge.acceptance must name exactly {list(ACCEPTANCE_KEYS)}, got {sorted(block) if isinstance(block, dict) else block!r}")
    out = {}
    for k in ("agreement", "kappa", "recall"):
        v = float(block[k])
        if not 0.0 <= v <= 1.0:
            raise CalibrationError(f"judge.acceptance.{k} must lie in [0, 1], got {v}")
        out[k] = v
    for k in ("min_per_class", "min_items"):
        v = block[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise CalibrationError(f"judge.acceptance.{k} must be a positive integer, got {v!r}")
        out[k] = v
    return out


def _check_pairs(pairs) -> list[tuple[float, int]]:
    out = []
    for score, label in pairs:
        if label not in (0, 1):
            raise CalibrationError(f"label must be 0 or 1, got {label!r}")
        s = float(score)
        if not 0.0 <= s <= 100.0:
            raise CalibrationError(f"score must lie in [0, 100], got {score!r}")
        out.append((s, label))
    return out


def predict(score: float, t: int) -> int:
    return 1 if score >= t else 0


def agreement_curve(pairs) -> list[float]:
    """agreement[t] for t in THRESHOLDS; every item weighted equally."""
    pairs = _check_pairs(pairs)
    if not pairs:
        raise CalibrationError("no items to fit")
    n = len(pairs)
    return [sum(1 for s, y in pairs if predict(s, t) == y) / n for t in THRESHOLDS]


def choose_threshold(curve: list[float]) -> int:
    """The lowest t with the highest agreement."""
    if len(curve) != len(THRESHOLDS):
        raise CalibrationError(f"curve must have {len(THRESHOLDS)} points, got {len(curve)}")
    best = max(curve)
    return next(t for t, a in zip(THRESHOLDS, curve) if a == best)


def confusion(pairs, t: int) -> dict:
    """Positive = correct. tp: both correct; fp: judge correct, human not;
    fn: human correct, judge not; tn: both incorrect."""
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for s, y in _check_pairs(pairs):
        p = predict(s, t)
        c["tp" if p and y else "fp" if p else "fn" if y else "tn"] += 1
    return c


def exact_rates(c: dict) -> dict[str, Fraction | None]:
    """Agreement, Cohen's kappa and the two recalls of a confusion table as
    exact fractions of the integer counts (review of November 14: a kappa of
    exactly 0.40 came out as 0.3999999999999999 in floating point and failed
    the bar). kappa is None when chance agreement is 1 (both sides give one
    class only); a recall is None when its class is empty."""
    tp, fp, fn, tn = (int(c[k]) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    if n == 0:
        return {"agreement": None, "kappa": None, "recall_correct": None, "recall_incorrect": None}
    observed = Fraction(tp + tn, n)
    chance = Fraction((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    kappa = None if chance == 1 else (observed - chance) / (1 - chance)
    return {
        "agreement": observed,
        "kappa": kappa,
        "recall_correct": Fraction(tp, tp + fn) if tp + fn else None,
        "recall_incorrect": Fraction(tn, fp + tn) if fp + tn else None,
    }


def _float(x):
    return None if x is None else float(x)


def cohen_kappa(c: dict) -> float | None:
    """Cohen's kappa of judge against human (float of the exact value); None
    when undefined."""
    return _float(exact_rates(c)["kappa"])


def metrics(pairs, t: int) -> dict:
    """Agreement, kappa, the two recalls and the two error counts at t."""
    return _metrics_of(confusion(pairs, t), int(t))


def _metrics_of(c: dict, t: int | None) -> dict:
    rates = exact_rates(c)
    return {
        "threshold": t,
        "n": sum(c.values()),
        "n_human_correct": c["tp"] + c["fn"],
        "n_human_incorrect": c["fp"] + c["tn"],
        "agreement": _float(rates["agreement"]),
        "kappa": _float(rates["kappa"]),
        "recall_correct": _float(rates["recall_correct"]),
        "recall_incorrect": _float(rates["recall_incorrect"]),
        "judge_correct_human_incorrect": c["fp"],
        "human_correct_judge_incorrect": c["fn"],
        "confusion": dict(c),
    }


def fit(pairs) -> dict:
    """The threshold rule on all given items, with the full curve."""
    curve = agreement_curve(pairs)
    t = choose_threshold(curve)
    return {"curve": curve, **metrics(pairs, t)}


def pooled_heldout(items: list[Item], score_of=lambda it: it.score, label_of=lambda it: it.label) -> dict:
    """Leave-one-scene-out: fit on the other scenes, predict the held-out
    scene, pool the predictions. Returns the per-fold thresholds and the pooled
    confusion with its metrics (the threshold key of the pooled metrics is
    None: no single t produced them)."""
    scenes = sorted({it.scene for it in items})
    if len(scenes) < 2:
        raise CalibrationError(f"leave-one-scene-out needs at least 2 scenes, got {len(scenes)}")
    folds, pooled = [], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for scene in scenes:
        train = [(score_of(it), label_of(it)) for it in items if it.scene != scene]
        held = [(score_of(it), label_of(it)) for it in items if it.scene == scene]
        t = choose_threshold(agreement_curve(train))
        c = confusion(held, t)
        for k in pooled:
            pooled[k] += c[k]
        folds.append({"scene": scene, "n_train": len(train), "n_held": len(held), "threshold": t, "confusion": c})
    return {"folds": folds, "n_scenes": len(scenes), "pooled": _metrics_of(pooled, None)}


def exact_bar(bar) -> Fraction:
    """A bar as written in config (0.40) as the exact rational 2/5, so that a
    metric equal to it passes."""
    return Fraction(str(bar)) if isinstance(bar, float) else Fraction(bar)


def acceptance(pooled: dict, n_correct: int, n_incorrect: int, bars: dict) -> tuple[bool, list[str]]:
    """Every bar must hold on the pooled held-out predictions, compared as
    exact fractions of the pooled confusion counts; the class counts are those
    of the calibration items. A metric that is undefined (None) fails its bar."""
    reasons = []
    if n_correct < bars["min_per_class"]:
        reasons.append(f"only {n_correct} human-correct items, {bars['min_per_class']} required")
    if n_incorrect < bars["min_per_class"]:
        reasons.append(f"only {n_incorrect} human-incorrect items, {bars['min_per_class']} required")
    rates = exact_rates(pooled["confusion"])
    for key, bar, name in (("agreement", bars["agreement"], "held-out agreement"), ("kappa", bars["kappa"], "held-out Cohen's kappa"),
                           ("recall_correct", bars["recall"], "held-out recall on human-correct items"), ("recall_incorrect", bars["recall"], "held-out recall on human-incorrect items")):
        v = rates[key]
        if v is None:
            reasons.append(f"{name} undefined, {float(bar):.2f} required")
        elif v < exact_bar(bar):
            reasons.append(f"{name} {float(v):.3f} below {float(bar):.2f}")
    return (not reasons), reasons


def calibrate(items: list[Item], bars: dict | None = None) -> dict:
    """The whole of decision 4 on the calibration items: the fit on all items,
    the leave-one-scene-out acceptance, the verdict with its reasons, and the
    sensitivity rows (the same rule on the pre-discussion majority label; the
    same rule on replicate 0 alone), which never enter the verdict."""
    bars = dict(bars or DEFAULT_ACCEPTANCE)
    if len(items) < bars["min_items"]:
        return {"n_items": len(items), "bars": bars, "calibrated": False, "accepted": False, "frozen_threshold": None,
                "reasons": [f"only {len(items)} calibration items, {bars['min_items']} required (plan section 7); no calibration"]}
    ids = [it.item for it in items]
    if len(set(ids)) != len(ids):
        raise CalibrationError("duplicate item ids among the calibration items")
    pairs = [(it.score, it.label) for it in items]
    full = fit(pairs)
    loso = pooled_heldout(items)
    ok, reasons = acceptance(loso["pooled"], full["n_human_correct"], full["n_human_incorrect"], bars)
    out = {
        "n_items": len(items),
        "n_scenes": loso["n_scenes"],
        "bars": bars,
        "calibrated": True,
        "fit": full,
        "loso": loso,
        "accepted": ok,
        "reasons": reasons,
        "frozen_threshold": full["threshold"] if ok else None,
        "sensitivity": {},
    }
    if all(it.majority is not None for it in items):
        maj_items = [Item(it.item, it.scene, it.score, it.majority) for it in items]
        maj_fit = fit([(it.score, it.label) for it in maj_items])
        maj_loso = pooled_heldout(maj_items)
        maj_ok, maj_reasons = acceptance(maj_loso["pooled"], maj_fit["n_human_correct"], maj_fit["n_human_incorrect"], bars)
        out["sensitivity"]["majority_label"] = {"fit": {k: v for k, v in maj_fit.items() if k != "curve"}, "heldout": maj_loso["pooled"], "accepted": maj_ok, "reasons": maj_reasons,
                                                "items_changed_by_discussion": sum(1 for it in items if it.majority != it.label)}
    if all(it.score_rep0 is not None for it in items):
        r0_items = [Item(it.item, it.scene, it.score_rep0, it.label) for it in items]
        r0_fit = fit([(it.score, it.label) for it in r0_items])
        r0_loso = pooled_heldout(r0_items)
        r0_ok, r0_reasons = acceptance(r0_loso["pooled"], r0_fit["n_human_correct"], r0_fit["n_human_incorrect"], bars)
        out["sensitivity"]["replicate_0_only"] = {"fit": {k: v for k, v in r0_fit.items() if k != "curve"}, "heldout": r0_loso["pooled"], "accepted": r0_ok, "reasons": r0_reasons}
    return out


# ---- inter-labeller statistics, reported before any discussion ----


def majority_label(labels: list[int]) -> int:
    """Majority of an odd number of 0/1 labels."""
    if len(labels) % 2 == 0:
        raise CalibrationError(f"majority needs an odd number of labels, got {len(labels)}")
    return 1 if sum(labels) * 2 > len(labels) else 0


def pairwise_agreement(labels_by_labeller: dict[str, list[int]]) -> dict[str, float]:
    """Share of items on which each pair of labellers gave the same label."""
    names = sorted(labels_by_labeller)
    n = {len(v) for v in labels_by_labeller.values()}
    if len(n) != 1:
        raise CalibrationError("every labeller must have labelled the same items")
    n = n.pop()
    if n == 0:
        raise CalibrationError("no labelled items")
    out = {}
    for a, b in combinations(names, 2):
        same = sum(1 for x, y in zip(labels_by_labeller[a], labels_by_labeller[b]) if x == y)
        out[f"{a}/{b}"] = same / n
    return out


def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """Fleiss' kappa for binary labels; rows are items, each a list of the
    labellers' 0/1 labels (the same count on every row). None when every
    labeller gave the same label on every item (chance agreement 1)."""
    if not rows:
        raise CalibrationError("no labelled items")
    k = len(rows[0])
    if k < 2 or any(len(r) != k for r in rows):
        raise CalibrationError("every item needs the same number (>= 2) of labels")
    n = len(rows)
    p_i = []
    ones_total = 0
    for r in rows:
        if any(x not in (0, 1) for x in r):
            raise CalibrationError(f"labels must be 0 or 1, got {r}")
        ones = sum(1 for x in r if x == 1)
        zeros = k - ones
        ones_total += ones
        p_i.append(Fraction(ones * (ones - 1) + zeros * (zeros - 1), k * (k - 1)))
    p_bar = sum(p_i) / n
    p1 = Fraction(ones_total, n * k)
    pe = p1 * p1 + (1 - p1) * (1 - p1)
    if pe >= 1:
        return None
    return float((p_bar - pe) / (1 - pe))


def item_dict(it: Item) -> dict:
    return asdict(it)


# ---- the blind sheet: what the three scripts share ----

BLIND_COLUMNS = ("item", "cell", "question", "ground_truth", "answer")
FORBIDDEN_SHEET_COLUMNS = ("precision", "question_id", "scene_token", "score", "judge", "model", "reply")


def label_column(name: str) -> str:
    return f"label_{name}"


def sheet_columns(labellers: list[str]) -> list[str]:
    return [*BLIND_COLUMNS, *(label_column(n) for n in labellers), "consensus", "consensus_reason"]


def display_sha256(cell: str, question: str, ground_truth: str, answer: str) -> str:
    """Digest of exactly what a labeller sees for an item. The key stores it
    at build time; the calibration recomputes it from the labelled sheet row
    and from the current answer record, so a label is bound to the text it was
    given for (review of November 14: the digest of the raw answer alone did
    not bind the displayed question, ground truth or answer)."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps([cell, question, ground_truth, answer], ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_label(raw: str | None, where: str) -> int | None:
    s = (raw or "").strip()
    if s == "":
        return None
    if s in ("0", "1"):
        return int(s)
    raise CalibrationError(f"{where}: label {raw!r} is not 0, 1 or empty")


def classify_sheet(rows: list[dict], labellers: list[str], need_consensus: bool) -> tuple[dict, list[str], dict[str, int]]:
    """Which items are complete and which are the unfinished tail, under the
    labelling rule (items are labelled in sheet order, each labeller
    independently, and if time runs out the calibration uses the items every
    labeller finished). Each labeller's labelled items must be the sheet's
    first rows (a labelled row after a blank one is refused: the sheet was not
    labelled by the rule). The complete items are the common prefix: the rows
    every labeller has reached. The tail (rows some labellers did not reach)
    is unfinished; labels on it are kept and counted, never used. With
    `need_consensus` (the calibration) every complete item must carry a
    consensus and no tail item may (a consensus needs all three labels).
    Without it (the pre-discussion report) the consensus columns are not read.
    Returns ({item: {labels, consensus, reason}}, [unfinished items in order],
    {labeller: rows labelled})."""
    parsed, seen = [], set()
    for row in rows:
        item = (row.get("item") or "").strip()
        if not item:
            raise CalibrationError("a sheet row has no item id")
        if item in seen:
            raise CalibrationError(f"item {item} appears twice on the sheet")
        seen.add(item)
        labels = {n: parse_label(row.get(label_column(n)), f"{item} {n}") for n in labellers}
        parsed.append((item, labels, row))
    progress = {}
    for n in labellers:
        done = [lab[n] is not None for _, lab, _ in parsed]
        k = sum(done)
        if done[:k] != [True] * k:
            first_blank = next(i for i, d in enumerate(done) if not d)
            later = next(parsed[i][0] for i in range(first_blank, len(done)) if done[i])
            raise CalibrationError(f"{n} labelled {later} but not {parsed[first_blank][0]} before it: items are labelled in sheet order, so a labeller's labels must be the sheet's first rows")
        progress[n] = k
    reach = min(progress.values()) if progress else 0
    complete, unfinished = {}, []
    for i, (item, labels, row) in enumerate(parsed):
        cons, reason = None, ""
        if need_consensus:
            cons = parse_label(row.get("consensus"), f"{item} consensus")
            reason = (row.get("consensus_reason") or "").strip()
            if i < reach and cons is None:
                raise CalibrationError(f"item {item} has every label but no consensus; the key stays closed until the consensus covers every item all labellers finished")
            if i >= reach and cons is not None:
                raise CalibrationError(f"item {item} has a consensus but not every labeller reached it; a consensus is given only on items all labellers finished")
        if i < reach:
            complete[item] = {"labels": labels, "consensus": cons, "reason": reason}
        else:
            unfinished.append(item)
    return complete, unfinished, progress
