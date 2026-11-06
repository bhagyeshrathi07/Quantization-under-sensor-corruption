"""Markdown tables T1-T7 for the report (plan section 8), formatted from the
producers' JSON: 05 --out (coverage, headline, contrasts, breakdowns, harmful
examples), 07 --report --out (pooled deployment profile), 06 summaries
(stability). Formatting only: no number here is computed from raw records."""


def _f(x, nd=3, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:.{nd}f}"


def _ci(ci, nd=3, signed=False):
    if not ci:
        return "n/a"
    fmt = f"{{:+.{nd}f}}" if signed else f"{{:.{nd}f}}"
    return f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def _table(header: list[str], rows: list[list], align: str | None = None) -> str:
    cols = len(header)
    sep = ["---"] * cols if align is None else [("---:" if a == "r" else ":---" if a == "l" else "---") for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        cells = [str(c) if c is not None else "n/a" for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _unassessed(row: dict) -> str:
    reason = row.get("reason") or "coverage"
    return f"UNASSESSED ({reason})"


# ---------------------------------------------------------------- T1


def t1_coverage(compare: dict) -> str:
    cov = compare.get("coverage") or {}
    rows = []
    for cond, cells in cov.items():
        for cell, c in cells.items():
            rows.append([cond, cell, c.get("eligible"), c["present_ref"], c["present_quant"], f"{c['errored_ref']} / {c['errored_quant']}", c["paired"], c["unpaired"], c["scenes"], "yes" if c["assessed"] else "no"])
    body = _table(["Condition", "Cell", "Eligible", "Present BF16", "Present INT4", "Errored BF16 / INT4", "Paired", "Unpaired", "Scenes", "Assessed"], rows) if rows else "_no coverage block in the comparison JSON_"
    return f"### T1. Coverage per cell ({compare.get('analysis_class', 'unknown')} analysis)\n\n{body}\n\nEligible = questions of the cell in the declared population (n/a when the population is not declared). Assessed = at least {compare.get('min_questions', '?')} paired questions from {compare.get('min_scenes', '?')} scenes.\n"


# ---------------------------------------------------------------- T2 / T4


def _group_row(label: str, g: dict) -> list:
    if not g.get("assessed"):
        return [label, g.get("n", 0), g.get("scenes", 0), _unassessed(g), "", "", "", "", "", ""]
    ci = g.get("ci") or {}
    return [
        label, g["n"], g["scenes"],
        f"{_f(g['ref_acc'])} / {_f(g['quant_acc'])}",
        f"{g['acc_delta']:+.3f} {_ci(ci.get('acc_delta'), signed=True)}",
        g["harmful"], g["beneficial"], g["neutral_change"],
        f"{_f(g['harmful_rate'])} {_ci(ci.get('harmful_rate'))}",
        f"{_f(g['change_rate'])} {_ci(ci.get('change_rate'))}",
    ]


_GROUP_HEADER = ["Group", "n", "Scenes", "Acc BF16 / INT4", "Δacc [95%]", "Harmful", "Beneficial", "Neutral", "F_h [95%]", "Change [95%]"]


def t2_headline(compare: dict) -> str:
    # Per-condition groups are written by 05 unconditionally; their absence is
    # reported, never read as "one condition".
    mcq = compare.get("mcq") or {}
    by_cond = (mcq.get("groups") or {}).get("condition")
    rows = [_group_row(g["label"], g) for g in by_cond] if by_cond else []
    vqa = compare.get("vqa") or {}
    vby = (vqa.get("groups") or {}).get("condition") if vqa else None
    vrows = [_group_row(g["label"], g) for g in vby] if vby else []
    out = f"### T2. Headline per condition, MCQ\n\n{_table(_GROUP_HEADER, rows) if rows else '_05 wrote no per-condition groups (rerun 05 --out with the current version)_'}\n"
    if vqa:
        out += f"\nVQA (judged, threshold {vqa.get('threshold')}):\n\n{_table(_GROUP_HEADER, vrows) if vrows else '_05 wrote no per-condition VQA groups_'}\n"
    out += f"\nGroup intervals: scene-level paired bootstrap, {compare.get('n_boot_group', '?')} draws, seed {compare.get('seed', '?')}. F_h = harmful / BF16-correct.\n"
    return out


def t4_breakdowns(compare: dict) -> str:
    parts = []
    for section, title in (("mcq", "MCQ"), ("vqa", "VQA (judged)")):
        groups = (compare.get(section) or {}).get("groups") or {}
        for factor, rows in groups.items():
            if factor == "condition":
                continue
            parts.append(f"{title}, by {factor}:\n\n" + _table(_GROUP_HEADER, [_group_row(g["label"], g) for g in rows]))
    body = "\n\n".join(parts) if parts else "_no breakdowns_"
    return f"### T4. Breakdowns by cell, views and margin\n\n{body}\n"


# ---------------------------------------------------------------- T3


def _contrast_row(r: dict) -> list:
    a, b = r["a"], r["b"]
    side = lambda s: f"{_f(s.get('harmful_rate'))} (n={s.get('n', 0)}, ref-correct {s.get('ref_correct', 0)}/{s.get('ref_correct_scenes', 0)} sc.)"  # noqa: E731
    if r.get("confirmatory", True) is False:
        status = "descriptive (no decision)"
    elif r["assessed"]:
        status = "assessed" + (" *" if r.get("excludes_zero_adjusted") else "")
    else:
        status = f"UNASSESSED ({r['reason']}: {r.get('reason_detail')})"
    return [r["name"], f"{r['a_label']} vs {r['b_label']}", r.get("population", ""), side(a), side(b),
            f"{r['delta']:+.3f}" if r.get("delta") is not None else "n/a", _ci(r.get("ci95"), signed=True), _ci(r.get("ci99_bonferroni"), signed=True),
            f"{r.get('valid_draws', 0)}/{r.get('n_boot_contrast', '?')}", status]


def t3_contrasts(compare: dict) -> str:
    fam = compare.get("contrasts") or []
    desc = compare.get("descriptive") or []
    header = ["Id", "Contrast", "Population", "F_h A", "F_h B", "Δ", "95% raw", "99% Bonferroni", "Valid draws", "Status"]
    if not fam:
        return "### T3. Prespecified contrasts\n\n_no contrasts in the comparison JSON (run 05 --contrasts)_\n"
    cs = compare.get("contrast_settings") or {}
    out = f"### T3. Prespecified contrasts (family of {cs.get('family_size', 'n/a')}, Bonferroni α = {_f(cs.get('alpha_adjusted'), 3)})\n\n{_table(header, [_contrast_row(r) for r in fam])}\n"
    if desc:
        out += f"\nDescriptive, outside the family:\n\n{_table(header, [_contrast_row(r) for r in desc])}\n"
        notes = [r.get("note") for r in desc if r.get("note")]
        if notes:
            out += "\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    out += f"\n`*` = the Bonferroni-adjusted interval excludes zero. Draws {cs.get('n_boot_contrast', '?')}, seed {cs.get('seed', '?')}; support rule: ≥ {cs.get('support_min_ref_correct', '?')} reference-correct pairs from ≥ {cs.get('support_min_ref_scenes', '?')} scenes per side, valid draws ≥ 50%.\n"
    dropped = [(r["name"], r["support_dropped"]) for r in fam if r.get("support_dropped")]
    if dropped:
        out += "\nC4 common support, ids dropped per input file: " + "; ".join(f"{n} {d}" for n, d in dropped) + "\n"
    return out


# ---------------------------------------------------------------- T5


def t5_stability(summaries: list[dict]) -> str:
    if not summaries:
        return "### T5. Stability floor\n\n_no stability summaries_\n"
    rows = []
    for s in summaries:
        v = s.get("vqa") or {}
        rows.append([s.get("precision"), s.get("condition"), s.get("model"), s.get("repeats"), s.get("mcq_common"), _f(s.get("self_change_rate")), _f(s.get("self_correctness_change_rate")),
                     _f(s.get("self_text_change_rate")), _f(s.get("margin_range_max")), v.get("n", "n/a"), _f(v.get("self_correctness_change_rate")), s.get("errors", 0)])
    header = ["Precision", "Condition", "Model", "Repeats", "MCQ n", "Letter change", "Correctness change", "Text change", "Margin range max", "VQA n", "VQA correctness change", "Errors"]
    return f"### T5. Stability floor (self-disagreement across repeats)\n\n{_table(header, rows)}\n\nVQA columns come from judged repeats when available (plan section 7); text change alone is not a correctness measure.\n"


# ---------------------------------------------------------------- T6


def t6_deployment(profile: dict) -> str:
    if not profile or not (profile.get("precisions") or profile.get("reference")):
        return "### T6. Deployment profile\n\n_no pooled profile (run 07 --report --out)_\n"
    header = ["Precision", "Sessions", "Requests", "Errors", "TTFT p50 / p95 / p99 (s)", "Latency p50 / p95 / p99 (s)", "Decode p50 (tok/s)", "Peak memory (MiB)", "Host"]
    rows = []
    for prec, p in (profile.get("precisions") or {}).items():
        pooled = p["pooled"]
        host = p.get("host") or {}
        rows.append([f"**{prec}** (pooled)", p["n_sessions"], pooled["requests"], pooled["errors"],
                     " / ".join(_f(pooled["ttft_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                     " / ".join(_f(pooled["latency_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                     _f(pooled["decode_tok_s"].get("p50"), 1), pooled.get("gpu_peak_mib_max"), f"{host.get('hostname', '?')} / {host.get('gpu_name', '?')}"])
        for s in p["sessions"]:
            rows.append([f"{prec} s{s['session']}", 1, s["n"], s["errors"],
                         " / ".join(_f(s["ttft_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                         " / ".join(_f(s["latency_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                         _f(s["decode_tok_s"].get("p50"), 1), s.get("gpu_peak_mib"), ""])
    for s in profile.get("reference") or []:
        ident = f": {s['model']}" if s.get("model") else ""
        rows.append([f"{s['precision']} s{s['session']} (reference, unpooled{ident})", 1, s["n"], s["errors"],
                     " / ".join(_f(s["ttft_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                     " / ".join(_f(s["latency_s"].get(k), 2) for k in ("p50", "p95", "p99")),
                     _f(s["decode_tok_s"].get("p50"), 1), s.get("gpu_peak_mib"), ""])
    b = profile.get("budgets") or {}
    red = profile.get("memory_reduction")

    def v(x):
        return "n/a" if x is None else ("pass" if x else "FAIL")

    budget = (f"Budgets (proposal 5.2): TTFT p95 ≤ 2 s {{{', '.join(f'{k}: {v(x)}' for k, x in (b.get('ttft_p95_le_2s') or {}).items()) or 'n/a'}}}; "
              f"decode p50 ≥ 15 tok/s {{{', '.join(f'{k}: {v(x)}' for k, x in (b.get('decode_p50_ge_15') or {}).items()) or 'n/a'}}}; "
              f"INT4 peak memory reduction {_f(red, pct=True)} (≥ 25%: {v(b.get('memory_reduction_ge_25pct'))}).")
    return f"### T6. Deployment profile (pooled over sessions, per-session rows below)\n\n{_table(header, rows)}\n\n{budget} Pooled percentiles are over the concatenated requests of sessions sharing checkpoint, serve settings and host.\n"


# ---------------------------------------------------------------- T7


def t7_harmful(compare: dict, limit: int | None = None) -> str:
    ex = compare.get("harmful_examples") or []
    if not ex:
        return "### T7. Harmful examples\n\n_none_\n"
    rows = []
    for e in ex[:limit] if limit else ex:
        answers = f"{e.get('ref_answer')} → {e.get('quant_answer')}" if e.get("ref_answer") is not None else f"score {_f(e.get('ref_score'), 0)} → {_f(e.get('quant_score'), 0)}"
        rows.append([e["question_id"], e["condition"], e["cell"], e["n_views"], _f(e.get("margin"), 2), answers])
    shown = f" (showing {limit} of {len(ex)})" if limit and len(ex) > limit else f" ({len(ex)})"
    return f"### T7. Harmful examples{shown}\n\n{_table(['Question', 'Condition', 'Cell', 'Views', 'BF16 margin', 'BF16 → INT4'], rows)}\n"


# ---------------------------------------------------------------- all


def render_all(compare: dict | None, profile: dict | None, stability: list[dict] | None, harmful_limit: int | None = None) -> str:
    parts = []
    if compare:
        parts += [t1_coverage(compare), t2_headline(compare), t3_contrasts(compare), t4_breakdowns(compare)]
    else:
        parts += ["### T1. Coverage per cell\n\n_no comparison JSON_\n", "### T2. Headline per condition\n\n_no comparison JSON_\n", "### T3. Prespecified contrasts\n\n_no comparison JSON_\n", "### T4. Breakdowns\n\n_no comparison JSON_\n"]
    parts.append(t5_stability(stability or []))
    parts.append(t6_deployment(profile or {}))
    parts.append(t7_harmful(compare, harmful_limit) if compare else "### T7. Harmful examples\n\n_no comparison JSON_\n")
    head = ""
    if compare:
        head = f"Analysis class: **{compare.get('analysis_class', 'unknown')}**"
        if compare.get("provenance", {}).get("frozen"):
            head += f"; core freeze {compare['provenance']['frozen'].get('date')}"
        head += ".\n\n"
    return head + "\n".join(parts)
