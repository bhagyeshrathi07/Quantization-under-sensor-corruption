"""Pair reference (BF16) and quantized (INT4) result files, classify flips,
and report point estimates with paired bootstrap CIs over scenes.

One pair:
    python scripts/05_compare_flips.py results/pilot/bf16_clean.jsonl results/pilot/int4_clean.jsonl

Several conditions at once (ref quant ref quant ...), which enables --by condition:
    python scripts/05_compare_flips.py \\
        results/final/bf16_clean.jsonl results/final/int4_clean.jsonl \\
        results/final/bf16_Fog.jsonl   results/final/int4_Fog.jsonl --by cell n_views margin condition

With judged VQA cells (sidecars from scripts/08_judge_vqa.py next to each file):
    python scripts/05_compare_flips.py ... --vqa --judge-threshold 60

Every input file must carry a .meta.json sidecar (written by 03/06); sidecars
must agree on split, inputs, and serving settings, and every record's scene must
be in the declared population. Final-split files also require the core freeze
and refuse CLI overrides of frozen analysis settings. --no-provenance skips the
checks for pilot files written before sidecars existed; it never applies to
final-split files.

MCQ flip classes (strict rule: option letter at the start of the answer):
  harmful        reference correct, quantized wrong
  beneficial     reference wrong, quantized correct
  neutral_change answer changed, both right or both wrong
  unchanged      same letter
VQA pairs use the judge score against the threshold and have no neutral_change.

F_h (harmful_rate) = harmful / reference-correct. CIs are percentile intervals
from resampling scenes with replacement (default 2000 draws, seed 0). --contrasts
adds the five prespecified F_h contrasts (qflip.contrasts). Groups
below the coverage minimum (--min-n questions or --min-scenes scenes, defaults
from config analysis:) are printed as unassessed and carry assessed=false in
the JSON report.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.contrasts import format_contrast, prespecified_contrasts  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402
from qflip.flips import PairingError, bootstrap_scenes, coverage, group_by, merge_pairs, pair_records, pair_vqa_records, summarize  # noqa: E402
from qflip.freeze import FreezeError  # noqa: E402
from qflip.guard import judge_settings_for, legacy_files_check, validate_analysis_files  # noqa: E402
from qflip.judge import JudgeInputError, JudgeSettingsError, ensure_visual_description, load_scores, load_visual_description, production_scores, sidecar_path  # noqa: E402
from qflip.records import read_records  # noqa: E402


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def report(label: str, pairs, n_boot: int, seed: int, min_n: int, min_scenes: int = 1) -> dict:
    s = summarize(pairs)
    assessed = coverage(s, min_n, min_scenes)
    if not assessed:
        print(f"  {label:24s} n={s.get('n', 0)} scenes={s.get('scenes', 0)}  UNASSESSED (below {min_n} questions or {min_scenes} scenes)")
        return {"label": label, **s, "assessed": False}
    ci = bootstrap_scenes(pairs, n_boot=n_boot, seed=seed)
    hr = f"{s['harmful_rate']:.3f}" if s["harmful_rate"] is not None else "n/a"
    score = ""
    if s.get("ref_score_mean") is not None:
        score = f"  score {s['ref_score_mean']:.1f}->{s['quant_score_mean']:.1f}"
    print(
        f"  {label:24s} n={s['n']:4d} scenes={s['scenes']:3d}  acc {s['ref_acc']:.3f}->{s['quant_acc']:.3f} "
        f"d={s['acc_delta']:+.3f} {fmt_ci(ci.get('acc_delta'))}  "
        f"harm={s['harmful']} ben={s['beneficial']} neut={s['neutral_change']}  "
        f"F_h={hr} {fmt_ci(ci.get('harmful_rate'))}  change={s['change_rate']:.3f} {fmt_ci(ci.get('change_rate'))}{score}"
    )
    return {"label": label, **s, "assessed": True, "ci": ci}


def load_vqa_pairs(ref_path: str, quant_path: str, threshold: float, settings: dict, desc_data: dict, question_text):
    """Production VQA scores: replicate 0 whose input hash matches the current
    answer under the (frozen or configured) judge settings, with the object
    description derived from the hash-checked description file; everything
    else in the sidecar is reported as stale and not used."""
    ref_scores, quant_scores = load_scores(sidecar_path(ref_path)), load_scores(sidecar_path(quant_path))
    if not ref_scores or not quant_scores:
        print(f"  no judge sidecar for {ref_path} / {quant_path}; run scripts/08_judge_vqa.py first")
        return []
    ref_recs, quant_recs = read_records(ref_path), read_records(quant_path)
    ref_sel, ref_stale = production_scores(ref_scores, ref_recs, settings, desc_data, question_text)
    quant_sel, quant_stale = production_scores(quant_scores, quant_recs, settings, desc_data, question_text)
    for path, sel, stale in ((ref_path, ref_sel, ref_stale), (quant_path, quant_sel, quant_stale)):
        print(f"  {path}: {len(sel)} usable judge scores, {len(stale)} stale")
        for qid, reason in list(stale.items())[:3]:
            print(f"    stale {qid}: {reason}")
    return pair_vqa_records(ref_recs, quant_recs, ref_sel, quant_sel, threshold, question_text)


def section(title: str, pairs, by, edges, args, min_n, min_scenes) -> dict:
    out = {"groups": {}}
    print(f"\n{title}, overall:")
    out["overall"] = report("all", pairs, args.n_boot, args.seed, 1, 1)
    for factor in by:
        print(f"\n{title}, by {factor}:")
        out["groups"][factor] = [
            report(name, grp, args.n_boot, args.seed, min_n, min_scenes) for name, grp in group_by(pairs, factor, edges).items()
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="ref quant [ref quant ...]")
    ap.add_argument("--by", nargs="*", default=["cell", "n_views", "margin"], choices=["cell", "n_views", "margin", "condition"])
    ap.add_argument("--margin-edges", help="bin edges for the BF16 option margin (default analysis.margin_edges)")
    ap.add_argument("--n-boot", type=int, help="group bootstrap draws (default analysis.n_boot_group)")
    ap.add_argument("--n-boot-contrast", type=int, help="contrast bootstrap draws (default analysis.n_boot_contrast)")
    ap.add_argument("--contrasts", action="store_true", help="evaluate the five prespecified F_h contrasts (plan section 4) over the MCQ pairs, grouped by condition")
    ap.add_argument("--seed", type=int, help="bootstrap seed (default analysis.seed; frozen on final-split files)")
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--no-provenance", action="store_true", help="skip sidecar/population validation (pilot files from before sidecars existed; never for final)")
    ap.add_argument("--min-n", type=int, help="coverage minimum, questions per group (default analysis.min_questions)")
    ap.add_argument("--min-scenes", type=int, help="coverage minimum, scenes per group (default analysis.min_scenes)")
    ap.add_argument("--vqa", action="store_true", help="also pair judged VQA cells from the .judge.jsonl sidecars")
    ap.add_argument("--judge-threshold", type=float, help="rubric score counted as correct (default judge.correct_threshold)")
    ap.add_argument("--desc", default="data/visual_description.json", help="object visual descriptions (judge input, hash-checked)")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--show-harmful", type=int, default=20, help="print up to N harmful examples")
    args = ap.parse_args()
    if len(args.files) % 2:
        sys.exit("files must come in ref/quant pairs")
    resolved = [str(Path(f).resolve()) for f in args.files]
    if len(set(resolved)) != len(resolved):
        sys.exit("the same file is given more than once; duplicated inputs are refused")
    cfg = load_config(args.config)
    an = cfg["analysis"]
    if args.seed is None:
        args.seed = an.get("seed", 0)
    edges = tuple(float(x) for x in args.margin_edges.split(",") if x) if args.margin_edges else tuple(float(x) for x in an["margin_edges"])
    n_boot = args.n_boot if args.n_boot is not None else an["n_boot_group"]
    min_n = args.min_n if args.min_n is not None else an["min_questions"]
    min_scenes = args.min_scenes if args.min_scenes is not None else an["min_scenes"]
    args.n_boot = n_boot
    n_boot_contrast = args.n_boot_contrast if args.n_boot_contrast is not None else an["n_boot_contrast"]
    effective = {"margin_edges": list(edges), "n_boot_group": n_boot, "n_boot_contrast": n_boot_contrast, "seed": args.seed, "min_questions": min_n, "min_scenes": min_scenes,
                 "support_min_ref_correct": an["support_min_ref_correct"], "support_min_ref_scenes": an["support_min_ref_scenes"]}

    if args.no_provenance:
        provenance = legacy_files_check(args.files, args.split_file, args.exclusions_file)
        print("EXPLORATORY: --no-provenance skips sidecar/population validation; this output is not a confirmatory result")
    else:
        provenance = validate_analysis_files(args.files, cfg, split_file=args.split_file, exclusions_file=args.exclusions_file, settings=effective)
        print(f"provenance ok: split={provenance['split']} ({provenance['analysis_class']}), {len(provenance['files'])} files" + (", frozen " + provenance["frozen"]["date"] if "frozen" in provenance else ""))
    question_text = {q.question_id: q.question for q in load_questions(cfg["data"]["annotations"])}

    groups, file_ids = [], {}
    for ref_path, quant_path in zip(args.files[::2], args.files[1::2]):
        ref_recs, quant_recs = read_records(ref_path), read_records(quant_path)
        try:
            p = pair_records(ref_recs, quant_recs, question_text)
        except PairingError as e:
            sys.exit(f"pairing refused for {ref_path} vs {quant_path}: {e}")
        print(f"{ref_path} vs {quant_path}: {len(p)} paired MCQ")
        groups.append(p)
        for path, recs in ((ref_path, ref_recs), (quant_path, quant_recs)):
            # identity from the validated sidecar, else from any record (errored included),
            # so a file whose MCQs all errored still enters the C4 audit with an empty id set
            ident = provenance["files"].get(str(path), {}) if provenance.get("files") else {}
            cond, prec = ident.get("condition"), ident.get("precision")
            if (cond is None or prec is None) and recs:
                cond, prec = recs[0].get("condition"), recs[0].get("precision")
            if cond is None or prec is None:
                sys.exit(f"{path}: cannot establish condition/precision (no sidecar and no records)")
            key = f"{cond}/{prec}"
            if key in file_ids:
                sys.exit(f"two input files hold {key} records; duplicated inputs are refused")
            file_ids[key] = {r["question_id"] for r in recs if (r.get("cell") or "").endswith("MCQ") and not r.get("error")}
    try:
        pairs = merge_pairs(groups)
    except PairingError as e:
        sys.exit(f"inputs refused: {e}")
    if not pairs:
        sys.exit("no common MCQ questions")

    out = {"margin_edges": edges, "n_boot_group": n_boot, "n_boot_contrast": n_boot_contrast, "seed": args.seed, "min_questions": min_n, "min_scenes": min_scenes,
           "analysis_class": provenance["analysis_class"], "provenance": provenance}
    out["mcq"] = section("MCQ", pairs, args.by, edges, args, min_n, min_scenes)

    if args.contrasts:
        by_cond = {}
        for p in pairs:
            by_cond.setdefault(p.condition, []).append(p)
        res = prespecified_contrasts(by_cond, effective, args.seed, file_ids)
        print(f"\nprespecified contrasts ({out['analysis_class']}; family of {res['family_size']}, Bonferroni alpha {res['alpha_adjusted']}, {n_boot_contrast} draws, seed {args.seed}; * = adjusted interval excludes zero):")
        for r in res["family"]:
            print(format_contrast(r))
        print("descriptive (not in the family):")
        for r in res["descriptive"]:
            print(format_contrast(r))
            print(f"        {r['note']}")
        out["contrasts"] = res["family"]
        out["descriptive"] = res["descriptive"]
        out["contrast_settings"] = {k: effective[k] for k in ("n_boot_contrast", "seed", "support_min_ref_correct", "support_min_ref_scenes")}

    if args.vqa:
        # Final files: the frozen judge (settings, templates, system line, threshold)
        # and no CLI override; other files: the live config.
        try:
            js = judge_settings_for(provenance["split"], cfg)
        except FreezeError as e:
            sys.exit(f"--vqa: {e}")
        except JudgeSettingsError as e:
            sys.exit(f"--vqa: {e}; the judge settings decide which sidecar records are usable")
        judge_set = js["settings"]
        if provenance["split"] == "final":
            threshold = js["correct_threshold"]
            if args.judge_threshold is not None and args.judge_threshold != threshold:
                sys.exit(f"--judge-threshold {args.judge_threshold} differs from the frozen judge threshold {threshold}")
        else:
            threshold = args.judge_threshold if args.judge_threshold is not None else js["correct_threshold"]
        if threshold is None:
            sys.exit("--vqa needs a threshold: --judge-threshold or judge.correct_threshold in config (TO DECIDE on development scenes)")
        try:
            desc_data = load_visual_description(ensure_visual_description(args.desc))
        except JudgeInputError as e:
            sys.exit(f"--vqa: {e}")
        vqa = []
        for ref_path, quant_path in zip(args.files[::2], args.files[1::2]):
            vqa.extend(load_vqa_pairs(ref_path, quant_path, threshold, judge_set, desc_data, question_text))
        print(f"\n{len(vqa)} paired VQA at threshold {threshold}")
        if vqa:
            out["vqa"] = {"threshold": threshold, "judge_settings": judge_set, **section("VQA (judged)", vqa, [f for f in args.by if f != "margin"], edges, args, min_n, min_scenes)}
            pairs = pairs + vqa

    harmful = [p for p in pairs if p.flip == "harmful"]
    if harmful and args.show_harmful:
        print(f"\nharmful examples ({len(harmful)} total, showing {min(len(harmful), args.show_harmful)}):")
        for p in harmful[: args.show_harmful]:
            m = f"{p.margin:.2f}" if p.margin is not None else "n/a"
            sc = f" score {p.ref_score:.0f}->{p.quant_score:.0f}" if p.ref_score is not None else ""
            print(f"  {p.question_id} {p.condition:9s} {p.cell:15s} views={p.n_views} margin={m}{sc}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=list))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
