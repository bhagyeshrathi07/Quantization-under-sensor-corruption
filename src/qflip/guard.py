"""Entry-point guards shared by the run scripts (03, 06) and the analysis
script (05): provenance sidecars, active-server validation, the final-run
freeze guard, and analysis-side population checks."""

from pathlib import Path

from qflip.freeze import FROZEN_PATH, JUDGE_FROZEN_PATH, FreezeError, check_config, check_judge_frozen, check_meta_against_frozen, check_serve_manifest, load_frozen, load_judge_frozen
from qflip.judge import judge_settings
from qflip.provenance import (
    PRECISION_FIELDS,
    ProvenanceError,
    build_run_meta,
    check_server,
    check_records_against_meta,
    check_server_process,
    compat_diff,
    effective_base_url,
    ensure_meta,
    load_meta,
    parse_serve_manifest,
    sha256_file,
)
from qflip.records import iter_raw_records
from qflip.split import DEFAULT_EXCLUSIONS_FILE, DEFAULT_SPLIT_FILE, scenes_for


def final_guard(cfg: dict, precision: str, serve_manifest_path: str | Path, split_file: str | Path, exclusions_file: str | Path, frozen_path: str | Path = FROZEN_PATH) -> dict:
    """A final-split run needs the core freeze, the exclusion list, a config that
    matches the freeze, and a server whose manifest matches the frozen model and
    serving settings. Returns the frozen dict."""
    frozen = load_frozen(frozen_path)
    if frozen is None:
        raise FreezeError(f"final-split run refused: {frozen_path} missing (python scripts/freeze_config.py)")
    if not Path(exclusions_file).exists():
        raise FreezeError(f"final-split run refused: {exclusions_file} missing")
    diffs = check_config(cfg, frozen, split_file, exclusions_file)
    diffs += check_serve_manifest(parse_serve_manifest(serve_manifest_path), precision, frozen)
    if diffs:
        raise FreezeError("final-split run refused; differences from the freeze:\n  " + "\n  ".join(diffs))
    return frozen


def prepare_run(
    cfg: dict,
    *,
    config_path: str | Path,
    precision: str,
    condition: str,
    split: str,
    result_file: str | Path,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE,
    manifests_dir: str | Path = "manifests",
    base_url: str | None = None,
    check_live_server: bool = True,
    frozen_path: str | Path = FROZEN_PATH,
) -> dict:
    """Validate the server and write/check the sidecar before the first record.

    The live check uses the client's effective endpoint (QFLIP_BASE_URL or the
    config), requires the served alias, binds the manifest to the running
    process (pid, command line, port) and, when the server reports the model
    it loaded, requires it to be the manifest's model."""
    meta = validate_server_and_freeze(cfg, config_path=config_path, precision=precision, condition=condition, split=split, split_file=split_file,
                                      exclusions_file=exclusions_file, manifests_dir=manifests_dir, base_url=base_url, check_live_server=check_live_server,
                                      frozen_path=frozen_path, require_freeze=(split == "final"))
    allowed = None if split == "all" else scenes_for(split, split_file, exclusions_file)
    ensure_meta(result_file, meta, allowed_scenes=allowed)
    return meta


def validate_server_and_freeze(
    cfg: dict,
    *,
    config_path: str | Path,
    precision: str,
    condition: str,
    split: str,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE,
    manifests_dir: str | Path = "manifests",
    base_url: str | None = None,
    check_live_server: bool = True,
    frozen_path: str | Path = FROZEN_PATH,
    require_freeze: bool = False,
) -> dict:
    """The server binding every measurement needs (manifest present, served
    alias at the client's effective endpoint, listener owned by the manifest's
    process, loaded model when reported) and, when required, the core freeze
    (config and serve manifest equal to the frozen study). Returns the
    provenance block (the sidecar's compatibility fields)."""
    served = cfg["models"][precision]
    manifest = Path(manifests_dir) / f"serve_{precision}.txt"
    if not manifest.exists():
        raise ProvenanceError(f"{manifest} missing; start the server with scripts/01_serve.sh {precision}")
    man = parse_serve_manifest(manifest)
    if check_live_server:
        url = base_url or effective_base_url(cfg)
        check_server(url, served, expect_model=man.get("model"))
        check_server_process(man, url)
    if require_freeze:
        final_guard(cfg, precision, manifest, split_file, exclusions_file, frozen_path)
    return build_run_meta(
        split=split,
        precision=precision,
        condition=condition,
        config_path=config_path,
        annotations_path=cfg["data"]["annotations"],
        prompt_path=cfg["prompt"]["system_file"],
        split_file=split_file,
        exclusions_file=exclusions_file,
        serve_manifest_path=manifest,
        served_model_name=served,
    )


def prepare_measurement(cfg: dict, *, study: bool = True, **kw) -> dict:
    """07's guard, before the warm-up requests: the server binding always, and
    the core freeze unless the session is declared exploratory (`study=False`),
    in which case the file is marked and never pooled into the study."""
    return validate_server_and_freeze(cfg, require_freeze=study, **kw)


def holdout_check(files: list[str | Path], split_file: str | Path = DEFAULT_SPLIT_FILE, exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE) -> None:
    """No row of these files may come from the unexposed final holdout: files
    carrying such scenes are handled only as final-split files under the
    freezes, never on an exploratory path."""
    holdout = scenes_for("final", split_file, exclusions_file)
    for f in files:
        hit = {r["scene_token"] for r in iter_raw_records(f) if r.get("scene_token") in holdout}
        if hit:
            raise ProvenanceError(f"{f} contains {len(hit)} scene(s) of the unexposed final holdout; such files are judged or analyzed only as final-split files under the freezes")


def legacy_files_check(files: list[str | Path], split_file: str | Path = DEFAULT_SPLIT_FILE, exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE) -> dict:
    """`--no-provenance` is for pilot files written before sidecars existed:
    only sidecar-less files qualify (a file with a sidecar is validated the
    normal way), none of their rows may come from the final holdout, and
    whatever comes out is labeled exploratory."""
    for f in files:
        m = load_meta(f)
        if m is not None and m.get("split") == "final":
            raise ProvenanceError(f"{f} declares split=final; --no-provenance is not allowed for final-split files (run without it, with the freeze in place)")
        if m is not None:
            raise ProvenanceError(f"{f} has a .meta.json sidecar; --no-provenance is only for files without one (run without it)")
    holdout_check(files, split_file, exclusions_file)
    return {"split": "unknown", "analysis_class": "exploratory", "files": {str(f): {"sidecar": False} for f in files}}


def validate_analysis_files(
    files: list[str | Path],
    cfg: dict,
    *,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE,
    frozen_path: str | Path = FROZEN_PATH,
    settings: dict | None = None,
) -> dict:
    """Every result file needs a sidecar; all sidecars must agree on the inputs
    (split, exclusions, annotations, core config, prompt, serving settings),
    differing only in precision fields and condition; every record's scene must
    belong to the declared population and every raw row (retries included)
    must carry the sidecar's precision and condition. Final-split files additionally require
    the core freeze, every sidecar equal to the frozen inputs / core config /
    serving settings / model identity, and effective analysis settings equal to
    the frozen ones."""
    metas = {}
    for f in files:
        m = load_meta(f)
        if m is None:
            raise ProvenanceError(f"{f} has no .meta.json sidecar; results without provenance are not analyzed")
        metas[str(f)] = m
    first_name, first = next(iter(metas.items()))
    ignore = PRECISION_FIELDS + ("condition",)
    for name, m in metas.items():
        diff = compat_diff(first, m, ignore=ignore)
        if diff:
            raise ProvenanceError(f"{name} and {first_name} come from different setups; differing fields: {diff}")
    if first["split_file_sha256"] != sha256_file(split_file):
        raise ProvenanceError(f"{split_file} changed since the runs were made")
    excl_sha = sha256_file(exclusions_file) if Path(exclusions_file).exists() else None
    split = first["split"]
    if split == "final" and first.get("exclusions_sha256") != excl_sha:
        raise ProvenanceError(f"{exclusions_file} changed since the runs were made")
    allowed = None if split == "all" else scenes_for(split, split_file, exclusions_file)
    for f in files:
        check_records_against_meta(f, metas[str(f)], allowed)
    if split == "all":
        # an unrestricted run is exploratory; it must not touch the holdout
        holdout_check(files, split_file, exclusions_file)
    out = {
        "split": split,
        "analysis_class": "confirmatory" if split == "final" else "exploratory",
        "files": {name: {k: m.get(k) for k in ("precision", "condition", "model_path_or_id", "checkpoint_fingerprint", "code_revision", "created")} for name, m in metas.items()},
    }
    if split == "final":
        frozen = load_frozen(frozen_path)
        if frozen is None:
            raise FreezeError(f"final-split analysis refused: {frozen_path} missing")
        diffs = check_config(cfg, frozen, split_file, exclusions_file)
        fa = frozen["core"]["analysis"]
        for k, v in (settings or {}).items():
            if k in fa and fa[k] != v:
                diffs.append(f"analysis setting {k}: effective={v!r} frozen={fa[k]!r} (CLI overrides are not allowed on final-split files)")
        for name, m in metas.items():
            diffs += [f"{name}: {d}" for d in check_meta_against_frozen(m, frozen)]
        if diffs:
            raise FreezeError("final-split analysis refused; differences from the freeze:\n  " + "\n  ".join(diffs))
        out["frozen"] = frozen["frozen"]
    return out


def judge_settings_for(split: str | None, cfg: dict, cli_model: str | None = None, judge_frozen_path: str | Path = JUDGE_FROZEN_PATH) -> dict:
    """Judge request settings for files of `split`. Final-split files take the
    frozen judge (manifests/judge_frozen.yaml), whose fingerprints must match
    the live templates and system line and whose settings the live config and
    any CLI override must equal; other splits use the live config."""
    jcfg = dict(cfg.get("judge") or {})
    if cli_model:
        jcfg["model"] = cli_model
    if split != "final":
        return {"settings": judge_settings(jcfg), "correct_threshold": jcfg.get("correct_threshold"), "judge_frozen": None}
    jf = load_judge_frozen(judge_frozen_path)
    diffs = check_judge_frozen(jf, jcfg)
    if diffs:
        raise FreezeError("final-split judging refused; judge freeze:\n  " + "\n  ".join(diffs))
    return {"settings": judge_settings(jf["judge"]), "correct_threshold": jf["correct_threshold"], "judge_frozen": jf["frozen"]}


def prepare_judging(
    files: list[str | Path],
    cfg: dict,
    *,
    cli_model: str | None = None,
    no_provenance: bool = False,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE,
    frozen_path: str | Path = FROZEN_PATH,
    judge_frozen_path: str | Path = JUDGE_FROZEN_PATH,
) -> dict:
    """What 08 must establish before any judge call that carries result data:
    the files' provenance and population (the same validation as 05) and the
    judge settings those files may be judged with. Final-split files need the
    core freeze and the judge freeze; no file at all (a bare compat probe)
    needs neither."""
    if not files:
        prov = {"split": None, "analysis_class": "exploratory", "files": {}}
    elif no_provenance:
        prov = legacy_files_check(files, split_file, exclusions_file)
    else:
        prov = validate_analysis_files(files, cfg, split_file=split_file, exclusions_file=exclusions_file, frozen_path=frozen_path)
    if files and prov["split"] != "final":
        holdout_check(files, split_file, exclusions_file)  # whatever the sidecar says, the rows decide
    return {"provenance": prov, **judge_settings_for(prov["split"], cfg, cli_model, judge_frozen_path)}
