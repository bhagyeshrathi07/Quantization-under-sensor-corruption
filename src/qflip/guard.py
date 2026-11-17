"""Entry-point guards shared by the run scripts (03, 06) and the analysis
script (05): provenance sidecars, active-server validation, the final-run
freeze guard, and analysis-side population checks."""

from pathlib import Path

from qflip.freeze import FROZEN_PATH, JUDGE_FROZEN_PATH, JUDGE_MANIFEST_PATH, FreezeError, check_amendment, check_config, check_judge_frozen, check_judge_manifest, check_meta_against_frozen, check_serve_manifest, freeze_identity, load_frozen, load_judge_frozen, production_replicates, read_judge_manifest
from qflip.judge import judge_backend, judge_settings
from qflip.provenance import (
    build_run_meta,
    check_records_against_meta,
    check_server,
    check_server_process,
    compat_diff,
    effective_base_url,
    ensure_meta,
    load_meta,
    parse_serve_manifest,
    PRECISION_FIELDS,
    ProvenanceError,
    sha256_file,
)
from qflip.records import iter_raw_records
from qflip.split import DEFAULT_EXCLUSIONS_FILE, DEFAULT_SPLIT_FILE, scenes_for


def load_frozen_checked(frozen_path: str | Path, what: str, amendment: str | None = None) -> dict:
    """The freeze at `frozen_path`. With `amendment`, the file must carry an
    amendment block of that name and is checked against the core freeze it
    records as its base (digest, identity, inherited fingerprints, allowed
    difference only); without it, the file must be a plain core freeze."""
    frozen = load_frozen(frozen_path)
    if frozen is None:
        raise FreezeError(f"{what} refused: {frozen_path} missing (python scripts/freeze_config.py" + (f" --amendment {amendment}" if amendment else "") + ")")
    block = frozen.get("amendment") or {}
    if amendment is not None and block.get("name") != amendment:
        raise FreezeError(f"{what} refused: {frozen_path} carries " + (f"amendment block {block.get('name')!r}" if block else "no amendment block") + f", amendment {amendment!r} was selected")
    if amendment is None and block:
        raise FreezeError(f"{what} refused: {frozen_path} is the freeze of amendment {block.get('name')!r}; select it with --amendment {block.get('name')}")
    if frozen.get("amendment"):
        base_path = Path(frozen["amendment"]["base"]["path"])
        base = load_frozen(base_path) if base_path.resolve() != Path(frozen_path).resolve() else None
        diffs = check_amendment(frozen, base, sha256_file(base_path) if base is not None else None)
        if diffs:
            raise FreezeError(f"{what} refused; the amendment freeze does not fit the core freeze:\n  " + "\n  ".join(diffs))
    return frozen


def check_files_under_freeze(metas: dict, cfg: dict, frozen: dict, split_file, exclusions_file, what: str) -> None:
    """Every sidecar produced under this freeze: the live config equal to it,
    each sidecar's inputs, core config, serving settings and model identity
    equal to it, and `freeze_checked` equal to its identity (the run passed the
    freeze guard under this very freeze or amendment)."""
    diffs = check_config(cfg, frozen, split_file, exclusions_file)
    ident = freeze_identity(frozen)
    for name, m in metas.items():
        if m.get("freeze_checked") != ident:
            diffs.append(f"{name}: freeze_checked={m.get('freeze_checked')!r}, expected {ident!r} (the run must pass the freeze guard under this freeze: `03 --require-freeze` / REQUIRE_FREEZE=1, or --amendment)")
        diffs += [f"{name}: {d}" for d in check_meta_against_frozen(m, frozen)]
    if diffs:
        raise FreezeError(f"{what} refused; the files were not produced under this freeze:\n  " + "\n  ".join(diffs))


def amendment_cells_check(files: list[str | Path], frozen: dict) -> None:
    """Under an amendment, every raw row of every file must belong to one of
    the amendment's cells: the amendment covers those cells and nothing else,
    so an MCQ row under the VQA amendment is a protocol breach, not data."""
    am = frozen.get("amendment")
    if not am:
        return
    cells = set(am["cells"])
    for f in files:
        bad = {r.get("cell") for r in iter_raw_records(f) if r.get("cell") not in cells}
        if bad:
            raise ProvenanceError(f"{f}: rows of cell(s) {sorted(map(str, bad))} under amendment {am['name']}, which covers {sorted(cells)} only")


def final_guard(cfg: dict, precision: str, serve_manifest_path: str | Path, split_file: str | Path, exclusions_file: str | Path, frozen_path: str | Path = FROZEN_PATH, amendment: str | None = None) -> dict:
    """A final-split run needs the core freeze, the exclusion list, a config that
    matches the freeze, and a server whose manifest matches the frozen model and
    serving settings. Returns the frozen dict. An amendment freeze is accepted
    only if it still fits the core freeze."""
    frozen = load_frozen_checked(frozen_path, "final-split run", amendment)
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
    require_freeze: bool | None = None,
    amendment: str | None = None,
) -> dict:
    """Validate the server and write/check the sidecar before the first record.
    `require_freeze` makes any run, a development run included, refuse unless
    config and server manifest equal the core freeze; the final split is
    checked whatever the caller passes (False cannot switch it off); the sidecar then carries `freeze_checked` (the freeze's date and
    commit), so a calibration run on development scenes can be shown to have
    used the frozen generation settings.

    The live check uses the client's effective endpoint (QFLIP_BASE_URL or the
    config), requires the served alias, binds the manifest to the running
    process (pid, command line, port) and, when the server reports the model
    it loaded, requires it to be the manifest's model."""
    meta = validate_server_and_freeze(cfg, config_path=config_path, precision=precision, condition=condition, split=split, split_file=split_file,
                                      exclusions_file=exclusions_file, manifests_dir=manifests_dir, base_url=base_url, check_live_server=check_live_server,
                                      frozen_path=frozen_path, require_freeze=(split == "final") or bool(require_freeze), amendment=amendment)
    existing = load_meta(result_file)
    if existing is not None and (existing.get("freeze_checked") is not None or meta.get("freeze_checked") is not None) and existing.get("freeze_checked") != meta.get("freeze_checked"):
        # identities must match whenever either side has one: a checked run does not adopt an unchecked file, and an
        # unchecked run does not continue a checked file (it would keep the file's freeze_checked without the check)
        raise FreezeError(f"{result_file}: its sidecar records freeze_checked={existing.get('freeze_checked')!r}, this run has {meta.get('freeze_checked')!r}; a file is resumed only by a run under the same freeze check")
    if amendment and Path(result_file).exists():
        amendment_cells_check([result_file], load_frozen(frozen_path))
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
    amendment: str | None = None,
) -> dict:
    """The server binding every measurement needs (manifest present, served
    alias at the client's effective endpoint, listener owned by the manifest's
    process, loaded model when reported) and, when required or on the final
    split, the core freeze (config and serve manifest equal to the frozen study). Returns the
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
    # the final split is checked whatever the caller passed; `require_freeze` can only add the check
    if amendment and not (require_freeze or split == "final"):
        raise FreezeError(f"a run under amendment {amendment!r} must require its freeze (03 --amendment sets this)")
    frozen = final_guard(cfg, precision, manifest, split_file, exclusions_file, frozen_path, amendment) if (require_freeze or split == "final") else None
    meta = build_run_meta(
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
    if frozen is not None:
        meta["freeze_checked"] = freeze_identity(frozen)
    return meta


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
    amendment: str | None = None,
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
    # An amendment is always selected explicitly, so its freeze must exist and, on every split, the
    # files must have been produced under it (identity, sidecars, live config, cells) and the amendment
    # must still fit the core freeze. The population (split) keeps its own checks below; the inferential
    # status of amendment files is exploratory whatever the split: the VQA amendment is a secondary measure.
    produced_under = {m.get("freeze_checked", {}).get("amendment") for m in metas.values() if isinstance(m.get("freeze_checked"), dict)} - {None}
    if not amendment and produced_under:
        # files produced under an amendment are handled under that amendment on every split, never as plain core files
        raise FreezeError(f"these files were produced under amendment {sorted(produced_under)}; select it with --amendment {sorted(produced_under)[0]}")
    if amendment:
        am_frozen = load_frozen_checked(frozen_path, "analysis under the amendment", amendment)
        amendment_cells_check(files, am_frozen)
        check_files_under_freeze(metas, cfg, am_frozen, split_file, exclusions_file, "analysis under the amendment")
        out["amendment"] = am_frozen["amendment"]["name"]
        out["analysis_class"] = "exploratory"
        out["frozen"] = am_frozen["frozen"]
    if split == "final":
        frozen = load_frozen_checked(frozen_path, "final-split analysis", amendment)
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
        if frozen.get("amendment"):
            out["analysis_class"] = "exploratory"  # final population, secondary measure
    return out


def check_judge_server(settings: dict, backend: dict, manifest_path: str | Path = JUDGE_MANIFEST_PATH, proc_root: str | Path = "/proc") -> int:
    """A local judge's running server, bound the way the study servers are:
    the serve manifest names the frozen revision and model; the endpoint lists
    the served alias and its model card's `root` is the manifest's model; the
    manifest pid is alive, its command line names the model, the port and the
    frozen revision, and it (or a descendant) owns the LISTEN socket. Returns
    the pid."""
    diffs = check_judge_manifest(backend, manifest_path)
    if diffs:
        raise FreezeError("local judge identity:\n  " + "\n  ".join(diffs))
    man = read_judge_manifest(manifest_path)
    check_server(settings["base_url"], settings["model"], expect_model=man.get("model"))
    pid = check_server_process(man, settings["base_url"], proc_root)
    argv = [a.decode(errors="replace") for a in (Path(proc_root) / str(pid) / "cmdline").read_bytes().split(b"\0") if a]
    revs = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--revision"]
    if revs != [man["revision"]]:
        raise ProvenanceError(f"judge server pid {pid} command line revision {revs} is not the manifest revision {man['revision']!r}")
    return pid


def judge_settings_for(split: str | None, cfg: dict, cli_model: str | None = None, judge_frozen_path: str | Path = JUDGE_FROZEN_PATH) -> dict:
    """Judge request settings and backend description for files of `split`.
    Final-split files take the frozen judge (manifests/judge_frozen.yaml),
    whose fingerprints must match the live templates, system line and
    extractor and whose settings and backend the live config and any CLI
    override must equal; other splits use the live config."""
    jcfg = dict(cfg.get("judge") or {})
    if cli_model:
        jcfg["model"] = cli_model
    if split != "final":
        return {"settings": judge_settings(jcfg), "backend": judge_backend(jcfg), "correct_threshold": jcfg.get("correct_threshold"), "production_replicates": production_replicates(jcfg), "judge_frozen": None}
    jf = load_judge_frozen(judge_frozen_path)
    diffs = check_judge_frozen(jf, jcfg)
    if diffs:
        raise FreezeError("final-split judging refused; judge freeze:\n  " + "\n  ".join(diffs))
    # the frozen backend carries the config description plus the response model observed during validation
    return {"settings": judge_settings(jf["judge"]), "backend": dict(jf.get("backend") or {}), "correct_threshold": jf["correct_threshold"], "production_replicates": int(jf.get("production_replicates", 1)), "judge_frozen": jf["frozen"]}


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
    amendment: str | None = None,
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
        prov = validate_analysis_files(files, cfg, split_file=split_file, exclusions_file=exclusions_file, frozen_path=frozen_path, amendment=amendment)
    if files and prov["split"] != "final":
        holdout_check(files, split_file, exclusions_file)  # whatever the sidecar says, the rows decide
    return {"provenance": prov, **judge_settings_for(prov["split"], cfg, cli_model, judge_frozen_path)}


def prepare_calibration_files(
    files: list[str | Path],
    cfg: dict,
    *,
    no_provenance: bool = False,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    exclusions_file: str | Path = DEFAULT_EXCLUSIONS_FILE,
    frozen_path: str | Path = FROZEN_PATH,
    amendment: str | None = None,
) -> dict:
    """What the label sheet (10) and the calibration (11) must establish about
    the answer files: the same provenance as 05 and 08, membership of the
    development split, and production under the core freeze: every sidecar
    carries `freeze_checked` equal to the freeze's date and commit (written
    only by a run that passed `03 --require-freeze`) and records the frozen
    inputs, core config, serving settings and model identity
    (check_meta_against_frozen). The result is labelled `calibration`.
    `no_provenance` is for sidecar-less synthetic files only (tests), labelled
    `exploratory`; a file with a sidecar or a final-split row is refused there
    as everywhere."""
    if not files:
        raise ProvenanceError("no answer files given")
    if no_provenance:
        prov = legacy_files_check(files, split_file, exclusions_file)
        prov["note"] = "EXPLORATORY: --no-provenance skips the sidecar, population and freeze checks; nothing from this run may be frozen"
        return prov
    prov = validate_analysis_files(files, cfg, split_file=split_file, exclusions_file=exclusions_file, frozen_path=frozen_path, amendment=amendment)
    if prov["split"] != "development":
        raise ProvenanceError(f"calibration files must be development-split answers; these declare split={prov['split']!r}")
    frozen = load_frozen_checked(frozen_path, "calibration", amendment)
    check_files_under_freeze({str(f): load_meta(f) for f in files}, cfg, frozen, split_file, exclusions_file, "calibration")
    prov["analysis_class"] = "calibration"
    prov["frozen"] = frozen["frozen"]
    if frozen.get("amendment"):
        prov["amendment"] = frozen["amendment"]["name"]
    return prov
