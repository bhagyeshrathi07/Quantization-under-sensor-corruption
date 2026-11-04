"""Entry-point guards shared by the run scripts (03, 06) and the analysis
script (05): provenance sidecars, active-server validation, the final-run
freeze guard, and analysis-side population checks."""

from pathlib import Path

from qflip.freeze import FROZEN_PATH, FreezeError, check_config, check_meta_against_frozen, check_serve_manifest, load_frozen
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
    served = cfg["models"][precision]
    manifest = Path(manifests_dir) / f"serve_{precision}.txt"
    if not manifest.exists():
        raise ProvenanceError(f"{manifest} missing; start the server with scripts/01_serve.sh {precision}")
    man = parse_serve_manifest(manifest)
    if check_live_server:
        url = base_url or effective_base_url(cfg)
        check_server(url, served, expect_model=man.get("model"))
        check_server_process(man, url)
    if split == "final":
        final_guard(cfg, precision, manifest, split_file, exclusions_file, frozen_path)
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
    allowed = None if split == "all" else scenes_for(split, split_file, exclusions_file)
    ensure_meta(result_file, meta, allowed_scenes=allowed)
    return meta


def legacy_files_check(files: list[str | Path]) -> dict:
    """`--no-provenance` is for pilot files written before sidecars existed.
    A file whose sidecar declares the final split is never analyzed this way,
    and whatever comes out is labeled exploratory."""
    for f in files:
        m = load_meta(f)
        if m is not None and m.get("split") == "final":
            raise ProvenanceError(f"{f} declares split=final; --no-provenance is not allowed for final-split files (run without it, with the freeze in place)")
    return {"split": "unknown", "analysis_class": "exploratory", "files": {str(f): {"sidecar": load_meta(f) is not None} for f in files}}


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
