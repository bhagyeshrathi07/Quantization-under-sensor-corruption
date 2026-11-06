"""Core protocol freeze: a frozen copy of the core config plus fingerprints of
every input that defines the experiment, and the checks that runs and analyses
perform against it.

The judge protocol (config section `judge`, PROTOCOL section 7 VQA row) is
frozen separately in manifests/judge_frozen.yaml, so judge changes never touch
the core freeze. The core protocol text is snapshotted to
manifests/PROTOCOL_frozen.md at freeze time (the snapshot path and its hash are
part of the freeze); the live PROTOCOL.md may gain change-log entries afterwards
without invalidating the core check.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from qflip.provenance import (
    CORE_SECTIONS,
    SERVING_MANIFEST_KEYS,
    checkpoint_fingerprint,
    core_config,
    core_config_sha256,
    git_revision,
    quant_and_dtype,
    sha256_file,
)

__all__ = ["CORE_SECTIONS", "core_config"]

FROZEN_PATH = Path("manifests/config_frozen.yaml")
PROTOCOL_SNAPSHOT = Path("manifests/PROTOCOL_frozen.md")
JUDGE_FROZEN_PATH = Path("manifests/judge_frozen.yaml")
# The dtype each precision must run in (PROTOCOL section 2); frozen per model.
EXPECTED_DTYPE = {"bf16": "bfloat16", "int4": "float16"}
# PROTOCOL sections whose TO DECIDE rows do not block the core freeze.
OPEN_ALLOWED_SECTIONS = {7, 12}


class FreezeError(RuntimeError):
    pass


def input_fingerprints(cfg: dict, split_file: str | Path, exclusions_file: str | Path, protocol_snapshot: str | Path) -> dict:
    return {
        "split_sha256": sha256_file(split_file),
        "exclusions_sha256": sha256_file(exclusions_file),
        "annotations_sha256": sha256_file(cfg["data"]["annotations"]),
        "prompt_sha256": sha256_file(cfg["prompt"]["system_file"]),
        "core_config_sha256": core_config_sha256(cfg),
        "protocol_snapshot_sha256": sha256_file(protocol_snapshot) if Path(protocol_snapshot).exists() else None,
    }


def build_frozen(cfg: dict, split_file: str | Path, exclusions_file: str | Path, int4_model_path: str, repo_root: str | Path = ".", protocol_snapshot: str | Path = PROTOCOL_SNAPSHOT) -> dict:
    if not Path(protocol_snapshot).exists():
        raise FreezeError(f"protocol snapshot {protocol_snapshot} missing; the freeze requires its hash")
    rev, dirty = git_revision(repo_root)
    q = cfg["quantize"]
    return {
        "frozen": {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": rev,
            "git_dirty": dirty,
            "protocol_snapshot": str(protocol_snapshot),
            **input_fingerprints(cfg, split_file, exclusions_file, protocol_snapshot),
        },
        "models_frozen": {
            "bf16": {"id": q["base_model"], "revision": q["revision"], "dtype": EXPECTED_DTYPE["bf16"]},
            "int4": {
                "path": int4_model_path,
                "checkpoint_fingerprint": checkpoint_fingerprint(int4_model_path),
                "quantization": "compressed-tensors",
                "dtype": EXPECTED_DTYPE["int4"],
            },
        },
        "core": core_config(cfg),
    }


def load_frozen(path: str | Path = FROZEN_PATH) -> dict | None:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else None


def check_config(cfg: dict, frozen: dict, split_file: str | Path, exclusions_file: str | Path) -> list[str]:
    """Differences between the live core config / inputs / protocol snapshot and the frozen copy."""
    diffs = []
    live, froz = core_config(cfg), frozen["core"]
    for section in sorted(set(live) | set(froz)):
        if live.get(section) != froz.get(section):
            for key in sorted(set(live.get(section, {})) | set(froz.get(section, {}))):
                if live.get(section, {}).get(key) != froz.get(section, {}).get(key):
                    diffs.append(f"config {section}.{key}: live={live.get(section, {}).get(key)!r} frozen={froz.get(section, {}).get(key)!r}")
    snapshot = frozen["frozen"].get("protocol_snapshot")
    if not snapshot or not frozen["frozen"].get("protocol_snapshot_sha256"):
        diffs.append("protocol_snapshot: the freeze records no snapshot path/hash; re-run scripts/freeze_config.py")
        snapshot = snapshot or PROTOCOL_SNAPSHOT
    fp = input_fingerprints(cfg, split_file, exclusions_file, snapshot)
    for k, v in fp.items():
        if frozen["frozen"].get(k) != v:
            diffs.append(f"{k}: live={str(v)[:12]} frozen={str(frozen['frozen'].get(k))[:12]}")
    return diffs


def frozen_serving_values(frozen: dict) -> dict[str, str]:
    """Manifest-form strings for every frozen serving setting."""
    s = frozen["core"]["serving"]
    return {
        "max_pixels": str(s["max_pixels"]),
        "max_model_len": str(s["max_model_len"]),
        "kv_args": f"--kv-cache-memory-bytes {s['kv_bytes']}",
        "seed": str(s["seed"]),
        "max_num_seqs": str(s["max_num_seqs"]),
        "prefix_caching": "true" if s["prefix_caching"] else "false",
    }


def _check_model(values: dict, precision: str, frozen: dict, fingerprint: str | None, quantization: str | None, dtype: str | None) -> list[str]:
    diffs = []
    m = frozen["models_frozen"][precision]
    if "dtype" not in m:
        diffs.append("dtype: the freeze records no dtype for this precision; re-run scripts/freeze_config.py")
    elif dtype != m["dtype"]:
        diffs.append(f"dtype: server={dtype!r} frozen={m['dtype']!r}")
    if precision == "bf16":
        if values.get("model") != m["id"]:
            diffs.append(f"model: server={values.get('model')!r} frozen={m['id']!r}")
        if values.get("revision") != m["revision"]:
            diffs.append(f"revision: server={values.get('revision')!r} frozen={m['revision']!r}")
    else:
        if values.get("model") != m["path"]:
            diffs.append(f"model: server={values.get('model')!r} frozen={m['path']!r}")
        elif fingerprint != m["checkpoint_fingerprint"]:
            diffs.append("checkpoint_fingerprint: the INT4 checkpoint differs from the frozen one")
        if quantization != m["quantization"]:
            diffs.append(f"quantization: server={quantization!r} frozen={m['quantization']!r}")
    return diffs


def check_serve_manifest(manifest: dict, precision: str, frozen: dict) -> list[str]:
    """Compare a parsed serve manifest (qflip.provenance.parse_serve_manifest)
    with every frozen serving setting and the frozen model for that precision."""
    diffs = []
    for k, v in frozen_serving_values(frozen).items():
        if manifest.get(k) != v:
            diffs.append(f"{k}: server={manifest.get(k)!r} frozen={v!r}")
    quant, dtype = quant_and_dtype(manifest.get("extra_args", ""))
    fp = checkpoint_fingerprint(manifest["model"]) if precision == "int4" and manifest.get("model") and Path(manifest["model"]).is_dir() else None
    diffs += _check_model({"model": manifest.get("model"), "revision": manifest.get("revision") or None}, precision, frozen, fp, quant, dtype)
    return diffs


def check_meta_against_frozen(meta: dict, frozen: dict) -> list[str]:
    """A result file's sidecar must record the frozen inputs, core config,
    serving settings, and model identity; the sidecar is what the file was
    actually produced under, independent of the live config."""
    diffs = []
    f = frozen["frozen"]
    for meta_key, frozen_key in (("split_file_sha256", "split_sha256"), ("exclusions_sha256", "exclusions_sha256"), ("annotations_sha256", "annotations_sha256"), ("prompt_sha256", "prompt_sha256"), ("core_config_sha256", "core_config_sha256")):
        if meta.get(meta_key) != f.get(frozen_key):
            diffs.append(f"{meta_key}: file={str(meta.get(meta_key))[:12]} frozen={str(f.get(frozen_key))[:12]}")
    for k, v in frozen_serving_values(frozen).items():
        if meta.get(k) != v:
            diffs.append(f"{k}: file={meta.get(k)!r} frozen={v!r}")
    diffs += _check_model({"model": meta.get("model_path_or_id"), "revision": meta.get("model_revision")}, meta["precision"], frozen, meta.get("checkpoint_fingerprint"), meta.get("quantization"), meta.get("dtype"))
    return diffs


_HEADING = re.compile(r"^##\s+(\d+)\.\s")
_CHANGELOG = re.compile(r"^##\s+Change log")


def protocol_open_items(text: str) -> list[tuple[int | None, str]]:
    """Table rows whose status cell (last column) starts with TO DECIDE, with
    their section number. Prose mentions of the phrase are ignored; the change
    log is skipped."""
    section, out, in_changelog = None, [], False
    for line in text.splitlines():
        h = _HEADING.match(line)
        if h:
            section, in_changelog = int(h.group(1)), False
            continue
        if _CHANGELOG.match(line):
            in_changelog = True
            continue
        if in_changelog or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[-1].upper().startswith("TO DECIDE"):
            out.append((section, cells[0]))
    return out


def blocking_open_items(text: str) -> list[tuple[int | None, str]]:
    return [(s, label) for s, label in protocol_open_items(text) if s not in OPEN_ALLOWED_SECTIONS]


def load_judge_frozen(path: str | Path = JUDGE_FROZEN_PATH) -> dict | None:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- judge freeze


def build_judge_frozen(jcfg: dict, correct_threshold: float, repo_root: str | Path = ".", validation: dict | None = None) -> dict:
    """The judge protocol freeze: request settings as sent, the correctness
    threshold, and fingerprints of the system line and every template. Written
    after development validation (plan section 7)."""
    from qflip.judge import judge_fingerprints, judge_settings

    if correct_threshold is None:
        raise FreezeError("the judge freeze needs correct_threshold (decided on development scenes)")
    rev, dirty = git_revision(repo_root)
    return {
        "frozen": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": rev, "git_dirty": dirty},
        "judge": judge_settings(jcfg),
        "correct_threshold": correct_threshold,
        "fingerprints": judge_fingerprints(),
        "validation": validation or {},
    }


def check_judge_frozen(jf: dict | None, jcfg: dict | None = None) -> list[str]:
    """Differences between the judge freeze and the live code (system line,
    templates) and, when `jcfg` is given, the live judge settings. Empty list =
    the frozen judge is the one this code and config would run."""
    from qflip.judge import JudgeSettingsError, judge_fingerprints, judge_settings

    if jf is None:
        return [f"{JUDGE_FROZEN_PATH} missing (judge protocol freeze)"]
    diffs = []
    if jf.get("correct_threshold") is None:
        diffs.append("correct_threshold missing from the judge freeze")
    if not jf.get("judge"):
        diffs.append("`judge` block (model and request settings) missing from the judge freeze")
        frozen_settings = None
    else:
        try:
            frozen_settings = judge_settings(jf["judge"])
        except JudgeSettingsError as e:
            diffs.append(f"frozen judge settings invalid: {e}")
            frozen_settings = None
        else:
            if frozen_settings != jf["judge"]:
                diffs.append(f"frozen judge settings are not canonical: {jf['judge']} vs {frozen_settings}")
    live = judge_fingerprints()
    fp = jf.get("fingerprints") or {}
    if fp.get("system_line_sha256") != live["system_line_sha256"]:
        diffs.append("judge system line differs from the frozen one")
    frozen_t = fp.get("templates") or {}
    for cell, sha in live["templates"].items():
        if cell not in frozen_t:
            diffs.append(f"template for {cell} not fingerprinted in the judge freeze")
        elif frozen_t[cell] != sha:
            diffs.append(f"template for {cell} differs from the frozen one ({sha} vs frozen {frozen_t[cell]})")
    if fp.get("visual_description_sha256") != live["visual_description_sha256"]:
        diffs.append("object-description file digest differs from the frozen one (a description-source update needs a new judge freeze)")
    if jcfg is not None and frozen_settings is not None:
        try:
            live_settings = judge_settings(jcfg)
        except JudgeSettingsError as e:
            diffs.append(f"live judge settings: {e}")
        else:
            for k in sorted(set(live_settings) | set(frozen_settings)):
                if live_settings.get(k) != frozen_settings.get(k):
                    diffs.append(f"judge setting {k}: live={live_settings.get(k)!r} frozen={frozen_settings.get(k)!r} (overrides are not allowed on final-split files)")
    return diffs
