"""Core protocol freeze.

    python scripts/freeze_config.py                 # write manifests/config_frozen.yaml + PROTOCOL_frozen.md
    python scripts/freeze_config.py --check         # live config and inputs match the frozen copy (exit 1 otherwise)
    python scripts/freeze_config.py --check --serve-manifest manifests/serve_int4.txt --precision int4
    python scripts/freeze_config.py --amendment vqa512           # write manifests/config_frozen_vqa512.yaml from config/vqa512.yaml, on the core freeze
    python scripts/freeze_config.py --amendment vqa512 --check   # the amendment config, inputs and its fit to the core freeze

An amendment (qflip.freeze.AMENDMENTS) inherits the core freeze's environment,
models, snapshot and requirements and may differ from it in its registered
settings only; it is written from a clean commit like the core freeze.

Writing refuses when the git tree is dirty or when PROTOCOL.md still has a
TO DECIDE status outside sections 7 (VQA judge, frozen separately) and 12
(mitigation, deferred under its frozen selection rule). --check is what
run_matrix.sh, 03 --split final, and 05 call.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.guard import load_frozen_checked  # noqa: E402
from qflip.freeze import (  # noqa: E402
    AMENDMENTS,
    FROZEN_PATH,
    amendment,
    build_amendment_frozen,
    PROTOCOL_SNAPSHOT,
    REQUIREMENTS_FROZEN,
    FreezeError,
    blocking_open_items,
    build_frozen,
    check_config,
    check_serve_manifest,
    installed_versions,
    load_frozen,
    runtime_from_manifest,
)
from qflip.provenance import git_revision, parse_serve_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default config/pilot.yaml, or the amendment's config with --amendment")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--protocol", default="PROTOCOL.md")
    ap.add_argument("--frozen", help=f"default {FROZEN_PATH}, or the amendment's frozen path with --amendment")
    ap.add_argument("--snapshot", default=str(PROTOCOL_SNAPSHOT))
    ap.add_argument("--int4-model", help="own checkpoint directory (default quantize.out_dir)")
    ap.add_argument("--requirements", default=str(REQUIREMENTS_FROZEN), help="pinned serving environment recorded in the freeze")
    ap.add_argument("--quantize-manifest", help="scripts/quantize_awq.py manifest of the own checkpoint (default manifests/quantize_<checkpoint name>.json when present)")
    ap.add_argument("--runtime", default="auto", help="exact serving builds recorded in the freeze: auto (this environment: run in the serving venv on the host), manifest:<serve manifest>, or pins (dry runs only: the pins stand in for the builds)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--serve-manifest", help="with --check: also compare a serve manifest")
    ap.add_argument("--precision", choices=["bf16", "int4"])
    ap.add_argument("--allow-dirty", action="store_true", help="tests only")
    ap.add_argument("--amendment", choices=sorted(AMENDMENTS), help="write or check this amendment's freeze instead of the core freeze (its config and frozen path unless given explicitly)")
    ap.add_argument("--base-frozen", default=str(FROZEN_PATH), help="with --amendment (write): the core freeze the amendment is built on; --check uses the base the amendment file names")
    args = ap.parse_args()

    am = None
    if args.amendment:
        am = amendment(args.amendment)
        args.config = args.config or am["config"]
        args.frozen = args.frozen or am["frozen"]
    args.config = args.config or "config/pilot.yaml"
    args.frozen = args.frozen or str(FROZEN_PATH)
    cfg = load_config(args.config)
    if args.check:
        try:
            # the runtime guards' loader: an amendment must carry its block and fit the core freeze file it names (digest included)
            frozen = load_frozen_checked(args.frozen, "freeze check", am["name"] if am else None)
        except FreezeError as e:
            print("FREEZE CHECK FAILED")
            print("  " + str(e))
            sys.exit(1)
        diffs = check_config(cfg, frozen, args.split_file, args.exclusions_file)
        if args.serve_manifest:
            if not args.precision:
                sys.exit("--serve-manifest needs --precision")
            diffs += check_serve_manifest(parse_serve_manifest(args.serve_manifest), args.precision, frozen)
        if diffs:
            print("FREEZE CHECK FAILED")
            for d in diffs:
                print("  " + d)
            sys.exit(1)
        print(f"freeze check passed ({frozen['frozen']['date']}, commit {frozen['frozen']['git_commit'][:8]})" + (f", amendment {frozen['amendment']['name']} on core freeze {frozen['amendment']['base']['date']}" if frozen.get("amendment") else ""))
        return

    if am:
        rev, dirty = git_revision(".")
        if dirty and not args.allow_dirty:
            sys.exit("working tree is dirty; commit first so the amendment freeze records a clean commit")
        if Path(args.frozen).exists():
            sys.exit(f"{args.frozen} exists; a changed amendment is a new dated freeze (remove the old file deliberately and record it in the PROTOCOL change log)")
        base = load_frozen(args.base_frozen)
        if base is None:
            sys.exit(f"{args.base_frozen} missing; an amendment is built on the core freeze")
        try:
            frozen = build_amendment_frozen(cfg, am["name"], base, args.split_file, args.exclusions_file, git_state=(rev, dirty), base_path=args.base_frozen)
        except FreezeError as e:
            sys.exit(str(e))
        Path(args.frozen).parent.mkdir(parents=True, exist_ok=True)
        Path(args.frozen).write_text(yaml.safe_dump(frozen, sort_keys=False))
        print(f"wrote {args.frozen}: amendment {am['name']} on core freeze {base['frozen']['date']} (commit {base['frozen']['git_commit'][:8]}); differs {frozen['amendment']['differs']}; cells {frozen['amendment']['cells']}\n  commit {rev[:8]} dirty={dirty}")
        return

    open_items = blocking_open_items(Path(args.protocol).read_text())
    if open_items:
        print("PROTOCOL.md still has TO DECIDE rows outside the allowed sections:")
        for s, label in open_items:
            print(f"  section {s}: {label}")
        sys.exit(1)
    # the git state is verified here, before the snapshot and the frozen file are
    # written, and that state is what the freeze records
    rev, dirty = git_revision(".")
    if dirty and not args.allow_dirty:
        sys.exit("working tree is dirty; commit first so the freeze records a clean commit")
    if not Path(args.exclusions_file).exists():
        sys.exit(f"{args.exclusions_file} missing (scripts/exclude_exposed_scenes.py)")
    int4 = args.int4_model or cfg["quantize"]["out_dir"]
    if not Path(int4).is_dir():
        sys.exit(f"own checkpoint not found at {int4}; the freeze fingerprints it, so run on the host that has it")
    qm = args.quantize_manifest
    if qm is None:
        cand = Path("manifests") / f"quantize_{Path(int4).name}.json"
        qm = str(cand) if cand.exists() else None
    try:
        if args.runtime == "auto":
            runtime = installed_versions()
        elif args.runtime == "pins":
            runtime = None
        elif args.runtime.startswith("manifest:"):
            runtime = runtime_from_manifest(parse_serve_manifest(args.runtime.split(":", 1)[1]))
        else:
            sys.exit(f"--runtime must be auto, pins or manifest:<path>, got {args.runtime!r}")
        shutil.copyfile(args.protocol, args.snapshot)
        frozen = build_frozen(cfg, args.split_file, args.exclusions_file, int4, protocol_snapshot=args.snapshot, requirements_file=args.requirements, git_state=(rev, dirty), quantize_manifest=qm, runtime=runtime)
    except FreezeError as e:
        sys.exit(str(e))
    Path(args.frozen).parent.mkdir(parents=True, exist_ok=True)
    Path(args.frozen).write_text(yaml.safe_dump(frozen, sort_keys=False))
    env = frozen["models_frozen"]["int4"].get("quantization_env")
    print(f"wrote {args.frozen} and {args.snapshot} (sha256 {frozen['frozen']['protocol_snapshot_sha256'][:12]})\n  commit {rev[:8]} dirty={dirty}  int4 {frozen['models_frozen']['int4']['checkpoint_fingerprint'][:19]}"
          f"\n  pins {frozen['versions']}\n  runtime builds {frozen['builds']}" + ("  (pins stood in for the builds: dry run)" if args.runtime == "pins" else "")
          + (f"\n  quantization env {env}" if env else "\n  quantization env: no quantize manifest found (pass --quantize-manifest)"))


if __name__ == "__main__":
    main()
