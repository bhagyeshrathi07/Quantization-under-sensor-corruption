"""Core protocol freeze.

    python scripts/freeze_config.py                 # write manifests/config_frozen.yaml + PROTOCOL_frozen.md
    python scripts/freeze_config.py --check         # live config and inputs match the frozen copy (exit 1 otherwise)
    python scripts/freeze_config.py --check --serve-manifest manifests/serve_int4.txt --precision int4

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
from qflip.freeze import (  # noqa: E402
    FROZEN_PATH,
    PROTOCOL_SNAPSHOT,
    blocking_open_items,
    build_frozen,
    check_config,
    check_serve_manifest,
    load_frozen,
)
from qflip.provenance import git_revision, parse_serve_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--exclusions-file", default="manifests/scene_exclusions.json")
    ap.add_argument("--protocol", default="PROTOCOL.md")
    ap.add_argument("--frozen", default=str(FROZEN_PATH))
    ap.add_argument("--snapshot", default=str(PROTOCOL_SNAPSHOT))
    ap.add_argument("--int4-model", help="own checkpoint directory (default quantize.out_dir)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--serve-manifest", help="with --check: also compare a serve manifest")
    ap.add_argument("--precision", choices=["bf16", "int4"])
    ap.add_argument("--allow-dirty", action="store_true", help="tests only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.check:
        frozen = load_frozen(args.frozen)
        if frozen is None:
            sys.exit(f"no freeze: {args.frozen} missing")
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
        print(f"freeze check passed ({frozen['frozen']['date']}, commit {frozen['frozen']['git_commit'][:8]})")
        return

    open_items = blocking_open_items(Path(args.protocol).read_text())
    if open_items:
        print("PROTOCOL.md still has TO DECIDE rows outside the allowed sections:")
        for s, label in open_items:
            print(f"  section {s}: {label}")
        sys.exit(1)
    rev, dirty = git_revision(".")
    if dirty and not args.allow_dirty:
        sys.exit("working tree is dirty; commit first so the freeze records a clean commit")
    if not Path(args.exclusions_file).exists():
        sys.exit(f"{args.exclusions_file} missing (scripts/exclude_exposed_scenes.py)")
    int4 = args.int4_model or cfg["quantize"]["out_dir"]
    if not Path(int4).is_dir():
        sys.exit(f"own checkpoint not found at {int4}; the freeze fingerprints it, so run on the host that has it")
    shutil.copyfile(args.protocol, args.snapshot)
    frozen = build_frozen(cfg, args.split_file, args.exclusions_file, int4, protocol_snapshot=args.snapshot)
    Path(args.frozen).parent.mkdir(parents=True, exist_ok=True)
    Path(args.frozen).write_text(yaml.safe_dump(frozen, sort_keys=False))
    print(f"wrote {args.frozen} and {args.snapshot} (sha256 {frozen['frozen']['protocol_snapshot_sha256'][:12]})\n  commit {rev[:8]}  int4 {frozen['models_frozen']['int4']['checkpoint_fingerprint'][:19]}")


if __name__ == "__main__":
    main()
