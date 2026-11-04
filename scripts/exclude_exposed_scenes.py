"""Record final-split scenes that were exposed by pilot runs, so the
confirmatory population excludes them (PROTOCOL section 5).

    python scripts/exclude_exposed_scenes.py results/pilot/*.jsonl --reason "November 3 pilot slices drew from --split all"

Appends to manifests/scene_exclusions.json (idempotent: a scene is listed
once, with every file that exposed it). The random split itself is never
regenerated.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.provenance import sha256_file  # noqa: E402
from qflip.records import read_records  # noqa: E402
from qflip.split import load_split  # noqa: E402


def update_exclusions(existing: dict | None, split: dict[str, set[str]], exposures: dict[str, set[str]], reason: str, split_file: str) -> dict:
    """exposures: scene_token -> files that contain it. Only final-split scenes are recorded."""
    out = existing or {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split_file_sha256": sha256_file(split_file), "entries": []}
    by_scene = {e["scene_token"]: e for e in out["entries"]}
    for scene, files in sorted(exposures.items()):
        if scene not in split["final"]:
            continue
        e = by_scene.setdefault(scene, {"scene_token": scene, "exposed_by": [], "reason": reason, "date": datetime.now(timezone.utc).date().isoformat()})
        e["exposed_by"] = sorted(set(e["exposed_by"]) | set(files))
    out["entries"] = sorted(by_scene.values(), key=lambda e: e["scene_token"])
    out["n_excluded"] = len(out["entries"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--out", default="manifests/scene_exclusions.json")
    args = ap.parse_args()
    split = load_split(args.split_file)
    exposures: dict[str, set[str]] = {}
    for f in args.files:
        for r in read_records(f):
            exposures.setdefault(r["scene_token"], set()).add(f)
    existing = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else None
    out = update_exclusions(existing, split, exposures, args.reason, args.split_file)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{out['n_excluded']} final scenes excluded -> {args.out}")


if __name__ == "__main__":
    main()
