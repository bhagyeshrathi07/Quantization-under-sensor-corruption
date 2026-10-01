"""Reserve disjoint calibration / development / final scene lists from
DriveBench and publish them with per-cell question counts.

This is the proposal's fallback split (section 3, "Scene separation"): if
compatible DriveLM training scenes are used for calibration/development
instead, generate that split separately and keep this file for the record.

    python scripts/04_scene_split.py
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.drivebench import load_questions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.10)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--out", default="manifests/scene_split.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    questions = load_questions(cfg["data"]["annotations"])
    scenes = sorted({q.scene_token for q in questions})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_cal = round(len(scenes) * args.calib_frac)
    n_dev = round(len(scenes) * args.dev_frac)
    split = {
        "calibration": sorted(scenes[:n_cal]),
        "development": sorted(scenes[n_cal : n_cal + n_dev]),
        "final": sorted(scenes[n_cal + n_dev :]),
    }
    assert not (set(split["calibration"]) & set(split["development"]) & set(split["final"]))

    by_scene = defaultdict(list)
    for q in questions:
        by_scene[q.scene_token].append(q)
    counts = {}
    for name, toks in split.items():
        cells = Counter(q.cell or "dropped(prediction MCQ)" for t in toks for q in by_scene[t])
        counts[name] = {"scenes": len(toks), "questions": sum(cells.values()), "per_cell": dict(sorted(cells.items()))}

    out = {
        "source": cfg["data"]["annotations"],
        "seed": args.seed,
        "fractions": {"calibration": args.calib_frac, "development": args.dev_frac},
        "total_scenes": len(scenes),
        "counts": counts,
        "split": split,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "split"}, indent=2))
    print(f"\nwrote {args.out}")
    print("prediction/VQA count in 'final' is the cell to watch (clean total is 61 upstream).")


if __name__ == "__main__":
    main()
