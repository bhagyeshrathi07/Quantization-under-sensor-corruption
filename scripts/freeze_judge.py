"""Write or check the judge protocol freeze, manifests/judge_frozen.yaml
(PROTOCOL section 7, VQA row; plan section 5.2). Separate from the core
freeze: it binds the judge request settings as sent, the correctness
threshold, and fingerprints of the system line and every rubric template.

    python scripts/freeze_judge.py --threshold 60 --note "40x3 noise: ..."   # after development validation
    python scripts/freeze_judge.py --check                                   # live code and config equal the freeze

Final-split judging (08) and final-split VQA analysis (05 --vqa) refuse
without this file or when the live templates, system line, or settings differ.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.freeze import JUDGE_FROZEN_PATH, FreezeError, build_judge_frozen, check_judge_frozen, load_judge_frozen  # noqa: E402
from qflip.judge import JudgeSettingsError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--frozen", default=str(JUDGE_FROZEN_PATH))
    ap.add_argument("--threshold", type=float, help="correctness threshold decided on development scenes (default judge.correct_threshold)")
    ap.add_argument("--note", action="append", default=[], help="validation note (noise floor, threshold agreement); repeatable")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg.get("judge") or {}

    if args.check:
        diffs = check_judge_frozen(load_judge_frozen(args.frozen), jcfg)
        if diffs:
            print("judge freeze check FAILED:\n  " + "\n  ".join(diffs))
            sys.exit(1)
        print(f"judge freeze check ok: {args.frozen}")
        return

    if Path(args.frozen).exists():
        sys.exit(f"{args.frozen} exists; a changed judge protocol is a new dated freeze (remove the old file deliberately and record it in the PROTOCOL change log)")
    threshold = args.threshold if args.threshold is not None else jcfg.get("correct_threshold")
    try:
        frozen = build_judge_frozen(jcfg, threshold, validation={"notes": args.note})
    except (FreezeError, JudgeSettingsError) as e:
        sys.exit(str(e))
    Path(args.frozen).write_text(yaml.safe_dump(frozen, sort_keys=False))
    print(f"wrote {args.frozen}: model {frozen['judge']['model']}, threshold {threshold}, templates {len(frozen['fingerprints']['templates'])}")


if __name__ == "__main__":
    main()
