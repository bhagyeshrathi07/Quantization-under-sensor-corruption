"""Recompute MCQ fields in result JSONL files from the stored raw answers.

Use after a change to the normalization rules so old runs match new runs
without re-querying the model.

    python scripts/rescore.py results/pilot/*.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.drivebench import normalize_mcq, normalize_mcq_strict  # noqa: E402
from qflip.records import read_records  # noqa: E402


def rescore(rec: dict) -> dict:
    if not (rec.get("cell") or "").endswith("MCQ"):
        return rec
    gt = normalize_mcq(rec["gt"])
    strict = normalize_mcq_strict(rec["pred"])
    released = normalize_mcq(rec["pred"])
    rec.update(
        {
            "mcq_pred": released,
            "mcq_pred_strict": strict,
            "mcq_gt": gt,
            "mcq_correct": strict == gt and strict != "",
            "mcq_correct_released": released == gt and released != "",
        }
    )
    return rec


def main():
    for path in sys.argv[1:]:
        p = Path(path)
        recs = [rescore(r) for r in read_records(p)]
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        mcq = [r for r in recs if (r.get("cell") or "").endswith("MCQ")]
        if mcq:
            s = sum(r["mcq_correct"] for r in mcq)
            rl = sum(r["mcq_correct_released"] for r in mcq)
            print(f"{p}: {len(mcq)} MCQ  strict {s}/{len(mcq)}  released-rule {rl}/{len(mcq)}")
        else:
            print(f"{p}: no MCQ records")


if __name__ == "__main__":
    main()
