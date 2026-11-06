"""Write the report tables T1-T7 as markdown from the producers' JSON. Formats
only; every number comes from 05 --out, 07 --report --out, or 06 summaries.

    python scripts/09_tables.py --compare results/final/compare.json \\
        --profile results/profile/pooled.json \\
        --stability results/stability/*.summary.json \\
        --out results/final/tables.md

Missing inputs leave their table with a note, so the file can be generated at
any stage. T7 lists every harmful example unless --harmful-limit is given.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.tables import render_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", help="JSON written by 05 --out")
    ap.add_argument("--profile", help="JSON written by 07 --report --out")
    ap.add_argument("--stability", nargs="*", default=[], help="*.summary.json written by 06")
    ap.add_argument("--harmful-limit", type=int, help="rows in T7 (default all)")
    ap.add_argument("--out", help="markdown file (default stdout)")
    args = ap.parse_args()
    if not (args.compare or args.profile or args.stability):
        sys.exit("nothing to format: give --compare, --profile and/or --stability")
    compare = json.loads(Path(args.compare).read_text()) if args.compare else None
    profile = json.loads(Path(args.profile).read_text()) if args.profile else None
    stability = [json.loads(Path(p).read_text()) for p in args.stability]
    md = render_all(compare, profile, stability, args.harmful_limit)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
