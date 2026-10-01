"""JSONL result records with resume support."""

import json
from pathlib import Path


def load_done_ids(path: str | Path) -> set[str]:
    """IDs of successfully completed records. Errored records are retried on rerun."""
    p = Path(path)
    if not p.exists():
        return set()
    ids = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if not rec.get("error"):
                    ids.add(rec["question_id"])
    return ids


def append_record(path: str | Path, rec: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_records(path: str | Path) -> list[dict]:
    """All records; when a question_id appears more than once (retry after an
    error), the last record wins."""
    by_id: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_id[rec["question_id"]] = rec
    return list(by_id.values())
