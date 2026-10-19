"""Scene split (manifests/scene_split.json) helpers."""

import json
from pathlib import Path

SPLITS = ("calibration", "development", "final")


def load_split(path: str | Path = "manifests/scene_split.json") -> dict[str, set[str]]:
    data = json.loads(Path(path).read_text())
    return {name: set(tokens) for name, tokens in data["split"].items()}


def filter_split(questions, split_name: str | None, split_path: str | Path = "manifests/scene_split.json"):
    """Keep questions whose scene is in the named split; None keeps everything."""
    if split_name is None or split_name == "all":
        return list(questions)
    if split_name not in SPLITS:
        raise ValueError(f"unknown split {split_name!r}; choose from {SPLITS} or all")
    keep = load_split(split_path)[split_name]
    return [q for q in questions if q.scene_token in keep]
