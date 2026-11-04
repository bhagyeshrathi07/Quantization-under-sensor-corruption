"""Scene split (manifests/scene_split.json) and exclusions
(manifests/scene_exclusions.json) helpers.

"final" is the confirmatory population: final-split scenes minus the
exclusion list (scenes exposed by pilot runs). "final_exposed" is the excluded
part, for exploratory use only. The random split is never regenerated; the
exclusion list is additive and carries its own provenance.
"""

import json
from pathlib import Path

SPLITS = ("calibration", "development", "final")
SPLIT_NAMES = SPLITS + ("final_exposed",)
DEFAULT_SPLIT_FILE = "manifests/scene_split.json"
DEFAULT_EXCLUSIONS_FILE = "manifests/scene_exclusions.json"


def load_split(path: str | Path = DEFAULT_SPLIT_FILE) -> dict[str, set[str]]:
    data = json.loads(Path(path).read_text())
    return {name: set(tokens) for name, tokens in data["split"].items()}


def load_exclusions(path: str | Path | None = DEFAULT_EXCLUSIONS_FILE) -> set[str]:
    """Excluded scene tokens; empty set when the file does not exist."""
    if path is None or not Path(path).exists():
        return set()
    data = json.loads(Path(path).read_text())
    return {e["scene_token"] for e in data["entries"]}


def scenes_for(split_name: str, split_path: str | Path = DEFAULT_SPLIT_FILE, exclusions_path: str | Path | None = DEFAULT_EXCLUSIONS_FILE) -> set[str]:
    split = load_split(split_path)
    if split_name == "final":
        return split["final"] - load_exclusions(exclusions_path)
    if split_name == "final_exposed":
        return split["final"] & load_exclusions(exclusions_path)
    if split_name in SPLITS:
        return split[split_name]
    raise ValueError(f"unknown split {split_name!r}; choose from {SPLIT_NAMES} or all")


def filter_split(questions, split_name: str | None, split_path: str | Path = DEFAULT_SPLIT_FILE, exclusions_path: str | Path | None = DEFAULT_EXCLUSIONS_FILE):
    """Keep questions whose scene is in the named split; None or "all" keeps everything."""
    if split_name is None or split_name == "all":
        return list(questions)
    keep = scenes_for(split_name, split_path, exclusions_path)
    return [q for q in questions if q.scene_token in keep]
