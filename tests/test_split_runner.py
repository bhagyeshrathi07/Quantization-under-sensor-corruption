import json

import pytest

from qflip.runner import select_stratified
from qflip.split import filter_split


class Q:
    def __init__(self, qid, scene, cell):
        self.question_id, self.scene_token, self.cell = qid, scene, cell


def test_filter_split(tmp_path):
    f = tmp_path / "split.json"
    f.write_text(json.dumps({"split": {"calibration": ["a"], "development": ["b"], "final": ["c"]}}))
    qs = [Q("1", "a", "x"), Q("2", "b", "x"), Q("3", "c", "x")]
    assert [q.question_id for q in filter_split(qs, "development", f)] == ["2"]
    assert len(filter_split(qs, "all", f)) == 3
    assert len(filter_split(qs, None, f)) == 3
    with pytest.raises(ValueError):
        filter_split(qs, "test", f)


def test_select_stratified_round_robin_and_deterministic():
    qs = [Q(f"m{i}", "s", "behavior/MCQ") for i in range(5)] + [Q(f"v{i}", "s", "planning/VQA") for i in range(5)] + [Q("d", "s", None)]
    a = select_stratified(qs, 4, seed=0)
    b = select_stratified(qs, 4, seed=0)
    assert [q.question_id for q in a] == [q.question_id for q in b]
    assert sum(q.cell == "behavior/MCQ" for q in a) == 2
    assert len(select_stratified(qs, 0)) == 10  # dropped cell excluded
