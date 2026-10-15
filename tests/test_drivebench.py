from pathlib import Path

from qflip.client import option_margin
from qflip.drivebench import (
    condition_image_path,
    normalize_mcq,
    normalize_mcq_strict,
    rewrite_camera_sentence,
    route_cell,
    select_views,
)

SIX = {
    "CAM_FRONT": "data/nuscenes/samples/CAM_FRONT/a.jpg",
    "CAM_FRONT_LEFT": "data/nuscenes/samples/CAM_FRONT_LEFT/b.jpg",
    "CAM_FRONT_RIGHT": "data/nuscenes/samples/CAM_FRONT_RIGHT/c.jpg",
    "CAM_BACK": "data/nuscenes/samples/CAM_BACK/d.jpg",
    "CAM_BACK_LEFT": "data/nuscenes/samples/CAM_BACK_LEFT/e.jpg",
    "CAM_BACK_RIGHT": "data/nuscenes/samples/CAM_BACK_RIGHT/f.jpg",
}
PROMPT = (
    "Intro. You are provided with up to six camera images in the sequence "
    "[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]. Tail."
)


def test_select_views_keeps_referenced_in_camera_order():
    q = "Status of <c1,CAM_BACK,0.5,0.6> and <c2,CAM_FRONT,0.4,0.5>?"
    assert list(select_views(q, SIX)) == ["CAM_FRONT", "CAM_BACK"]


def test_select_views_keeps_all_when_unreferenced():
    assert list(select_views("What are the important objects?", SIX)) == list(SIX)


def test_route_cell():
    assert route_cell("perception", [0]) == "perception/MCQ"
    assert route_cell("perception", [2]) == "perception/VQA"
    assert route_cell("prediction", [3]) == "prediction/VQA"
    assert route_cell("prediction", [0]) is None  # dropped upstream
    assert route_cell("planning", [1]) == "planning/VQA"
    assert route_cell("behavior", [0]) == "behavior/MCQ"


def test_rewrite_camera_sentence():
    assert "a single camera image: [CAM_BACK]." in rewrite_camera_sentence(PROMPT, ["CAM_BACK"])
    assert "2 camera images in the sequence [CAM_FRONT, CAM_BACK]." in rewrite_camera_sentence(PROMPT, ["CAM_BACK", "CAM_FRONT"])


def test_resolve_image_path_drops_leading_data():
    from qflip.drivebench import resolve_image_path

    assert resolve_image_path("data/nuscenes/samples/CAM_FRONT/a.jpg", "data") == Path("data/nuscenes/samples/CAM_FRONT/a.jpg")
    assert resolve_image_path("data/corruption/Fog/CAM_FRONT/a.jpg", "/x/data") == Path("/x/data/corruption/Fog/CAM_FRONT/a.jpg")
    assert resolve_image_path("nuscenes/samples/CAM_FRONT/a.jpg", "data") == Path("data/nuscenes/samples/CAM_FRONT/a.jpg")


def test_condition_image_path():
    p = "data/nuscenes/samples/CAM_FRONT/a.jpg"
    assert condition_image_path(p, "clean", None) == p
    assert condition_image_path(p, "NoImage", None) == p
    assert condition_image_path(p, "Fog", "Fog") == "data/corruption/Fog/CAM_FRONT/a.jpg"


def test_normalize_mcq_matches_released_rule():
    assert normalize_mcq("B. Turn left.") == "B"
    assert normalize_mcq("The answer is option c") == "C"
    # released rule matches the article "a"; strict rule does not
    assert normalize_mcq("There is a car turning, so B.") == "A"
    assert normalize_mcq_strict("There is a car turning, so B.") == ""
    assert normalize_mcq_strict("B. Turn left.") == "B"


def test_option_margin():
    top = [{"token": "B", "logprob": -0.1}, {"token": " A", "logprob": -2.5}, {"token": "The", "logprob": -3.0}]
    letter, margin, probs = option_margin(top)
    assert letter == "B"
    assert set(probs) == {"A", "B"}
    assert 0.8 < margin < 0.9
    assert option_margin([{"token": "The", "logprob": -0.1}]) == (None, None, {})
    assert option_margin(None) == (None, None, {})
