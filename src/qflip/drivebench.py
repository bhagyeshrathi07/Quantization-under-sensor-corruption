"""DriveBench annotation handling.

Mirrors the released DriveBench code where behavior must match it:
- view selection from `<c1,CAM_BACK,x,y>` tags (tools/preprocess.py)
- camera sentence rewriting in the system prompt (inference/utils.py)
- corruption image path mapping and the NoImage control (inference/llava1.5.py)
- task/format routing by question_type and tag (evaluate/eval.py)
- MCQ answer normalization (evaluate/utils.py)

Annotation record fields: scene_token, frame_token, question_type, question,
answer (ground truth), tag (list of int), image_path (dict camera -> path).
"""

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Same regex as upstream tools/preprocess.py
_CAMERA_TAG_RE = re.compile(r"<[^,]+,([^,]+),[^>]+>")
_CAMERA_SENTENCE_RE = re.compile(
    r"You are provided with up to six camera images in the sequence "
    r"\[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT\]\."
)

# Same patterns as upstream evaluate/utils.py preprocess_answer. Note: the
# pattern for "A" also matches the article "a" in free text; this is the
# released scorer's behavior and is reproduced here for scorer-compatible
# accuracy. See normalize_mcq_strict for the stricter version we also record.
_CHOICE_PATTERNS = {
    "A": r"\b(a|option a)\b",
    "B": r"\b(b|option b)\b",
    "C": r"\b(c|option c)\b",
    "D": r"\b(d|option d)\b",
}


@dataclass
class Question:
    question_id: str
    scene_token: str
    frame_token: str
    question_type: str
    question: str
    answer: str
    tag: list[int]
    image_path: dict[str, str]  # only the selected views, in CAMERA_ORDER
    raw: dict = field(repr=False)

    @property
    def views(self) -> list[str]:
        return list(self.image_path.keys())

    @property
    def n_views(self) -> int:
        return len(self.image_path)

    @property
    def cell(self) -> str | None:
        return route_cell(self.question_type, self.tag)

    @property
    def is_mcq(self) -> bool:
        return (self.cell or "").endswith("MCQ")


def make_question_id(rec: dict) -> str:
    h = hashlib.sha1(rec["question"].encode()).hexdigest()[:10]
    return f"{rec['scene_token'][:8]}-{rec['frame_token'][:8]}-{h}"


def select_views(question: str, image_path: dict[str, str]) -> dict[str, str]:
    """Keep only cameras referenced in the question; keep all if none referenced.
    Output is ordered by CAMERA_ORDER, matching the upstream prompt convention."""
    referenced = set(_CAMERA_TAG_RE.findall(question))
    if referenced:
        kept = {cam: p for cam, p in image_path.items() if cam in referenced}
    else:
        kept = dict(image_path)
    return {cam: kept[cam] for cam in CAMERA_ORDER if cam in kept}


def route_cell(question_type: str, tag: list[int]) -> str | None:
    """Return the DriveBench evaluation cell, or None if the released scorer
    drops it (prediction MCQ) or would raise on it."""
    if question_type == "perception":
        if 2 in tag:
            return "perception/VQA"
        if 0 in tag:
            return "perception/MCQ"
    elif question_type == "prediction":
        if 3 in tag:
            return "prediction/VQA"
        if 0 in tag:
            return None  # upstream: `pass`
    elif question_type == "planning":
        if 1 in tag:
            return "planning/VQA"
    elif question_type == "behavior":
        if 0 in tag:
            return "behavior/MCQ"
    return None


def load_questions(path: str | Path) -> list[Question]:
    with open(path) as f:
        records = json.load(f)
    out = []
    for rec in records:
        out.append(
            Question(
                question_id=make_question_id(rec),
                scene_token=rec["scene_token"],
                frame_token=rec["frame_token"],
                question_type=rec["question_type"],
                question=rec["question"],
                answer=rec["answer"],
                tag=list(rec["tag"]),
                image_path=select_views(rec["question"], rec["image_path"]),
                raw=rec,
            )
        )
    return out


def rewrite_camera_sentence(system_prompt: str, views: list[str]) -> str:
    """Upstream inference/utils.py replace_system_prompt."""
    ordered = [c for c in CAMERA_ORDER if c in views]
    if not ordered:
        raise ValueError("No valid camera images provided.")
    cams = ", ".join(ordered)
    if len(ordered) == 1:
        new = f"You are provided with a single camera image: [{cams}]."
    else:
        new = f"You are provided with {len(ordered)} camera images in the sequence [{cams}]."
    updated, n = _CAMERA_SENTENCE_RE.subn(new, system_prompt)
    if n == 0:
        raise ValueError("Camera sentence not found in system prompt.")
    return updated


def condition_image_path(path: str, condition: str, condition_dir: str | None) -> str:
    """Upstream: img_path.replace('nuscenes/samples', f'corruption/{corruption}')."""
    if condition in ("clean", "NoImage") or condition_dir is None:
        return path
    return path.replace("nuscenes/samples", f"corruption/{condition_dir}")


def resolve_image_path(path: str, data_root: str | Path) -> Path:
    """Annotation paths are written as `data/nuscenes/samples/...` relative to
    the upstream repo root. Map that onto our data root, which is the `data/`
    directory itself, so the leading `data/` is dropped."""
    p = Path(path)
    if p.parts and p.parts[0] == "data":
        p = Path(*p.parts[1:])
    return Path(data_root) / p


def load_image_bytes(
    path: str | None,
    data_root: str | Path,
    resize_to: tuple[int, int] | None,
    noimage_size: tuple[int, int] = (224, 224),
) -> bytes:
    """Return JPEG/PNG bytes for one view. path=None yields the NoImage black image."""
    if path is None:
        img = Image.new("RGB", tuple(noimage_size), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    full = resolve_image_path(path, data_root)
    if resize_to is None:
        return full.read_bytes()
    img = Image.open(full).convert("RGB").resize(tuple(resize_to))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def to_data_url(image_bytes: bytes) -> str:
    mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def build_messages(
    q: Question,
    system_prompt: str,
    condition: str,
    condition_dir: str | None,
    data_root: str | Path,
    resize_to: tuple[int, int] | None,
    noimage_size: tuple[int, int],
    mcq_suffix: str,
) -> list[dict]:
    """OpenAI-style chat messages: images first (in CAMERA_ORDER), then the
    system prompt with the rewritten camera sentence, then the question.
    Mirrors the upstream prompt layout `<image>...<system><question>`."""
    content = []
    for cam in q.views:
        src = None if condition == "NoImage" else condition_image_path(q.image_path[cam], condition, condition_dir)
        img_bytes = load_image_bytes(src, data_root, resize_to, noimage_size)
        content.append({"type": "image_url", "image_url": {"url": to_data_url(img_bytes)}})
    text = rewrite_camera_sentence(system_prompt, q.views) + q.question
    if q.is_mcq and mcq_suffix:
        text += mcq_suffix
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def normalize_mcq(answer: str) -> str:
    """Released scorer behavior (evaluate/utils.py preprocess_answer)."""
    a = answer.strip().lower()
    for choice, pattern in _CHOICE_PATTERNS.items():
        if re.search(pattern, a):
            return choice
    return ""


def normalize_mcq_strict(answer: str) -> str:
    """Stricter alternative: option letter must appear at the start of the answer,
    optionally followed by punctuation. Recorded alongside the released rule."""
    m = re.match(r"\s*(?:option\s+)?([abcd])\b", answer.strip().lower())
    return m.group(1).upper() if m else ""
