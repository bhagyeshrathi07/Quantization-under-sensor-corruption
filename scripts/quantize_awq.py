"""Build our own INT4 checkpoint: AWQ W4A16 from the pinned base revision with
llm-compressor, calibrated on calibration-split DriveBench questions rendered
exactly as the runs render them (same system prompt, views, max_pixels).

    uv pip install -e ".[quant]"                       # GPU host only
    python scripts/quantize_awq.py --dry-run           # calibration set + manifest, no model, no GPU
    python scripts/quantize_awq.py                     # ~30-60 min on the 4090; needs ~40 GB host RAM
    python scripts/quantize_awq.py --retain "re:.*language_model\\.layers\\.(3|7)\\..*"   # mitigation variant

Output: a compressed-tensors checkpoint under quantize.out_dir (served by
01_serve.sh with INT4_MODEL=<dir>) and manifests/quantize_<name>.json with
the recipe, tool versions, and the calibration question list.

Why not the GPTQ example from llm-compressor: the official Qwen checkpoint is
AWQ, so our checkpoint keeps the algorithm and only changes the calibration
data, which keeps the comparison about calibration domain, not method.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.config import load_config  # noqa: E402
from qflip.drivebench import CAMERA_ORDER, condition_image_path, load_image_bytes, load_questions, rewrite_camera_sentence  # noqa: E402
from qflip.runner import read_system_prompt, select_stratified  # noqa: E402
from qflip.split import filter_split  # noqa: E402


def calibration_questions(cfg: dict, split_file: str, n_samples: int, seed: int = 0):
    """Eligible calibration-split questions, stratified like the runs."""
    qs = filter_split(load_questions(cfg["data"]["annotations"]), "calibration", split_file)
    return select_stratified(qs, n_samples, seed=seed)


def calibration_sample(q, cfg: dict, system_prompt: str) -> tuple[list[dict], list[str]]:
    """Chat messages with image placeholders (processor.apply_chat_template
    format) plus the image file paths, in CAMERA_ORDER, as the runs send them."""
    content = [{"type": "image"} for _ in q.views]
    text = rewrite_camera_sentence(system_prompt, q.views) + q.question
    if q.is_mcq and cfg["prompt"]["mcq_suffix"]:
        text += cfg["prompt"]["mcq_suffix"]
    content.append({"type": "text", "text": text})
    paths = [q.image_path[cam] for cam in CAMERA_ORDER if cam in q.image_path]
    return [{"role": "user", "content": content}], paths


def build_recipe(qcfg: dict, retain: list[str]) -> dict:
    """Serializable description of the recipe; the modifier objects are built
    from it in quantize() so this function needs no llm-compressor import."""
    symmetric = not qcfg["scheme"].endswith("_ASYM")
    return {
        "algorithm": "AWQ (llm-compressor AWQModifier + QuantizationModifier)",
        "scheme": qcfg["scheme"],
        "targets": ["Linear"],
        "ignore": list(qcfg["ignore"]) + list(retain),
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": symmetric,
            "group_size": int(qcfg["group_size"]),
            "strategy": "group",
            "dynamic": False,
        },
        "retained_bf16": list(retain),
    }


def manifest_name(qcfg: dict, retain: list[str]) -> str:
    base = Path(qcfg["out_dir"]).name
    return f"quantize_{base}" + ("_retain" if retain else "")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def tool_versions() -> dict:
    out = {}
    for name in ("llmcompressor", "compressed_tensors", "transformers", "torch", "datasets"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "not installed"
    return out


def quantize(cfg: dict, questions, out_dir: Path, recipe: dict, condition: str, system_prompt: str) -> None:
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    qcfg = cfg["quantize"]
    model_id, rev = qcfg["base_model"], qcfg["revision"]
    print(f"[quantize] loading {model_id}@{rev[:8]} on CPU (bf16)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16, device_map=None)
    processor = AutoProcessor.from_pretrained(model_id, revision=rev, max_pixels=int(qcfg["max_pixels"]))

    data_root = cfg["data"]["root"]
    resize_to = cfg["images"]["resize_to"]
    noimage = tuple(cfg["images"]["noimage_size"])
    cond_dir = cfg["data"]["conditions"][condition]
    def render(q):
        messages, paths = calibration_sample(q, cfg, system_prompt)
        images = []
        for p in paths:
            src = None if condition == "NoImage" else condition_image_path(p, condition, cond_dir)
            images.append(Image.open(io.BytesIO(load_image_bytes(src, data_root, resize_to, noimage))).convert("RGB"))
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=images, padding=False, max_length=int(qcfg["max_seq_len"]), truncation=True)
        return {k: (v[0] if k in ("input_ids", "attention_mask") else v) for k, v in enc.items()}

    print(f"[quantize] rendering {len(questions)} calibration samples ({condition})")
    ds = Dataset.from_list([render(q) for q in questions])

    def collate(batch):
        assert len(batch) == 1
        return {k: (torch.tensor(v, dtype=torch.bfloat16) if k == "pixel_values" else torch.tensor(v)) for k, v in batch[0].items()}

    modifiers = [
        AWQModifier(),
        QuantizationModifier(
            ignore=recipe["ignore"],
            config_groups={"group_0": {"targets": recipe["targets"], "weights": recipe["weights"]}},
        ),
    ]
    t0 = time.time()
    oneshot(
        model=model,
        processor=processor,
        recipe=modifiers,
        dataset=ds,
        max_seq_length=int(qcfg["max_seq_len"]),
        num_calibration_samples=len(questions),
        data_collator=collate,
        sequential_targets=["Qwen2_5_VLDecoderLayer"],
    )
    print(f"[quantize] oneshot done in {time.time() - t0:.0f}s; saving {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    processor.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split-file", default="manifests/scene_split.json")
    ap.add_argument("--out-dir", help="override quantize.out_dir")
    ap.add_argument("--n-samples", type=int, help="override quantize.n_samples (0 = all eligible calibration questions)")
    ap.add_argument("--retain", nargs="*", default=[], help="extra ignore patterns: decoder modules kept at BF16 (mitigation)")
    ap.add_argument("--manifest-dir", default="manifests")
    ap.add_argument("--dry-run", action="store_true", help="write the manifest and calibration list only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    qcfg = cfg["quantize"]
    out_dir = Path(args.out_dir or qcfg["out_dir"])
    n = qcfg["n_samples"] if args.n_samples is None else args.n_samples
    questions = calibration_questions(cfg, args.split_file, n)
    recipe = build_recipe(qcfg, args.retain)
    system_prompt = read_system_prompt(cfg)
    manifest = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_model": qcfg["base_model"],
        "revision": qcfg["revision"],
        "out_dir": str(out_dir),
        "recipe": recipe,
        "calibration": {
            "split": "calibration",
            "split_file": args.split_file,
            "condition": qcfg["condition"],
            "n_samples": len(questions),
            "max_seq_len": qcfg["max_seq_len"],
            "max_pixels": qcfg["max_pixels"],
            "scenes": sorted({q.scene_token for q in questions}),
            "question_ids": [q.question_id for q in questions],
            "per_cell": {c: sum(q.cell == c for q in questions) for c in sorted({q.cell for q in questions})},
        },
        "tool_versions": tool_versions(),
        "dry_run": args.dry_run,
    }
    mpath = Path(args.manifest_dir) / f"{manifest_name(qcfg, args.retain)}.json"
    write_manifest(mpath, manifest)
    print(f"[quantize] {len(questions)} calibration questions from {len(manifest['calibration']['scenes'])} scenes: {manifest['calibration']['per_cell']}")
    print(f"[quantize] recipe: {recipe['scheme']} g{recipe['weights']['group_size']} ignore={recipe['ignore']}")
    print(f"[quantize] manifest -> {mpath}")
    if args.dry_run:
        return
    quantize(cfg, questions, out_dir, recipe, qcfg["condition"], system_prompt)
    manifest["dry_run"] = False
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["tool_versions"] = tool_versions()
    write_manifest(mpath, manifest)
    print(f"[quantize] done. serve with:\n  INT4_MODEL={out_dir} bash scripts/01_serve.sh int4")


if __name__ == "__main__":
    main()
