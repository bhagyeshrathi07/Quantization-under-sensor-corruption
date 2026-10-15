"""Download DriveBench into data/ in the layout upstream docs/DATA_PREPAER.md expects.

  1. Annotation JSONs from HF dataset drive-bench/arena (~32 MB) -> data/*.json
  2. Image archive DriveBench.zip from Google Drive (7.2 GB): the 15 corruption
     types only, no clean images -> data/DriveBench.zip
  3. Extract and arrange as data/corruption/<Type>/CAM_*/
  4. Verify every clean image path in drivebench-test.json exists under
     data/nuscenes/samples/ (clean images come from nuScenes itself, not the zip)

    python scripts/download_drivebench.py             # everything
    python scripts/download_drivebench.py --json-only # step 1 only (enough for 04_scene_split.py)
    python scripts/download_drivebench.py --verify    # step 4 only

Every step is skipped when its output already exists, so rerunning is safe.
Needs: uv pip install -e ".[data]"   (huggingface_hub, gdown) and `unzip`.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HF_REPO = "drive-bench/arena"
GDRIVE_ID = "1_MqbX1oXH9S55eC0r_rZvvaoAD5GVOyW"  # DriveBench.zip, from upstream docs/DATA_PREPAER.md
ANNOTATION = "drivebench-test.json"


def download_json(data: Path) -> None:
    if (data / ANNOTATION).exists():
        print(f"[json] {data / ANNOTATION} exists, skipping")
        return
    from huggingface_hub import snapshot_download

    print(f"[json] downloading {HF_REPO} -> {data}")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=data, allow_patterns=["*.json"])
    print(f"[json] files: {sorted(p.name for p in data.glob('*.json'))}")


def download_zip(data: Path) -> Path:
    zip_path = data / "DriveBench.zip"
    if zip_path.exists():
        print(f"[zip] {zip_path} exists ({zip_path.stat().st_size / 1e9:.1f} GB), skipping download")
        return zip_path
    import gdown

    print("[zip] downloading DriveBench.zip from Google Drive (large; resumable on rerun)")
    out = gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False, resume=True)
    if not out:
        sys.exit(
            "[zip] gdown failed. Google Drive may be rate-limiting large downloads; "
            "download DriveBench.zip in a browser from "
            f"https://drive.google.com/file/d/{GDRIVE_ID}/view and place it at {zip_path}, then rerun."
        )
    return zip_path


def extract_and_arrange(data: Path, zip_path: Path) -> None:
    """DriveBench.zip (7.2 GB, verified October 2026) contains ONLY the 15
    corruption types as DriveBench/<Type>/CAM_*/<nuScenes file name>.jpg.
    Clean images are not included; they are nuScenes keyframes and must be
    obtained from nuscenes.org (registration) into data/nuscenes/samples/."""
    if (data / "corruption").is_dir():
        print("[extract] data/corruption exists, skipping")
        return
    tmp = data / "_extract"
    if not tmp.exists():
        print(f"[extract] unzipping {zip_path} -> {tmp} (7.5 GB extracted)")
        tmp.mkdir(parents=True)
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(tmp)], check=True)

    # Find the directory whose children are corruption types (each holding CAM_* dirs).
    def is_type_dir(p: Path) -> bool:
        return p.is_dir() and (p / "CAM_FRONT").is_dir()

    root = next((p for p in [tmp, *tmp.rglob("*")] if p.is_dir() and any(is_type_dir(c) for c in p.iterdir())), None)
    if root is None:
        print("[extract] could not find <Type>/CAM_FRONT directories in the archive. Top of tree:")
        for p in sorted(tmp.rglob("*"))[:40]:
            print("   ", p.relative_to(tmp))
        sys.exit("[extract] arrange data/corruption/<Type>/CAM_*/ by hand, then rerun with --verify")

    shutil.move(str(root), str(data / "corruption"))
    print(f"[extract] corruption types: {sorted(p.name for p in (data / 'corruption').iterdir() if p.is_dir())}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("[extract] done; DriveBench.zip kept (delete it to reclaim 7.2 GB)")
    if not (data / "nuscenes" / "samples").is_dir():
        print("[extract] NOTE: clean images are not in the zip. Put nuScenes keyframes at data/nuscenes/samples/"
              " (symlink to a nuScenes checkout, or download from nuscenes.org) before running the clean condition.")


def verify(data: Path) -> None:
    ann = data / ANNOTATION
    if not ann.exists():
        sys.exit(f"[verify] {ann} missing; run without --verify first")
    records = json.load(open(ann))
    paths = {p for r in records for p in r["image_path"].values()}
    missing = [p for p in paths if not (data / Path(p).relative_to("data")).exists()]
    print(f"[verify] {len(records)} questions, {len({r['scene_token'] for r in records})} scenes, {len(paths)} unique clean images")
    print(f"[verify] cells: {dict(sorted(Counter((r['question_type'], tuple(r['tag'])) for r in records).items()))}")
    if missing:
        print(f"[verify] MISSING {len(missing)} clean images, e.g. {missing[:3]}")
    else:
        print("[verify] all clean images present")
    corr = data / "corruption"
    if corr.is_dir():
        for d in sorted(p for p in corr.iterdir() if p.is_dir()):
            n = sum(1 for _ in d.rglob("*.jpg"))
            print(f"[verify] corruption/{d.name}: {n} jpg")
    else:
        print("[verify] no data/corruption directory yet")
    if missing:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify(data)
        return
    download_json(data)
    if args.json_only:
        return
    free_gb = shutil.disk_usage(data).free / 1e9
    print(f"[zip] free disk at {data}: {free_gb:.0f} GB (need room for the zip plus its extracted copy)")
    zip_path = download_zip(data)
    extract_and_arrange(data, zip_path)
    verify(data)


if __name__ == "__main__":
    main()
