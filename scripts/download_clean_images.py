"""Fetch the 1200 clean nuScenes keyframes DriveBench needs, without a nuScenes account.

Motional mirrors nuScenes on the AWS Open Data registry (bucket motional-nuscenes,
no credentials, CC BY-NC-SA 4.0). The trainval set ships as ten keyframe-only
tarballs, public/v1.0/v1.0-trainvalNN_keyframes.tgz, ~4-6 GB each (~45 GB total),
whose members are `samples/CAM_*/<file>.jpg` plus lidar/radar keyframes. Scenes
are spread across the tarballs with no published index, so this script streams
each tarball through `tar` and keeps only the files the annotations reference,
then stops as soon as nothing is missing. Nothing is written to disk except the
extracted images (~200 MB).

    python scripts/download_clean_images.py            # all ten blobs, or until complete
    python scripts/download_clean_images.py --blobs 3   # one blob, e.g. to test the pipeline
    python scripts/download_clean_images.py --verify    # just report what is missing

Rerunning is safe: files already present are skipped and blobs are only
streamed while something is still missing. Needs `curl` and GNU `tar`.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ANNOTATION = "drivebench-test.json"
# S3 is the origin (ap-northeast-1); CloudFront was ~1.5x faster from California
# in October 2026. Both are anonymous.
BASES = {
    "cloudfront": "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0",
    "s3": "https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0",
}
BLOB_IDS = list(range(1, 11))


def needed_members(data: Path) -> set[str]:
    """Tar member names (`samples/CAM_X/file.jpg`) for every clean image path
    in the annotations, which are written as data/nuscenes/samples/...."""
    records = json.loads((data / ANNOTATION).read_text())
    members = set()
    for r in records:
        for p in r["image_path"].values():
            parts = Path(p).parts
            i = parts.index("samples")
            members.add("/".join(parts[i:]))
    return members


def missing_members(members: set[str], nuscenes_dir: Path) -> list[str]:
    return sorted(m for m in members if not (nuscenes_dir / m).is_file())


def stream_blob(url: str, members: list[str], nuscenes_dir: Path) -> None:
    """curl <url> | tar -xz --files-from=<members> into nuscenes_dir."""
    nuscenes_dir.mkdir(parents=True, exist_ok=True)
    listfile = nuscenes_dir / ".wanted.txt"
    listfile.write_text("".join(m + "\n" for m in members))
    curl = subprocess.Popen(
        ["curl", "-sSfL", "--retry", "5", "--retry-all-errors", url],
        stdout=subprocess.PIPE,
    )
    tar = subprocess.run(
        ["tar", "-xz", "-C", str(nuscenes_dir), "--files-from", str(listfile)],
        stdin=curl.stdout,
        stderr=subprocess.PIPE,
        text=True,
    )
    curl.stdout.close()
    curl_rc = curl.wait()
    listfile.unlink(missing_ok=True)
    if curl_rc != 0:
        sys.exit(f"curl failed (exit {curl_rc}) on {url}")
    # tar exits 2 and prints "Not found in archive" for listed members that live
    # in another blob; that is expected. Anything else is a real error.
    bad = [ln for ln in tar.stderr.splitlines() if "Not found in archive" not in ln
           and "Exiting with failure status" not in ln]
    if tar.returncode not in (0, 2) or bad:
        sys.exit(f"tar failed (exit {tar.returncode}) on {url}:\n" + "\n".join(bad or [tar.stderr]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--blobs", type=int, nargs="*", default=BLOB_IDS, help="trainval blob numbers to stream (1-10)")
    ap.add_argument("--mirror", choices=list(BASES), default="cloudfront")
    ap.add_argument("--urls", nargs="*", help="override blob URLs entirely (testing; file:// works)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    nuscenes_dir = data / "nuscenes"
    if not (data / ANNOTATION).exists():
        sys.exit(f"{data / ANNOTATION} missing; run scripts/download_drivebench.py --json-only first")

    members = needed_members(data)
    missing = missing_members(members, nuscenes_dir)
    print(f"{len(members)} clean images referenced, {len(members) - len(missing)} present, {len(missing)} missing")
    if args.verify or not missing:
        sys.exit(0 if not missing else 1)

    urls = args.urls or [f"{BASES[args.mirror]}/v1.0-trainval{n:02d}_keyframes.tgz" for n in args.blobs]
    for url in urls:
        print(f"streaming {url} ({len(missing)} still missing)", flush=True)
        stream_blob(url, missing, nuscenes_dir)
        missing = missing_members(members, nuscenes_dir)
        print(f"  -> {len(members) - len(missing)} present, {len(missing)} missing", flush=True)
        if not missing:
            break
    if missing:
        print(f"still missing {len(missing)}, e.g. {missing[:3]}")
        sys.exit(1)
    print(f"all {len(members)} clean images present under {nuscenes_dir / 'samples'}")


if __name__ == "__main__":
    main()
