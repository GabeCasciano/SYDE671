"""Extract frames from videos and drop near-duplicates.

Usage: python extract_frames.py VIDEO [VIDEO ...] [--fps 2] [--diff 0.04]
Frames go to data/frames/<video name>/.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# iPhone HDR video (HLG or PQ) looks washed out unless tone mapped to SDR
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def is_hdr(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_transfer", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [{}])
    return streams[0].get("color_transfer") in ("arib-std-b67", "smpte2084")


def thumb(path):
    """Small blurred grayscale copy in [0, 1] for comparing frames."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    small = cv2.resize(img, (128, round(128 * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32) / 255.0


def extract(video, fps, diff):
    out_dir = DATA_DIR / "frames" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}" + ("," + TONEMAP if is_hdr(video) else "")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf,
             "-q:v", "2", f"{tmp}/%05d.jpg"], check=True)
        frames = sorted(Path(tmp).glob("*.jpg"))

        last, kept = None, 0
        for f in frames:
            t = thumb(f)
            if last is None or np.abs(t - last).mean() > diff:
                kept += 1
                shutil.move(f, out_dir / f"{video.stem}_{kept:04d}.jpg")
                last = t

    print(f"{video.name}: kept {kept} of {len(frames)} frames -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    p.add_argument("--diff", type=float, default=0.04,
                   help="min mean abs difference from the last kept frame (0 to 1)")
    args = p.parse_args()
    for v in args.videos:
        extract(v, args.fps, args.diff)


if __name__ == "__main__":
    main()
