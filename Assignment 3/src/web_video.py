"""Convert videos to small H.264 MP4s with a poster frame for the website.

Usage: python web_video.py VIDEO [VIDEO ...] [--height 720]
Output: website/images/assignment-3/videos/<name>.mp4 and <name>_poster.jpg
"""

import argparse
import subprocess
from pathlib import Path

from extract_frames import TONEMAP, is_hdr

WEB_DIR = Path(__file__).resolve().parents[2] / "website" / "images" / "assignment-3" / "videos"


def convert(video, height):
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    mp4 = WEB_DIR / f"{video.stem}.mp4"
    poster = WEB_DIR / f"{video.stem}_poster.jpg"
    # Scale the short side to `height` so portrait and landscape both work
    vf = (TONEMAP + "," if is_hdr(video) else "") + \
        f"scale='if(gt(iw,ih),-2,{height})':'if(gt(iw,ih),{height},-2)'"

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf, "-an",
         "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-frames:v", "1",
         "-q:v", "3", str(poster)], check=True)
    print(f"{video.name} -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--height", type=int, default=720, help="short side in pixels")
    args = p.parse_args()
    for v in args.videos:
        convert(v, args.height)


if __name__ == "__main__":
    main()
