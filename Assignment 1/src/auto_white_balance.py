"""Auto white balance: estimate the illuminant with shades of gray and scale it to neutral."""

import csv
import time

import numpy as np

from auto_crop import best_cropped
from common import find_images, load_plate, parse_args, run_dir, save_image

DEFAULTS = {
    "images": "*.jpg",  # comma-separated globs or names in data/
    "p": 6.0,           # Minkowski norm: 1 is gray world, large values approach white patch
}


def white_balance(rgb, p):
    """Return (image, gains): each channel is scaled so the estimated illuminant becomes gray."""
    illum = np.mean(rgb.reshape(-1, 3) ** p, axis=0) ** (1 / p)
    gains = illum.mean() / illum
    return np.clip(rgb * gains, 0, 1), gains


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        cropped = best_cropped(load_plate(path))
        t = time.time()
        rgb, gains = white_balance(cropped, p["p"])
        secs = time.time() - t
        save_image(img_dir / "white_balance.jpg", rgb)
        print(f"{path.stem}  gains R {gains[0]:.3f} G {gains[1]:.3f} B {gains[2]:.3f}  {secs:.1f}s")
        rows.append([path.name, *(f"{g:.4f}" for g in gains), f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "gain_r", "gain_g", "gain_b", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
