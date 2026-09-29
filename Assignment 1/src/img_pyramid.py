"""Coarse-to-fine image pyramid alignment for full-size Prokudin-Gorskii plates."""

import csv
import time

import numpy as np

from common import find_images, load_plate, parse_args, run_dir, save_image, shift, split_plate
from l2_ncc import align

DEFAULTS = {
    "images": "*.jpg",     # comma-separated globs or names in data/
    "window": 25,          # search radius at the coarsest level
    "refine": 2,           # search radius at each finer level
    "coarse_size": 500,    # halve until the smaller channel side is at most this (px)
    "crop": 0.2,           # fraction trimmed from each side before scoring
    "metrics": "l2,ncc",   # comma-separated: l2, ncc
}


def downsample(img):
    """Halve the resolution by averaging 2x2 blocks (box filter against aliasing)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def pyramid_align(moving, ref, window, refine, crop_frac, metric, coarse_size):
    """Return the (dx, dy) that best aligns moving to ref, searching coarse to fine."""
    if min(ref.shape) <= coarse_size:
        return align(moving, ref, window, crop_frac, metric)
    dx, dy = pyramid_align(downsample(moving), downsample(ref), window, refine,
                           crop_frac, metric, coarse_size)
    dx, dy = 2 * dx, 2 * dy
    rdx, rdy = align(shift(moving, (dx, dy)), ref, refine, crop_frac, metric)
    return dx + rdx, dy + rdy


def colorize(plate, window, refine, crop_frac, metric, coarse_size):
    """Align G and R to B with the pyramid and return (rgb, g_shift, r_shift)."""
    b, g, r = split_plate(plate)
    g_d = pyramid_align(g, b, window, refine, crop_frac, metric, coarse_size)
    r_d = pyramid_align(r, b, window, refine, crop_frac, metric, coarse_size)
    return np.dstack([shift(r, r_d), shift(g, g_d), b]), g_d, r_d


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        plate = load_plate(path)
        b, g, r = split_plate(plate)
        save_image(img_dir / "unaligned.jpg", np.dstack([r, g, b]))
        for metric in p["metrics"].split(","):
            t = time.time()
            rgb, g_d, r_d = colorize(plate, p["window"], p["refine"], p["crop"], metric,
                                     p["coarse_size"])
            secs = time.time() - t
            save_image(img_dir / f"{metric}.jpg", rgb)
            print(f"{path.stem}  {metric}  G (dx, dy)={g_d}  R={r_d}  {secs:.1f}s")
            rows.append([path.name, metric, *g_d, *r_d, f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "metric", "g_dx", "g_dy", "r_dx", "r_dy", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
