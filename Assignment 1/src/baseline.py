"""Baseline: stack the B, G, R plates with no alignment."""

import csv
import time

import numpy as np

from common import find_images, load_plate, parse_args, run_dir, save_image, split_plate

DEFAULTS = {
    "images": "*.jpg",  # comma-separated globs or names in data/
}


def colorize(plate):
    """Stack the channels as-is; return (rgb, g_shift, r_shift) like the other scripts."""
    b, g, r = split_plate(plate)
    return np.dstack([r, g, b]), (0, 0), (0, 0)


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        t = time.time()
        rgb, g_d, r_d = colorize(load_plate(path))
        secs = time.time() - t
        save_image(img_dir / "baseline.jpg", rgb)
        print(f"{path.stem}  baseline  {secs:.1f}s")
        rows.append([path.name, *g_d, *r_d, f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "g_dx", "g_dy", "r_dx", "r_dy", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
