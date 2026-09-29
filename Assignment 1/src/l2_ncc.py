"""Single-scale exhaustive alignment of Prokudin-Gorskii plates using L2 or NCC."""

import csv
import time

import numpy as np

from common import crop, find_images, load_plate, parse_args, run_dir, save_image, shift, split_plate

DEFAULTS = {
    "images": "*v.jpg",    # comma-separated globs or names in data/
    "window": 20,          # search displacements in [-window, window]
    "crop": 0.1,           # fraction trimmed from each side before scoring
    "metrics": "l2,ncc",   # comma-separated: l2, ncc
}


def l2(a, b):
    """Euclidean distance (L2 norm of the difference), lower is better."""
    return np.sqrt(np.sum((a - b) ** 2))


def ncc(a, b):
    """Zero-mean normalized cross-correlation, higher is better."""
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


def align(moving, ref, window, crop_frac, metric):
    """Return the (dx, dy) that best aligns moving to ref."""
    ref_c = crop(ref, crop_frac)
    best_d, best_score = (0, 0), None
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            cand = crop(shift(moving, (dx, dy)), crop_frac)
            score = l2(cand, ref_c) if metric == "l2" else -ncc(cand, ref_c)
            if best_score is None or score < best_score:
                best_d, best_score = (dx, dy), score
    return best_d


def colorize(plate, window, crop_frac, metric):
    """Align G and R to B and return (rgb, g_shift, r_shift)."""
    b, g, r = split_plate(plate)
    g_d = align(g, b, window, crop_frac, metric)
    r_d = align(r, b, window, crop_frac, metric)
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
            rgb, g_d, r_d = colorize(plate, p["window"], p["crop"], metric)
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
