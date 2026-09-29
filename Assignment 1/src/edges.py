"""Pyramid alignment on edge maps (Sobel or Canny) instead of raw intensities."""

import csv
import time

import numpy as np
from scipy import ndimage
from skimage import feature

from common import find_images, load_plate, parse_args, run_dir, save_image, shift, split_plate
from img_pyramid import pyramid_align

DEFAULTS = {
    "images": "*.jpg",     # comma-separated globs or names in data/
    "detector": "sobel",   # sobel or canny
    "sigma": 2.0,          # Gaussian smoothing before edge detection (px)
    "low": 0.8,            # canny only: low hysteresis threshold, quantile of gradient magnitude
    "high": 0.9,           # canny only: high hysteresis threshold, quantile of gradient magnitude
    "window": 25,          # search radius at the coarsest level
    "refine": 2,           # search radius at each finer level
    "coarse_size": 500,    # halve until the smaller channel side is at most this (px)
    "crop": 0.2,           # fraction trimmed from each side before scoring
    "metrics": "l2,ncc",   # comma-separated: l2, ncc
}


def sobel_edges(img, sigma):
    """Gradient magnitude of the smoothed image."""
    s = ndimage.gaussian_filter(img, sigma)
    return np.hypot(ndimage.sobel(s, axis=0), ndimage.sobel(s, axis=1))


def canny_edges(img, sigma, low, high):
    """Binary Canny edge map as float32."""
    edges = feature.canny(img, sigma=sigma, low_threshold=low, high_threshold=high, use_quantiles=True)
    return edges.astype(np.float32)


def detect_edges(img, detector, sigma, low, high):
    if detector == "sobel":
        return sobel_edges(img, sigma)
    return canny_edges(img, sigma, low, high)


def colorize(plate, edges, window, refine, crop_frac, metric, coarse_size):
    """Align G and R to B using (b, g, r) edge maps; return (rgb, g_shift, r_shift)."""
    b, g, r = split_plate(plate)
    eb, eg, er = edges
    g_d = pyramid_align(eg, eb, window, refine, crop_frac, metric, coarse_size)
    r_d = pyramid_align(er, eb, window, refine, crop_frac, metric, coarse_size)
    return np.dstack([shift(r, r_d), shift(g, g_d), b]), g_d, r_d


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        plate = load_plate(path)
        t = time.time()
        edges = [detect_edges(c, p["detector"], p["sigma"], p["low"], p["high"]) for c in split_plate(plate)]
        edge_secs = time.time() - t
        eb, eg, er = (e / (np.percentile(e, 99) + 1e-8) for e in edges)
        save_image(img_dir / "edges.jpg", np.dstack([er, eg, eb]))
        for metric in p["metrics"].split(","):
            t = time.time()
            rgb, g_d, r_d = colorize(plate, edges, p["window"], p["refine"], p["crop"], metric,
                                     p["coarse_size"])
            secs = time.time() - t
            save_image(img_dir / f"{metric}.jpg", rgb)
            print(f"{path.stem}  {p['detector']} {metric}  G (dx, dy)={g_d}  R={r_d}  "
                  f"edges {edge_secs:.1f}s  align {secs:.1f}s")
            rows.append([path.name, p["detector"], metric, *g_d, *r_d, f"{edge_secs:.2f}", f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "detector", "metric", "g_dx", "g_dy", "r_dx", "r_dy", "edge_seconds", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
