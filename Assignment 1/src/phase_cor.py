"""Phase correlation alignment on cropped edge maps (Sobel or Canny)."""

import csv
import time

import numpy as np

from common import crop, find_images, load_plate, parse_args, run_dir, save_image, shift, split_plate
from edges import detect_edges

DEFAULTS = {
    "images": "*.jpg",     # comma-separated globs or names in data/
    "detector": "sobel",   # sobel or canny
    "sigma": 2.0,          # Gaussian smoothing before edge detection (px)
    "low": 0.8,            # canny only: low hysteresis threshold, quantile of gradient magnitude
    "high": 0.9,           # canny only: high hysteresis threshold, quantile of gradient magnitude
    "crop": 0.2,           # fraction trimmed from each side before correlating
    "taper": "hann",       # hann or none: window applied before the FFT to reduce edge effects
}


def phase_correlate(moving, ref, crop_frac, taper):
    """Return ((dx, dy), peak) aligning moving to ref; peak height is a confidence score."""
    a, b = crop(ref, crop_frac), crop(moving, crop_frac)
    a, b = a - a.mean(), b - b.mean()
    if taper == "hann":
        win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        a, b = a * win, b * win
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    # Peaks past the midpoint are negative shifts (the FFT wraps around)
    dx = dx - w if dx > w // 2 else dx
    dy = dy - h if dy > h // 2 else dy
    return (int(dx), int(dy)), float(corr.max())


def colorize(plate, edges, crop_frac, taper):
    """Align G and R to B using (b, g, r) edge maps; return (rgb, g_shift, r_shift, peaks)."""
    b, g, r = split_plate(plate)
    eb, eg, er = edges
    g_d, g_peak = phase_correlate(eg, eb, crop_frac, taper)
    r_d, r_peak = phase_correlate(er, eb, crop_frac, taper)
    return np.dstack([shift(r, r_d), shift(g, g_d), b]), g_d, r_d, (g_peak, r_peak)


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
        t = time.time()
        rgb, g_d, r_d, (g_peak, r_peak) = colorize(plate, edges, p["crop"], p["taper"])
        secs = time.time() - t
        save_image(img_dir / "phase.jpg", rgb)
        print(f"{path.stem}  {p['detector']}  G (dx, dy)={g_d} peak {g_peak:.3f}  R={r_d} peak {r_peak:.3f}  "
              f"edges {edge_secs:.1f}s  align {secs:.1f}s")
        rows.append([path.name, p["detector"], *g_d, *r_d, f"{g_peak:.4f}", f"{r_peak:.4f}",
                     f"{edge_secs:.2f}", f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "detector", "g_dx", "g_dy", "r_dx", "r_dy", "g_peak", "r_peak",
                    "edge_seconds", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
