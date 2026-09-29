"""Auto-cropping: detect and remove the plate borders from an aligned image."""

import csv
import time

import numpy as np

from common import find_images, load_plate, parse_args, run_dir, save_image, split_plate
from edges import DEFAULTS as EDGES, colorize, detect_edges

DEFAULTS = {
    "images": "*.jpg",   # comma-separated globs or names in data/
    "max_frac": 0.12,    # search at most this fraction of each side for border
    "spread": 0.5,       # border row: mean channel disagreement (max - min) above this
    "dark": 0.2,         # a channel value below this counts as dark
    "dark_frac": 0.8,    # border row: more than this fraction of one channel is dark
    "gap": 0.02,         # stop after this fraction of the side of clean rows
    "pad": 0.01,         # extra margin cut past the last border row
}


def best_aligned(plate):
    """Align with the best method from the experiments: Sobel edges + pyramid L2."""
    edges = [detect_edges(c, "sobel", EDGES["sigma"], EDGES["low"], EDGES["high"]) for c in split_plate(plate)]
    return colorize(plate, edges, EDGES["window"], EDGES["refine"], EDGES["crop"], "l2", EDGES["coarse_size"])


def border_rows(img, spread, dark, dark_frac):
    """True for rows that look like plate border: channels disagree, or one channel is mostly dark."""
    disagree = (img.max(axis=2) - img.min(axis=2)).mean(axis=1) > spread
    dark_row = ((img < dark).mean(axis=1) > dark_frac).any(axis=1)
    return disagree | dark_row


def border_width(flags, gap):
    """Rows to cut: through the last border row, stopping at the first run of `gap` clean rows."""
    last, clean = -1, 0
    for i, f in enumerate(flags):
        if f:
            last, clean = i, 0
        elif last >= 0:
            clean += 1
            if clean >= gap:
                break
    return last + 1


def auto_crop(rgb, shifts, max_frac, spread, dark, dark_frac, gap, pad):
    """Return (cropped image, (top, bottom, left, right)) given the (dx, dy) shifts of G and R."""
    h, w = rgb.shape[:2]
    # Rows and columns that np.roll wrapped around during alignment
    wrap = {"top": max(0, *(d[1] for d in shifts)), "bottom": max(0, *(-d[1] for d in shifts)),
            "left": max(0, *(d[0] for d in shifts)), "right": max(0, *(-d[0] for d in shifts))}
    sides = {"top": rgb, "bottom": rgb[::-1], "left": rgb.transpose(1, 0, 2), "right": rgb.transpose(1, 0, 2)[::-1]}
    cut = {}
    for side, img in sides.items():
        n = img.shape[0]
        flags = border_rows(img[:int(max_frac * n)], spread, dark, dark_frac)
        width = border_width(flags, max(3, int(gap * n)))
        if width:
            width += max(1, int(pad * n))
        cut[side] = max(wrap[side], width)
    t, b, l, r = cut["top"], cut["bottom"], cut["left"], cut["right"]
    return rgb[t:h - b, l:w - r], (t, b, l, r)


def best_cropped(plate):
    """Best alignment followed by auto-crop with the default settings."""
    rgb, g_d, r_d = best_aligned(plate)
    params = {k: v for k, v in DEFAULTS.items() if k != "images"}
    return auto_crop(rgb, (g_d, r_d), **params)[0]


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    params = {k: v for k, v in p.items() if k != "images"}
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        rgb, g_d, r_d = best_aligned(load_plate(path))
        t = time.time()
        cropped, (top, bottom, left, right) = auto_crop(rgb, (g_d, r_d), **params)
        secs = time.time() - t
        save_image(img_dir / "aligned.jpg", rgb)
        save_image(img_dir / "cropped.jpg", cropped)
        print(f"{path.stem}  cut top {top} bottom {bottom} left {left} right {right} px  {secs:.2f}s")
        rows.append([path.name, top, bottom, left, right, *rgb.shape[:2], f"{secs:.3f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "top", "bottom", "left", "right", "height", "width", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
