"""Auto-contrast: stretch intensities to the full range, then boost local contrast."""

import csv
import time

import numpy as np
from skimage import color, exposure

from auto_crop import best_cropped
from common import find_images, load_plate, parse_args, run_dir, save_image

DEFAULTS = {
    "images": "*.jpg",  # comma-separated globs or names in data/
    "low": 0.5,         # percentile mapped to black
    "high": 99.5,       # percentile mapped to white
    "clahe": 0.01,      # CLAHE clip limit on lightness; 0 disables it
}


def auto_contrast(rgb, low, high, clahe):
    """Return (image, (lo, hi)): a shared linear stretch, then optional CLAHE on L of CIELAB."""
    lo, hi = np.percentile(rgb, (low, high))
    out = np.clip((rgb - lo) / (hi - lo), 0, 1)
    if clahe > 0:
        lab = color.rgb2lab(out)
        lab[..., 0] = exposure.equalize_adapthist(lab[..., 0] / 100, clip_limit=clahe) * 100
        out = np.clip(color.lab2rgb(lab), 0, 1)
    return out, (float(lo), float(hi))


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        cropped = best_cropped(load_plate(path))
        t = time.time()
        rgb, (lo, hi) = auto_contrast(cropped, p["low"], p["high"], p["clahe"])
        secs = time.time() - t
        save_image(img_dir / "contrast.jpg", rgb)
        print(f"{path.stem}  input range [{lo:.3f}, {hi:.3f}]  {secs:.1f}s")
        rows.append([path.name, f"{lo:.4f}", f"{hi:.4f}", f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "lo", "hi", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
