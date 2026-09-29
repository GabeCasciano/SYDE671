"""Auto colour mapping: undo crosstalk between the spectrally overlapping plate filters.

Applied after white balance, as in a camera pipeline, since the correction exaggerates any colour cast.
"""

import csv
import time

import numpy as np

from auto_crop import best_cropped
from auto_white_balance import DEFAULTS as WB, white_balance
from common import find_images, load_plate, parse_args, run_dir, save_image

DEFAULTS = {
    "images": "*.jpg",  # comma-separated globs or names in data/
    "max_leak": 0.3,    # largest leakage tried between neighbouring filters
    "clip": 0.005,      # largest fraction of pixels allowed to leave [0, 1]
    "steps": 31,        # leakage values tried between 0 and max_leak
}


def leak_matrix(e):
    """Mixing from true to recorded RGB: red and green, green and blue each share a fraction e.

    Rows sum to 1, so grays stay gray. Red and blue do not overlap.
    """
    return np.array([[1 - e, e, 0], [e, 1 - 2 * e, e], [0, e, 1 - e]])


def colour_map(rgb, max_leak, clip, steps):
    """Return (image, leak): the strongest crosstalk correction that clips at most `clip` of pixels."""
    px = rgb.reshape(-1, 3)[::16]
    leak = 0.0
    for e in np.linspace(0, max_leak, steps):
        m = np.linalg.inv(leak_matrix(e))
        out = px @ m.T
        if ((out < -1e-6) | (out > 1 + 1e-6)).any(axis=1).mean() > clip:
            break
        leak = float(e)
    m = np.linalg.inv(leak_matrix(leak))
    return np.clip(rgb @ m.T, 0, 1), leak


def main():
    p = parse_args(DEFAULTS)
    out = run_dir(__file__, p)
    rows = []
    for path in find_images(p["images"]):
        img_dir = out / path.stem
        img_dir.mkdir()
        balanced, _ = white_balance(best_cropped(load_plate(path)), WB["p"])
        t = time.time()
        rgb, leak = colour_map(balanced, p["max_leak"], p["clip"], p["steps"])
        secs = time.time() - t
        save_image(img_dir / "colour_map.jpg", rgb)
        print(f"{path.stem}  leak {leak:.2f}  {secs:.1f}s")
        rows.append([path.name, f"{leak:.3f}", f"{secs:.2f}"])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "leak", "seconds"])
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
