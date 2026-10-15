"""Part 1.2: partial derivatives, gradient magnitude, and thresholded edges."""

import numpy as np
from scipy.signal import convolve2d

from common import DATA_DIR, DX, DY, load_image, parse_args, save_image, signed, web_dir

DEFAULTS = {
    "image": "camera-man.jpg",
    "thresholds": "0.05,0.1,0.15,0.2,0.25,0.3",
}


def gradients(img):
    """Return (dx, dy, magnitude)."""
    dx = convolve2d(img, DX, mode="same", boundary="symm")
    dy = convolve2d(img, DY, mode="same", boundary="symm")
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("1.2")
    img = load_image(DATA_DIR / p["image"], gray=True)
    dx, dy, mag = gradients(img)
    save_image(out / "gray.jpg", img)
    save_image(out / "dx.png", signed(dx))
    save_image(out / "dy.png", signed(dy))
    save_image(out / "mag.png", mag / mag.max())
    print(f"magnitude max {mag.max():.3f}, 50/90/99th pct {np.percentile(mag, [50, 90, 99]).round(3)}")
    for t in p["thresholds"].split(","):
        save_image(out / f"edges_{t}.png", (mag > float(t)).astype(float))


if __name__ == "__main__":
    main()
