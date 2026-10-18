"""Part 1.3: Gaussian smoothing and derivative of Gaussian (DoG) filters."""

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from common import DATA_DIR, DX, DY, gaussian_2d, load_image, parse_args, save_image, signed, web_dir
from finite_diff import gradients

DEFAULTS = {
    "image": "camera-man.jpg",
    "sigma": 2.0,
    "thresholds": "0.02,0.04,0.06,0.08,0.1,0.12",
    "dog_threshold": 0.06,   # chosen threshold, for the final DoG edge image
}


def conv(img, kernel):
    return convolve2d(img, kernel, mode="same", boundary="symm")


def save_kernel(path, k, size=256):
    """Save a small kernel enlarged with nearest neighbour so each tap is visible."""
    img = Image.fromarray((np.clip(signed(k), 0, 1) * 255).astype(np.uint8))
    img.resize((size, size * k.shape[0] // k.shape[1]), Image.NEAREST).save(path)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("1.3")
    img = load_image(DATA_DIR / p["image"], gray=True)
    g = gaussian_2d(p["sigma"])

    # Two convolutions: blur, then finite differences
    blurred = conv(img, g)
    dx, dy, mag = gradients(blurred)
    save_image(out / "blurred.jpg", blurred)
    save_image(out / "dx.png", signed(dx))
    save_image(out / "dy.png", signed(dy))
    save_image(out / "mag.png", mag / mag.max())
    print(f"blurred magnitude max {mag.max():.3f}, 50/90/99th pct {np.percentile(mag, [50, 90, 99]).round(3)}")
    for t in p["thresholds"].split(","):
        save_image(out / f"edges_{t}.png", (mag > float(t)).astype(float))

    # One convolution: derivative of Gaussian filters
    dog_x = convolve2d(g, DX)
    dog_y = convolve2d(g, DY)
    save_kernel(out / "gaussian.png", g / g.max())
    save_kernel(out / "dog_x.png", dog_x)
    save_kernel(out / "dog_y.png", dog_y)
    dx2, dy2 = conv(img, dog_x), conv(img, dog_y)
    mag2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
    save_image(out / "dog_dx.png", signed(dx2))
    save_image(out / "dog_dy.png", signed(dy2))
    save_image(out / "dog_mag.png", mag2 / mag2.max())
    save_image(out / f"dog_edges_{p['dog_threshold']}.png", (mag2 > p["dog_threshold"]).astype(float))

    diff = np.abs(mag - mag2)
    print(f"max |two step - DoG| magnitude: {diff.max():.2e}")
    save_image(out / "diff.png", diff / mag.max())  # same scale as the magnitude image


if __name__ == "__main__":
    main()
