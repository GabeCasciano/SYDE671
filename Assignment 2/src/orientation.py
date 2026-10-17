"""Bells & whistles: gradient orientation visualized with HSV and a cyclic colormap."""

import matplotlib
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

from common import DATA_DIR, DX, DY, gaussian_2d, load_image, parse_args, save_image, web_dir

DEFAULTS = {
    "image": "camera-man.jpg",
    "sigma": 2.0,
    "threshold": 0.06,     # edge threshold from 1.3
    "cmap": "twilight",    # matplotlib cyclic colormap
}


def arctan(t, terms=8):
    """arctan for t in [0, 1] without built-in angle functions.

    Halving twice with arctan(t) = 2 arctan(t / (1 + sqrt(1 + t^2))) brings t below
    tan(pi/16) ~ 0.2, where the Taylor series t - t^3/3 + t^5/5 - ... converges fast.
    """
    for _ in range(2):
        t = t / (1 + np.sqrt(1 + t * t))
    s = sum((-1) ** k * t ** (2 * k + 1) / (2 * k + 1) for k in range(terms))
    return 4 * s


def orientation(gx, gy):
    """Angle in [0, 2*pi), counter-clockwise from +x, built up from the first octant."""
    ax, ay = np.abs(gx), np.abs(gy)
    big = np.maximum(ax, ay)
    small = np.minimum(ax, ay)
    t = np.divide(small, big, out=np.zeros_like(big), where=big > 0)
    theta = arctan(t)                                  # [0, pi/4]
    theta = np.where(ay > ax, np.pi / 2 - theta, theta)  # [0, pi/2]
    theta = np.where(gx < 0, np.pi - theta, theta)       # [0, pi]
    theta = np.where(gy < 0, 2 * np.pi - theta, theta)   # [0, 2*pi]
    return np.mod(theta, 2 * np.pi)


def colourize(theta, value, cmap):
    """Hue from orientation, brightness from value in [0, 1]."""
    if cmap == "hsv":
        return hsv_to_rgb(np.dstack([theta / (2 * np.pi), np.ones_like(theta), value]))
    return matplotlib.colormaps[cmap](theta / (2 * np.pi))[..., :3] * value[..., None]


def wheel(cmap, size=256):
    """Colour wheel legend: angle as hue, radius as brightness, white outside."""
    y, x = np.mgrid[1:-1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(x ** 2 + y ** 2)
    rgb = colourize(orientation(x, y), np.clip(r, 0, 1), cmap)
    rgb[r > 1] = 1
    return rgb


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("bw1")
    img = load_image(DATA_DIR / p["image"], gray=True)
    g = gaussian_2d(p["sigma"])
    gx = convolve2d(img, convolve2d(g, DX), mode="same", boundary="symm")
    gy = -convolve2d(img, convolve2d(g, DY), mode="same", boundary="symm")  # flip so +y points up
    mag = np.sqrt(gx ** 2 + gy ** 2)
    theta = orientation(gx, gy)

    value = np.clip(mag / np.percentile(mag, 99), 0, 1)
    edges = (mag > p["threshold"]).astype(float)
    for name in ["hsv", p["cmap"]]:
        save_image(out / f"{name}.png", colourize(theta, value, name))
        save_image(out / f"{name}_edges.png", colourize(theta, edges, name))
        save_image(out / f"wheel_{name}.png", wheel(name))


if __name__ == "__main__":
    main()
