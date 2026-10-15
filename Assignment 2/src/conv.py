"""Part 1.1: convolution from scratch compared with scipy.signal.convolve2d."""

import time

import numpy as np
from scipy.signal import convolve2d

from common import DATA_DIR, DX, DY, load_image, parse_args, save_image, signed, web_dir

DEFAULTS = {
    "image": "pasta-boy.jpg",
    "box": 9,
}


def pad(img, kernel):
    """Zero pad so the output matches scipy's mode="same"."""
    kh, kw = kernel.shape
    return np.pad(img, ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2)))


def conv4(img, kernel):
    """Convolution with four loops."""
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    p = pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += p[i + u, j + v] * k[u, v]
            out[i, j] = total
    return out


def conv2(img, kernel):
    """Convolution with two loops."""
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    p = pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(p[i:i + kh, j:j + kw] * k)
    return out


def scipy_conv(img, kernel):
    return convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("1.1")
    img = load_image(DATA_DIR / p["image"], gray=True)
    save_image(out / "gray.jpg", img)

    box = np.ones((p["box"], p["box"])) / p["box"] ** 2
    results = {}
    for name, fn in [("four_loops", conv4), ("two_loops", conv2), ("scipy", scipy_conv)]:
        t = time.time()
        results[name] = fn(img, box)
        print(f"{name:10s} {time.time() - t:8.3f}s")
        save_image(out / f"box_{name}.jpg", results[name])
    for name in ["four_loops", "two_loops"]:
        print(f"max |{name} - scipy| = {np.abs(results[name] - results['scipy']).max():.2e}")

    for name, d in [("dx", DX), ("dy", DY)]:
        mine = conv2(img, d)
        print(f"max |{name} two_loops - scipy| = {np.abs(mine - scipy_conv(img, d)).max():.2e}")
        save_image(out / f"{name}.jpg", signed(mine))


if __name__ == "__main__":
    main()
