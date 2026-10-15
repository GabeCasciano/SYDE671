"""Part 2.3: Gaussian and Laplacian stacks, and Szeliski fig 3.42 for the oraple."""

import numpy as np
from scipy.signal import convolve2d

from common import DATA_DIR, load_image, parse_args, save_image, web_dir

DEFAULTS = {
    "im1": "burt_apple.png",
    "im2": "burt_orange.png",
    "levels": 8,
}

BINOMIAL = np.array([1, 4, 6, 4, 1]) / 16


def blur(img, level):
    """Separable binomial filter with 2**level - 1 zeros between taps."""
    k = np.zeros(4 * 2 ** level + 1)
    k[::2 ** level] = BINOMIAL
    k = k[None, :]
    if img.ndim == 3:
        return np.dstack([blur(img[..., c], level) for c in range(img.shape[2])])
    img = convolve2d(img, k, mode="same", boundary="symm")
    return convolve2d(img, k.T, mode="same", boundary="symm")


def gaussian_stack(img, levels):
    stack = [img]
    for i in range(levels - 1):
        stack.append(blur(stack[-1], i))
    return stack


def laplacian_stack(img, levels):
    """Band-pass levels G_i - G_(i+1); the last level is the Gaussian residual."""
    g = gaussian_stack(img, levels)
    return [g[i] - g[i + 1] for i in range(levels - 1)] + [g[-1]]


def blend_stacks(im1, im2, mask, levels):
    """Per level: G_mask * L1 + (1 - G_mask) * L2. Returns the three stacks (masked 1, masked 2, blended)."""
    if im1.ndim == 3 and mask.ndim == 2:
        mask = np.dstack([mask] * im1.shape[2])
    gm = gaussian_stack(mask, levels)
    l1, l2 = laplacian_stack(im1, levels), laplacian_stack(im2, levels)
    a = [m * x for m, x in zip(gm, l1)]
    b = [(1 - m) * x for m, x in zip(gm, l2)]
    return a, b, [x + y for x, y in zip(a, b)]


def show(level, scale=None):
    """Band-pass level around mid grey, scaled by the 99th percentile of |level| (or a given scale)."""
    scale = scale or np.percentile(np.abs(level), 99)
    return np.clip(0.5 + level / (2 * scale), 0, 1)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.3")
    n = p["levels"]
    im1 = load_image(DATA_DIR / p["im1"])
    im2 = load_image(DATA_DIR / p["im2"])

    for name, img in [("apple", im1), ("orange", im2)]:
        g, l = gaussian_stack(img, n), laplacian_stack(img, n)
        print(f"{name}: max |sum(L) - image| = {np.abs(sum(l) - img).max():.2e}")
        for i in range(n):
            save_image(out / f"{name}_g{i}.jpg", g[i])
            save_image(out / f"{name}_l{i}.jpg", np.clip(l[i], 0, 1) if i == n - 1 else show(l[i]))

    # Szeliski fig 3.42: rows are Laplacian levels 0, 2, 4 and the collapsed stacks
    mask = np.zeros(im1.shape[:2])
    mask[:, : im1.shape[1] // 2] = 1
    a, b, c = blend_stacks(im1, im2, mask, n)
    for i in [0, 2, 4]:
        scale = np.percentile(np.abs(c[i]), 99)  # one scale per row so the three columns compare
        for tag, s in [("apple", a), ("orange", b), ("blend", c)]:
            save_image(out / f"fig342_{tag}_l{i}.jpg", show(s[i], scale))
    for tag, s in [("apple", a), ("orange", b), ("blend", c)]:
        save_image(out / f"fig342_{tag}_sum.jpg", sum(s))


if __name__ == "__main__":
    main()
