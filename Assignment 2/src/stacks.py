"""Part 2.3: Gaussian and Laplacian stacks, and Szeliski fig 3.42 for the oraple."""

import numpy as np

from common import DATA_DIR, load_image, parse_args, resize, save_image, web_dir

DEFAULTS = {
    "im1": "burt_apple.png",
    "im2": "burt_orange.png",
    "levels": 8,
    "extra": "father.JPG,sister-1.JPG,sister-2.JPG",  # more images to show stacks for
    "max_size": 1000,
}

BINOMIAL = np.array([1, 4, 6, 4, 1]) / 16


def blur(img, level):
    """Separable binomial filter with 2**level - 1 zeros between taps (only the 5 taps are computed)."""
    step = 2 ** level
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (2 * step, 2 * step)
        p = np.pad(img, pad, mode="symmetric")
        n = img.shape[axis]
        img = sum(w * np.take(p, np.arange(j * step, j * step + n), axis=axis) for j, w in enumerate(BINOMIAL))
    return img


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


def save_process(out, prefix, a, b, c):
    """Szeliski fig 3.42 / Burt and Adelson fig 10: levels 0, 2, 4 and the collapsed stacks."""
    for i in [0, 2, 4]:
        scale = np.percentile(np.abs(c[i]), 99)  # one scale per row so the three columns compare
        for tag, s in [("a", a), ("b", b), ("blend", c)]:
            save_image(out / f"{prefix}_{tag}_l{i}.jpg", show(s[i], scale))
    for tag, s in [("a", a), ("b", b), ("blend", c)]:
        save_image(out / f"{prefix}_{tag}_sum.jpg", sum(s))


def save_stacks(out, name, img, n):
    g, l = gaussian_stack(img, n), laplacian_stack(img, n)
    print(f"{name}: max |sum(L) - image| = {np.abs(sum(l) - img).max():.2e}")
    for i in range(n):
        save_image(out / f"{name}_g{i}.jpg", g[i])
        save_image(out / f"{name}_l{i}.jpg", np.clip(l[i], 0, 1) if i == n - 1 else show(l[i]))


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.3")
    n = p["levels"]
    im1 = load_image(DATA_DIR / p["im1"])
    im2 = load_image(DATA_DIR / p["im2"])
    save_stacks(out, "apple", im1, n)
    save_stacks(out, "orange", im2, n)

    mask = np.zeros(im1.shape[:2])
    mask[:, : im1.shape[1] // 2] = 1
    save_process(out, "fig342", *blend_stacks(im1, im2, mask, n))

    for name in filter(None, p["extra"].split(",")):
        img = load_image(DATA_DIR / name)
        img = resize(img, min(1.0, p["max_size"] / max(img.shape[:2])))
        save_stacks(out, name.split(".")[0].lower(), img, n)


if __name__ == "__main__":
    main()
