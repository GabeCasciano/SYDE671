"""Part 2.4: multiresolution blending with Gaussian and Laplacian stacks."""

import numpy as np
from matplotlib.path import Path

from common import DATA_DIR, load_image, parse_args, save_image, web_dir
from hybrid import gray, load, warp_to, with_mask
from stacks import blend_stacks, save_process

DEFAULTS = {
    "levels": 8,
    "max_size": 1000,
    "favourite": "father_eyes_sister2",  # gets the fig 10 process figure and a grayscale version
}

# (name, image A, image B, mask). The mask selects A; B is warped onto A using the eye points.
BLENDS = [
    ("father_sister1_seam", "father.JPG", "sister-1.JPG", "vertical"),
    ("father_sister2_seam", "father.JPG", "sister-2.JPG", "vertical"),
    ("sister1_sister2_seam", "sister-1.JPG", "sister-2.JPG", "vertical"),
    ("father_eyes_sister2", "father.JPG", "sister-2.JPG", "eyes"),
    ("sister1_face_father", "sister-1.JPG", "father.JPG", "face"),
    ("sister2_nose_father", "sister-2.JPG", "father.JPG", "nose"),
    ("pastaboy_face_littlecat", "pasta-boy.jpg", "little-cat.jpeg", "face"),
    ("bigcat_face_pastaboy", "big-cat.jpeg", "pasta-boy.jpg", "face"),
]


def ellipse(shape, centre, axes):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (((x - centre[0]) / axes[0]) ** 2 + ((y - centre[1]) / axes[1]) ** 2 <= 1).astype(float)


def polygon(shape, centre, radius, sides, rotation=0.0):
    """Regular polygon (hexagon, pentagon, square, ...) filled with matplotlib's Path."""
    t = rotation + 2 * np.pi * np.arange(sides) / sides
    path = Path(np.c_[centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])
    y, x = np.mgrid[:shape[0], :shape[1]]
    return path.contains_points(np.c_[x.ravel(), y.ravel()]).reshape(shape).astype(float)


def make_mask(kind, shape, pts):
    """Masks placed relative to the eyes: centre c and eye distance d. 1 selects the region from the other image."""
    c, d = pts.mean(0), np.hypot(*(pts[1] - pts[0]))
    if kind == "vertical":
        m = np.zeros(shape)
        m[:, : int(c[0])] = 1
        return m
    if kind == "eyes":
        return ellipse(shape, c, (0.95 * d, 0.4 * d))
    if kind == "face":
        return ellipse(shape, (c[0], c[1] + 0.45 * d), (1.1 * d, 1.6 * d))
    if kind == "nose":
        return polygon(shape, (c[0], c[1] + 0.95 * d), 0.7 * d, 6, np.pi / 6)
    raise ValueError(kind)


def blend(a, b, mask, levels):
    a_stack, b_stack, c_stack = blend_stacks(a, b, mask, levels)
    return np.clip(sum(c_stack), 0, 1), (a_stack, b_stack, c_stack)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.4")
    n = p["levels"]

    # Oraple with a vertical step mask
    apple, orange = load_image(DATA_DIR / "burt_apple.png"), load_image(DATA_DIR / "burt_orange.png")
    mask = np.zeros(apple.shape[:2])
    mask[:, : apple.shape[1] // 2] = 1
    result, stacks = blend(apple, orange, mask, n)
    save_image(out / "oraple.jpg", result)
    save_image(out / "oraple_gray.jpg", blend(gray(apple), gray(orange), mask, n)[0])
    save_process(out, "oraple", *stacks)

    for name, fa, fb, kind in BLENDS:
        a, pts_a = load(fa, p["max_size"])
        b, pts_b = load(fb, p["max_size"])
        bw = warp_to(with_mask(b), pts_b, pts_a, a.shape[:2])
        b, valid = np.clip(bw[..., :3], 0, 1), bw[..., 3] > 0.99
        region = make_mask(kind, a.shape[:2], pts_a)
        # Vertical seams keep A on the left; shape masks paste B inside the shape (where B has pixels)
        mask = region * valid + (1 - valid) if kind == "vertical" else 1 - region * valid
        result, stacks = blend(a, b, mask, n)
        save_image(out / f"{name}_a.jpg", a)
        save_image(out / f"{name}_b.jpg", b)
        save_image(out / f"{name}_mask.png", mask)
        save_image(out / f"{name}.jpg", result)
        if name == p["favourite"]:
            save_image(out / f"{name}_gray.jpg", blend(gray(a), gray(b), mask, n)[0])
            save_process(out, name, *stacks)
        print(f"{name}: {a.shape[:2]}")


if __name__ == "__main__":
    main()
