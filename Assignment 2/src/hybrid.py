"""Part 2.2: hybrid images, ported from starter/hybrid_image_starter.m and align_images.m."""

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

from common import DATA_DIR, gaussian_blur, load_image, parse_args, save_image, signed, web_dir

DEFAULTS = {
    "im1": "DerekPicture.jpg",   # low frequencies (seen from far)
    "im2": "nutmeg.jpg",         # high frequencies (seen up close)
    "name": "derek_nutmeg",
    "sigma_low": 8.0,
    "sigma_high": 4.0,
    "sweep_low": "4,8,16",       # empty to skip the cutoff sweep
    "sweep_high": "2,4,8",
    "process": 1,                # 1 saves the aligned images and FFTs
    "colour": 1,                 # 1 saves the colour variants (bells & whistles)
}


def get_points(img, n=2):
    plt.figure(1)
    plt.clf()
    plt.imshow(img, cmap="gray")
    plt.title(f"Click {n} points")
    pts = plt.ginput(n, timeout=0)
    plt.close(1)
    return np.array(pts)


def pad_x(img, t):
    w = [(0, 0), (t, 0) if t > 0 else (0, -t)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, w)


def pad_y(img, t):
    w = [(t, 0) if t > 0 else (0, -t), (0, 0)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, w)


def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)


def align_images(im1, im2, pts1, pts2):
    """Translate, scale, and rotate so the point pairs line up, then crop to the same size."""
    (x1, y1), (x2, y2) = pts1.T, pts2.T
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Translate so the centre of the reference points is the centre of the image
    im1 = pad_y(pad_x(im1, round((w1 / 2 - x1.mean()) * 2)), round((h1 / 2 - y1.mean()) * 2))
    im2 = pad_y(pad_x(im2, round((w2 / 2 - x2.mean()) * 2)), round((h2 / 2 - y2.mean()) * 2))

    # Downscale the larger image so the distances between the points match
    len1 = np.hypot(x1[1] - x1[0], y1[1] - y1[0])
    len2 = np.hypot(x2[1] - x2[0], y2[1] - y2[0])
    dscale = len2 / len1
    if dscale < 1:
        im1 = resize(im1, dscale)
    else:
        im2 = resize(im2, 1 / dscale)

    # Rotate im1 so the angle between the points matches
    theta1 = np.arctan2(-(y1[1] - y1[0]), x1[1] - x1[0])
    theta2 = np.arctan2(-(y2[1] - y2[0]), x2[1] - x2[0])
    im1 = rotate(im1, np.degrees(theta2 - theta1), reshape=True, order=1)

    # Crop both images to the same height and width
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    im1 = im1[(h1 - h) // 2:(h1 - h) // 2 + h, (w1 - w) // 2:(w1 - w) // 2 + w]
    im2 = im2[(h2 - h) // 2:(h2 - h) // 2 + h, (w2 - w) // 2:(w2 - w) // 2 + w]
    return im1, im2


def crop_valid(im1, im2):
    """Replaces the starter's manual crop: trim borders until both images have real pixels everywhere.

    The last channel of each image is a validity mask that the padding and rotation zero out.
    """
    valid = (im1[..., -1] > 0.99) & (im2[..., -1] > 0.99)
    t, b, l, r = 0, valid.shape[0], 0, valid.shape[1]
    while not valid[t:b, l:r].all():
        bad = [(~valid[t, l:r]).sum(), (~valid[b - 1, l:r]).sum(), (~valid[t:b, l]).sum(), (~valid[t:b, r - 1]).sum()]
        side = int(np.argmax(bad))
        t, b, l, r = t + (side == 0), b - (side == 1), l + (side == 2), r - (side == 3)
    return im1[t:b, l:r, :-1], im2[t:b, l:r, :-1]


def hybrid_image(im1, im2, sigma_low, sigma_high):
    """Low-pass im1 plus high-pass im2 (impulse minus Gaussian)."""
    low = gaussian_blur(im1, sigma_low)
    high = im2 - gaussian_blur(im2, sigma_high)
    return low, high, low + high


def cutoff(sigma, n):
    """Frequency (cycles per image of size n) where the Gaussian's response falls to 0.5."""
    return np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma) * n


def log_fft(gray):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))


def gray(img):
    return img @ np.array([0.299, 0.587, 0.114])


def load_aligned(p):
    """Load both images and align them, clicking the points once and caching them."""
    im1 = load_image(DATA_DIR / p["im1"])
    im2 = load_image(DATA_DIR / p["im2"])
    ones = lambda im: np.ones(im.shape[:2] + (1,), im.dtype)
    cache = DATA_DIR / f"align_{p['name']}.json"
    if cache.exists():
        pts = json.loads(cache.read_text())
    else:
        print(f"Click two points on {p['im2']} (e.g. the eyes), then the same two on {p['im1']}")
        pts = {"im2": get_points(im2).tolist(), "im1": get_points(im1).tolist()}
        cache.write_text(json.dumps(pts))
    # Same call order as the starter: [im2, im1] = align_images(im2, im1)
    im2a, im1a = align_images(np.dstack([im2, ones(im2)]), np.dstack([im1, ones(im1)]),
                              np.array(pts["im2"]), np.array(pts["im1"]))
    im1a, im2a = crop_valid(im1a, im2a)
    return im1, im2, np.clip(im1a, 0, 1), np.clip(im2a, 0, 1)


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.2")
    name = p["name"]
    im1, im2, im1a, im2a = load_aligned(p)
    g1, g2 = gray(im1a), gray(im2a)
    n = max(g1.shape)
    print(f"aligned size {g1.shape}")

    low, high, hyb = hybrid_image(g1, g2, p["sigma_low"], p["sigma_high"])
    save_image(out / f"{name}_im1.jpg", im1)
    save_image(out / f"{name}_im2.jpg", im2)
    save_image(out / f"{name}_hybrid.jpg", hyb)
    print(f"cutoffs: low {cutoff(p['sigma_low'], n):.1f}, high {cutoff(p['sigma_high'], n):.1f} cycles/image")

    if p["process"]:
        save_image(out / f"{name}_im1_aligned.jpg", g1)
        save_image(out / f"{name}_im2_aligned.jpg", g2)
        save_image(out / f"{name}_low.jpg", low)
        save_image(out / f"{name}_high.jpg", signed(high))
        # Same display range for every spectrum so they can be compared
        ffts = {k: log_fft(v) for k, v in [("im1", g1), ("im2", g2), ("low", low), ("high", high), ("hybrid", hyb)]}
        vmin = min(ffts["im1"].min(), ffts["im2"].min())
        vmax = max(ffts["im1"].max(), ffts["im2"].max())
        for k, f in ffts.items():
            plt.imsave(out / f"{name}_fft_{k}.png", np.clip(f, vmin, vmax), cmap="gray", vmin=vmin, vmax=vmax)
        # Shrinking the hybrid simulates viewing it from further away
        for s in [0.5, 0.25, 0.125]:
            save_image(out / f"{name}_hybrid_{s:g}.jpg", resize(hyb, s))

    if p["sweep_low"]:
        for sl in p["sweep_low"].split(","):
            for sh in p["sweep_high"].split(","):
                _, _, h = hybrid_image(g1, g2, float(sl), float(sh))
                save_image(out / f"{name}_sweep_low{sl}_high{sh}.jpg", h)
                print(f"sweep low {sl} ({cutoff(float(sl), n):.1f}), high {sh} ({cutoff(float(sh), n):.1f})")

    if p["colour"]:
        for mode in ["none", "low", "high", "both"]:
            a = im1a if mode in ("low", "both") else np.dstack([g1] * 3)
            b = im2a if mode in ("high", "both") else np.dstack([g2] * 3)
            _, _, h = hybrid_image(a, b, p["sigma_low"], p["sigma_high"])
            save_image(out / f"{name}_colour_{mode}.jpg", h)


if __name__ == "__main__":
    main()
