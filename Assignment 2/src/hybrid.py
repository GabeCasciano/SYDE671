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
    "im_mid": "",                # optional middle band for a three-band hybrid
    "name": "derek_nutmeg",
    "sigma_low": 8.0,
    "sigma_high": 4.0,
    "sigma_mid": "3,10",         # band-pass G(small) - G(large) for the middle image
    "max_size": 1500,            # downscale inputs so the longest side is at most this
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


def crop_valid(*imgs):
    """Replaces the starter's manual crop: trim borders until every image has real pixels everywhere.

    The last channel of each image is a validity mask that the padding and rotation zero out.
    """
    valid = np.all([im[..., -1] > 0.99 for im in imgs], axis=0)
    t, b, l, r = 0, valid.shape[0], 0, valid.shape[1]
    while not valid[t:b, l:r].all():
        bad = [(~valid[t, l:r]).sum(), (~valid[b - 1, l:r]).sum(), (~valid[t:b, l]).sum(), (~valid[t:b, r - 1]).sum()]
        side = int(np.argmax(bad))
        t, b, l, r = t + (side == 0), b - (side == 1), l + (side == 2), r - (side == 3)
    return [im[t:b, l:r, :-1] for im in imgs]


def warp_to(img, pts, ref_pts, shape):
    """Same two-point translate, scale, and rotate as align_images, as one warp into a reference frame."""
    angle = lambda q: np.arctan2(-(q[1, 1] - q[0, 1]), q[1, 0] - q[0, 0])
    length = lambda q: np.hypot(*(q[1] - q[0]))
    c, c_ref = pts.mean(0), ref_pts.mean(0)
    m = cv2.getRotationMatrix2D(tuple(c), np.degrees(angle(ref_pts) - angle(pts)), length(ref_pts) / length(pts))
    m[:, 2] += c_ref - c
    return cv2.warpAffine(img, m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


def hybrid_image(im1, im2, sigma_low, sigma_high):
    """Low-pass im1 plus high-pass im2 (impulse minus Gaussian)."""
    low = gaussian_blur(im1, sigma_low)
    high = im2 - gaussian_blur(im2, sigma_high)
    return low, high, low + high


def three_band(im1, im_mid, im2, sigma_low, sigma_mid, sigma_high):
    """Low-pass im1, band-pass im_mid (difference of two Gaussians), high-pass im2."""
    low = gaussian_blur(im1, sigma_low)
    mid = gaussian_blur(im_mid, sigma_mid[0]) - gaussian_blur(im_mid, sigma_mid[1])
    high = im2 - gaussian_blur(im2, sigma_high)
    return low, mid, high, low + mid + high


def cutoff(sigma, n):
    """Frequency (cycles per image of size n) where the Gaussian's response falls to 0.5."""
    return np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma) * n


def log_fft(gray):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))


def gray(img):
    return img @ np.array([0.299, 0.587, 0.114])


def load(name, max_size):
    """Load an image downscaled to max_size, with its two reference points (clicked once per image)."""
    img = load_image(DATA_DIR / name)
    scale = min(1.0, max_size / max(img.shape[:2]))
    img = resize(img, scale) if scale < 1 else img
    store = DATA_DIR / "points.json"
    points = json.loads(store.read_text()) if store.exists() else {}
    if name not in points:
        print(f"Click the left eye, then the right eye, on {name}")
        points[name] = (get_points(img) / scale).tolist()
        store.write_text(json.dumps(points, indent=1))
    return img, np.array(points[name]) * scale


def with_mask(img):
    return np.dstack([img, np.ones(img.shape[:2], img.dtype)])


def load_aligned(p):
    """Two images aligned with the starter's align_images."""
    im1, pts1 = load(p["im1"], p["max_size"])
    im2, pts2 = load(p["im2"], p["max_size"])
    # Same call order as the starter: [im2, im1] = align_images(im2, im1)
    im2a, im1a = align_images(with_mask(im2), with_mask(im1), pts2, pts1)
    im1a, im2a = crop_valid(im1a, im2a)
    return im1, im2, np.clip(im1a, 0, 1), np.clip(im2a, 0, 1)


def main_three(p, out):
    """Three-band hybrid: each image is warped onto the low-frequency image using its two points."""
    name = p["name"]
    imgs, pts = zip(*[load(p[k], p["max_size"]) for k in ("im1", "im_mid", "im2")])
    shape = imgs[0].shape[:2]
    warped = [with_mask(imgs[0])] + [warp_to(with_mask(im), q, pts[0], shape) for im, q in zip(imgs[1:], pts[1:])]
    g1, gm, g2 = [gray(np.clip(im, 0, 1)) for im in crop_valid(*warped)]
    sm = [float(v) for v in p["sigma_mid"].split(",")]
    low, mid, high, hyb = three_band(g1, gm, g2, p["sigma_low"], sm, p["sigma_high"])
    for k, im in zip(("im1", "im_mid", "im2"), imgs):
        save_image(out / f"{name}_{k}.jpg", im)
    if p["process"]:
        save_image(out / f"{name}_low.jpg", low)
        save_image(out / f"{name}_mid.jpg", signed(mid))
        save_image(out / f"{name}_high.jpg", signed(high))
    save_image(out / f"{name}_hybrid.jpg", hyb)
    for s in [0.5, 0.25, 0.125]:
        save_image(out / f"{name}_hybrid_{s:g}.jpg", resize(hyb, s))
    n = max(g1.shape)
    print(f"{name}: size {g1.shape}, low < {cutoff(p['sigma_low'], n):.1f}, "
          f"mid {cutoff(sm[1], n):.1f}-{cutoff(sm[0], n):.1f}, high > {cutoff(p['sigma_high'], n):.1f} cycles/image")


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.2")
    if p["im_mid"]:
        return main_three(p, out)
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
