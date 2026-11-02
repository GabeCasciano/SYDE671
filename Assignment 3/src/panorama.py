"""B.5 Full-video panoramas on a sphere.

Usage: python panorama.py [VIDEO ...] [--redo]
Uses the dense frames from: python extract_frames.py VIDEO --diff 0.04 --out frames_dense
Matches every pair of consecutive kept frames automatically (Part B), turns the homographies
into camera rotations with an estimated focal length, and renders all frames onto a sphere.
"""

import argparse
import json

import cv2
import numpy as np

from blend import masked_blur
from common import DATA_DIR, gray, load_image, save_image, web_dir
from features import detect, match, ransac
from homography import sample_bilinear

# Frames left out: these show a person's face
SKIP = {"IMG_0852_0064", "IMG_0852_0065"}


def frames(video):
    files = sorted((DATA_DIR / "frames_dense" / video).glob("*.jpg"))
    return [f"frames_dense/{video}/{f.name}" for f in files if f.stem not in SKIP]


def pair_homographies(files):
    feats = [detect(gray(load_image(f))) for f in files]
    Hs, stats = [], []
    for i in range(len(files) - 1):
        (pa, da), (pb, db) = feats[i][:2], feats[i + 1][:2]
        m, _ = match(da, db)
        H, inl = ransac(pa[m[:, 0]], pb[m[:, 1]]) if len(m) >= 8 else (None, np.zeros(0, bool))
        if H is None or inl.sum() < 12:
            raise RuntimeError(f"no homography between {files[i]} and {files[i + 1]}")
        Hs.append(H)
        stats.append([len(m), int(inl.sum())])
        print(f"{i + 1}/{len(files) - 1}: {len(m)} matches, {inl.sum()} inliers")
    return Hs, stats


def centred(H, c):
    """H in coordinates with the origin at the image centre."""
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    return T @ H @ np.linalg.inv(T)


def to_rotation(H, f):
    K = np.diag([f, f, 1.0])
    M = np.linalg.inv(K) @ H @ K
    M /= np.cbrt(np.linalg.det(M))
    return M


def nearest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else -R


def focal(Hs, lo=400, hi=5000):
    """Focal length (pixels) that makes every K^-1 H K closest to a rotation."""
    fs = np.geomspace(lo, hi, 400)
    cost = [sum(np.linalg.norm(to_rotation(H, f) @ to_rotation(H, f).T - np.eye(3)) for H in Hs) for f in fs]
    return float(fs[int(np.argmin(cost))])


def camera_rotations(Rs):
    """Q_j maps a world ray (reference = middle frame) into camera j."""
    n, m = len(Rs) + 1, len(Rs) // 2
    Q = [np.eye(3) for _ in range(n)]
    for j in range(m + 1, n):
        Q[j] = Rs[j - 1] @ Q[j - 1]
    for j in range(m - 1, -1, -1):
        Q[j] = Rs[j].T @ Q[j + 1]
    # Level the panorama (Brown and Lowe): with no camera roll every x axis is horizontal,
    # so "up" is the direction most perpendicular to all of them
    x = np.array([q.T @ [1, 0, 0] for q in Q])
    up = np.linalg.eigh(x.T @ x)[1][:, 0]
    up *= np.sign(up @ Q[m].T @ [0, 1, 0])
    fwd = Q[m].T @ [0, 0, 1]
    fwd = fwd - (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    W = np.array([np.cross(up, fwd), up, fwd])
    return [q @ W.T for q in Q]


def rays(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(phi), np.cos(theta) * np.cos(phi)], -1)


def project(Q, d, f, c):
    """World rays to pixel coordinates in a camera; z <= 0 marks rays behind it."""
    p = d @ Q.T
    z = p[..., 2]
    zs = np.where(z > 1e-6, z, 1)
    return f * p[..., 0] / zs + c[0], f * p[..., 1] / zs + c[1], z > 1e-6


def angular_box(Q, f, shape, theta_ref):
    """Range of (theta, phi) covered by a frame, with theta unwrapped near theta_ref."""
    h, w = shape[:2]
    t = np.linspace(0, 1, 50)
    border = np.r_[np.c_[t * w, 0 * t], np.c_[t * w, 0 * t + h], np.c_[0 * t, t * h], np.c_[0 * t + w, t * h]]
    d = np.c_[(border[:, 0] - w / 2) / f, (border[:, 1] - h / 2) / f, np.ones(len(border))] @ Q
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    theta = np.arctan2(d[:, 0], d[:, 2])
    theta = theta_ref + (theta - theta_ref + np.pi) % (2 * np.pi) - np.pi
    phi = np.arcsin(d[:, 1])
    return theta.min(), theta.max(), phi.min(), phi.max()


def render(files, Qs, f, sigma=8.0, seam=20.0, scale=1.0):
    """Two-band blend on an equirectangular canvas, scale * f pixels per radian.

    Each pixel is first assigned to the frame with the largest edge weight. The high band comes
    from that frame and the low band from the assignment masks blurred by seam pixels.
    """
    im0 = load_image(files[0])
    c = ((im0.shape[1] - 1) / 2, (im0.shape[0] - 1) / 2)
    boxes, ref = [], 0.0
    for Q in Qs:
        fwd = Q.T @ [0, 0, 1]
        theta_c = ref + (np.arctan2(fwd[0], fwd[2]) - ref + np.pi) % (2 * np.pi) - np.pi
        boxes.append(angular_box(Q, f, im0.shape, theta_c))
        ref = theta_c
    boxes = np.array(boxes)
    t0, p0 = boxes[:, 0].min(), boxes[:, 2].min()
    ppr = scale * f
    W = int(np.ceil((boxes[:, 1].max() - t0) * ppr)) + 1
    Hc = int(np.ceil((boxes[:, 3].max() - p0) * ppr)) + 1
    print(f"canvas {W} x {Hc}, {np.degrees(boxes[:, 1].max() - t0):.0f} x {np.degrees(boxes[:, 3].max() - p0):.0f} degrees")

    def sample(file, Q, box, read):
        ta, tb, pa, pb = box
        x0, x1 = int((ta - t0) * ppr), int(np.ceil((tb - t0) * ppr)) + 1
        y0, y1 = int((pa - p0) * ppr), int(np.ceil((pb - p0) * ppr)) + 1
        th, ph = np.meshgrid(t0 + np.arange(x0, x1) / ppr, p0 + np.arange(y0, y1) / ppr)
        xs, ys, front = project(Q, rays(th, ph), f, c)
        h, w = im0.shape[:2]
        valid = front & (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        wgt = np.clip(np.minimum.reduce([xs + 1, w - xs, ys + 1, h - ys]) / (min(w, h) / 2), 0, 1) * valid
        rgb = sample_bilinear(load_image(file), xs, ys)[0].astype(np.float32) if read else None
        return (slice(y0, y1), slice(x0, x1)), wgt.astype(np.float32), valid, rgb

    best = np.zeros((Hc, W), np.float32)
    owner = np.full((Hc, W), -1, np.int32)
    for k, (Q, box) in enumerate(zip(Qs, boxes)):
        sl, wgt, _, _ = sample(None, Q, box, False)
        take = wgt > best[sl]
        best[sl][take] = wgt[take]
        owner[sl][take] = k

    low = np.zeros((Hc, W, 3), np.float32)
    wsum = np.zeros((Hc, W), np.float32)
    high = np.zeros((Hc, W, 3), np.float32)
    for k, (file, Q, box) in enumerate(zip(files, Qs, boxes)):
        sl, _, valid, rgb = sample(file, Q, box, True)
        a = valid.astype(np.float32)
        lo = masked_blur(rgb, a, sigma)
        mine = owner[sl] == k
        mw = cv2.GaussianBlur(mine.astype(np.float32), (0, 0), seam) * a
        low[sl] += lo * mw[..., None]
        wsum[sl] += mw
        high[sl][mine] = (rgb - lo)[mine]
    covered = wsum > 0
    out = low / np.maximum(wsum, 1e-6)[..., None] + high
    out = np.clip(out, 0, 1) * covered[..., None]
    ys, xs = np.nonzero(covered)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", default=["IMG_0852", "IMG_0861"])
    p.add_argument("--redo", action="store_true", help="recompute the homographies")
    args = p.parse_args()
    path = DATA_DIR / "results_panorama.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for video in args.videos:
        files = frames(video)
        if args.redo or video not in results:
            Hs, stats = pair_homographies(files)
            results[video] = dict(files=files, H=[H.tolist() for H in Hs], matches=stats)
            path.write_text(json.dumps(results, indent=1))
        r = results[video]
        h, w = load_image(files[0]).shape[:2]
        Hs = [centred(np.array(H), ((w - 1) / 2, (h - 1) / 2)) for H in r["H"]]
        f = focal(Hs)
        Qs = camera_rotations([nearest_rotation(to_rotation(H, f)) for H in Hs])
        print(f"{video}: {len(files)} frames, f = {f:.0f} px")
        pano = render(files, Qs, f)
        save_image(web_dir("b5") / f"panorama_{video}.jpg", pano)
        r.update(focal=f, size=[pano.shape[1], pano.shape[0]])
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
