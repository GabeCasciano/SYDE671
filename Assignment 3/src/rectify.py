"""A.3: rectification with nearest neighbour and bilinear inverse warping.

Usage: python rectify.py
Uses every "rect|<image>" entry saved by click_points.py --rect.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

import viz
from common import DATA_DIR, load_image, load_points, save_image, web_dir
from homography import computeH, warp_box, warpImageBilinear, warpImageNearestNeighbor


def zoom(img, box, factor=4):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def main():
    points = load_points()
    out = web_dir("a3")
    results = {}
    for key, entry in points.items():
        if not key.startswith("rect|"):
            continue
        file = key[5:]
        stem = Path(file).stem
        im = load_image(file)
        corners = np.array(entry["corners"])
        w, h = entry["size"]
        # Target rectangle keeps roughly the clicked width, with the real aspect ratio
        W = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
        Ht = W * h / w
        target = np.array([[0, 0], [W, 0], [W, Ht], [0, Ht]])
        H = computeH(corners, target)

        # Limit the output to a margin around the rectangle so far-away content does not explode
        x0, y0, x1, y1 = warp_box(H, im.shape)
        box = (max(x0, int(-W)), max(y0, int(-Ht)), min(x1, int(2 * W)), min(y1, int(2 * Ht)))

        t = time.perf_counter()
        nn = warpImageNearestNeighbor(im, H, box)
        t_nn = time.perf_counter() - t
        t = time.perf_counter()
        bil = warpImageBilinear(im, H, box)
        t_bil = time.perf_counter() - t

        viz.points(im, corners, out / f"{stem}_corners.jpg", color="#ffee00", size=60, labels=True)
        save_image(out / f"{stem}_nn.jpg", nn[..., :3])
        save_image(out / f"{stem}_bilinear.jpg", bil[..., :3])

        # Zoomed crop inside the rectangle, in output pixel coordinates
        cx, cy = int(W / 2 - box[0]), int(Ht / 2 - box[1])
        s = int(min(W, Ht) / 8)
        crop = (cx - s, cy - s, cx + s, cy + s)
        save_image(out / f"{stem}_nn_zoom.jpg", zoom(nn[..., :3], crop))
        save_image(out / f"{stem}_bilinear_zoom.jpg", zoom(bil[..., :3], crop))

        results[file] = dict(H=H.tolist(), output=[nn.shape[1], nn.shape[0]],
                             seconds_nn=round(t_nn, 3), seconds_bilinear=round(t_bil, 3))
        print(stem, results[file]["output"], f"nn {t_nn:.2f}s, bilinear {t_bil:.2f}s")

    (DATA_DIR / "results_rectify.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
