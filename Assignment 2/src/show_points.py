"""Draw the stored alignment points on an image, for the website."""

import matplotlib.pyplot as plt

from common import parse_args, web_dir
from hybrid import load

DEFAULTS = {
    "images": "father.JPG,sister-2.JPG",
    "max_size": 1000,
}


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("points")
    for name in p["images"].split(","):
        img, pts = load(name, p["max_size"])
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.plot(*pts.mean(0), "+", color="yellow", ms=24, mew=3)
        for i, (x, y) in enumerate(pts, 1):
            ax.plot(x, y, "o", mfc="none", mec="red", ms=26, mew=3)
            ax.text(x, y - 0.12 * h / 4, str(i), color="red", fontsize=26, ha="center", weight="bold")
        ax.plot(*pts.T, "--", color="red", lw=2)
        ax.axis("off")
        fig.savefig(out / f"{name.split('.')[0].lower()}.jpg")
        plt.close(fig)


if __name__ == "__main__":
    main()
