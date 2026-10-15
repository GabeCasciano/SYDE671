"""Part 2.1: unsharp masking as a single convolution."""

import numpy as np
from scipy.signal import convolve2d

from common import DATA_DIR, gaussian_2d, load_image, parse_args, save_image, signed, web_dir

DEFAULTS = {
    "images": "taj.jpg,crazy-cat.jpg",
    "sharp_image": "pasta-boy.jpg",   # sharp image to blur and re-sharpen
    "sigma": 2.0,
    "amounts": "0.5,1,2,4",
}


def conv(img, kernel):
    """Convolve each colour channel."""
    return np.dstack([convolve2d(img[..., c], kernel, mode="same", boundary="symm") for c in range(img.shape[2])])


def unsharp_kernel(sigma, a):
    """(1 + a) * impulse - a * G."""
    g = gaussian_2d(sigma)
    impulse = np.zeros_like(g)
    impulse[g.shape[0] // 2, g.shape[1] // 2] = 1
    return (1 + a) * impulse - a * g


def main():
    p = parse_args(DEFAULTS)
    out = web_dir("2.1")
    amounts = [float(a) for a in p["amounts"].split(",")]

    for name in p["images"].split(","):
        stem = name.split(".")[0]
        img = load_image(DATA_DIR / name)
        blurred = conv(img, gaussian_2d(p["sigma"]))
        save_image(out / f"{stem}.jpg", img)
        save_image(out / f"{stem}_blurred.jpg", blurred)
        save_image(out / f"{stem}_high.jpg", signed(img - blurred))
        for a in amounts:
            save_image(out / f"{stem}_sharp_{a:g}.jpg", conv(img, unsharp_kernel(p["sigma"], a)))

    stem = p["sharp_image"].split(".")[0]
    img = load_image(DATA_DIR / p["sharp_image"])
    blurred = conv(img, gaussian_2d(p["sigma"]))
    save_image(out / f"{stem}.jpg", img)
    save_image(out / f"{stem}_blurred.jpg", blurred)
    for a in amounts:
        resharp = np.clip(conv(blurred, unsharp_kernel(p["sigma"], a)), 0, 1)
        save_image(out / f"{stem}_resharp_{a:g}.jpg", resharp)
        print(f"{stem} a={a:g}  RMSE original vs re-sharpened {np.sqrt(np.mean((img - resharp) ** 2)):.4f}")
    print(f"{stem} RMSE original vs blurred {np.sqrt(np.mean((img - blurred) ** 2)):.4f}")


if __name__ == "__main__":
    main()
