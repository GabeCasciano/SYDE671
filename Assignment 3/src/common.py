"""Shared helpers for the Assignment 3 scripts."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT.parent / "website" / "images" / "assignment-3"
POINTS = DATA_DIR / "points.json"

sys.path.insert(0, str(ROOT / "starter"))

# Each mosaic is an ordered list of images (paths relative to data/); the middle one is the reference
MOSAICS = {
    "canyon_slope": ["frames/IMG_0852/IMG_0852_0009.jpg",
                     "frames/IMG_0852/IMG_0852_0010.jpg",
                     "frames/IMG_0852/IMG_0852_0011.jpg"],
    "canyon_ridge": ["frames/IMG_0861/IMG_0861_0040.jpg",
                     "frames/IMG_0861/IMG_0861_0044.jpg",
                     "frames/IMG_0861/IMG_0861_0048.jpg"],
}


def load_image(path):
    """Load an RGB image as float32 in [0, 1]."""
    img = ImageOps.exif_transpose(Image.open(DATA_DIR / path))
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def save_image(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def gray(img):
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def web_dir(part):
    path = WEB_DIR / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_points():
    return json.loads(POINTS.read_text()) if POINTS.exists() else {}


def save_points(points):
    POINTS.write_text(json.dumps(points, indent=1))


def pair_key(a, b):
    return f"{a}|{b}"
