"""
ImageNet preparation: select the N highest-resolution images, optionally
resize them if they are very large, add a small noise perturbation, and
copy everything into a flat PNG output directory.

This mirrors the pre-processing used in the original MCUCoder pipeline.

Run via the repo-root entry point:
    python main.py  →  choose option 3
"""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import CONFIG


# ── Resolution scanning ────────────────────────────────────────────────────────

def _scan_resolutions(image_folder: str) -> list:
    """Walk image_folder recursively and return [(path, pixel_count), …]."""
    records = []
    extensions = {".jpg", ".jpeg", ".png"}
    for root, _, files in tqdm(os.walk(image_folder), desc="Scanning"):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in extensions:
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        records.append((path, w * h))
                except Exception as exc:
                    print(f"  [skip] {path}: {exc}")
    return records


def _select_largest(records: list, n: int) -> list:
    """Return the paths of the n images with the largest pixel counts."""
    return [p for p, _ in sorted(records, key=lambda x: x[1], reverse=True)[:n]]


# ── Image processing helpers ───────────────────────────────────────────────────

def _maybe_halve(image: Image.Image) -> Image.Image:
    """Halve resolution via bicubic downsampling if the shorter side exceeds 512 px."""
    w, h = image.size
    if min(w, h) > 512:
        arr = cv2.resize(
            np.array(image),
            (w // 2, h // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        return Image.fromarray(arr)
    return image


def _add_noise(image: Image.Image) -> Image.Image:
    """Add small uniform noise to discourage overfitting on exact pixel values."""
    arr   = np.array(image).astype(np.int16)
    noise = np.random.randint(0, 2, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# ── Main preparation function ──────────────────────────────────────────────────

def prepare_imagenet() -> str:
    """Select, resize, and copy ImageNet images to a flat output directory.

    Returns:
        Absolute path to the output directory.
    """
    raw_dir = os.path.abspath(CONFIG["imagenet_raw_dir"])
    out_dir = os.path.abspath(CONFIG["imagenet_out_dir"])
    n_imgs  = CONFIG["num_images_to_select"]

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"ImageNet raw directory not found: {raw_dir}\n"
            f"Please place the raw ImageNet training images there."
        )

    print(f"Scanning {raw_dir} …")
    records = _scan_resolutions(raw_dir)
    print(f"  Found {len(records):,} images.")

    print(f"Selecting {n_imgs:,} highest-resolution images …")
    selected = _select_largest(records, n_imgs)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving processed images to {out_dir} …")

    for src_path in tqdm(selected, desc="Processing"):
        try:
            stem = os.path.splitext(os.path.basename(src_path))[0]
            dst  = os.path.join(out_dir, f"{stem}.png")
            with Image.open(src_path) as img:
                img = _maybe_halve(img.convert("RGB"))
                img = _add_noise(img)
                img.save(dst, format="PNG")
        except Exception as exc:
            print(f"  [skip] {src_path}: {exc}")

    print(f"Done. Prepared images saved to: {out_dir}")
    return out_dir
