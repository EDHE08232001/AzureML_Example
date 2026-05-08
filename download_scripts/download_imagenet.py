"""
download_imagenet.py — Robust ImageNet-1k subset downloader via Hugging Face.

Features:
  - Auto-resume: skips already-downloaded images, no re-downloading on retry
  - Retry with exponential backoff on any network/IO error
  - HF token auto-detection from environment or huggingface-cli login cache
  - Progress bar with ETA, speed, and per-run stats
  - Validates saved images (corrupt JPEG detection) with optional --verify flag
  - Configurable via CLI args or environment variables
  - Graceful Ctrl-C: prints a clean summary and exits without a traceback
  - Coloured terminal output (degrades gracefully if not supported)

Usage:
    # Basic (downloads 300 k images, resumes automatically)
    python download_scripts/download_imagenet.py

    # Custom count
    python download_scripts/download_imagenet.py --num-images 50000

    # Verify already-downloaded images for corruption, then resume
    python download_scripts/download_imagenet.py --verify

    # Pass an HF token explicitly (overrides env var and cached login)
    python download_scripts/download_imagenet.py --hf-token hf_xxxxxxxx

    # All options
    python download_scripts/download_imagenet.py \\
        --num-images 300000 \\
        --dataset    benjamin-paine/imagenet-1k-256x256 \\
        --out-dir    datasets/imagenet/train \\
        --max-retries 8 \\
        --verify

Environment variables (override defaults, overridden by CLI flags):
    HF_TOKEN            Hugging Face access token
    IMAGENET_NUM_IMAGES Number of images to download
    IMAGENET_OUT_DIR    Output directory
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# ── Colour helpers (no external dependency) ────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, *codes: str) -> str:
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + _RESET

def info(msg: str)    -> None: print(_c(f"  {msg}", _CYAN))
def ok(msg: str)      -> None: print(_c(f"  ✔  {msg}", _GREEN))
def warn(msg: str)    -> None: print(_c(f"  ⚠  {msg}", _YELLOW))
def error(msg: str)   -> None: print(_c(f"  ✘  {msg}", _RED), file=sys.stderr)
def header(msg: str)  -> None: print(_c(f"\n{'─'*60}\n  {msg}\n{'─'*60}", _BOLD))

# ── Path resolution ────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent
REPO_ROOT  = _HERE.parent

# ── Defaults (overridable via env vars or CLI) ─────────────────────────────────

DEFAULT_NUM_IMAGES  = int(os.environ.get("IMAGENET_NUM_IMAGES", 300_000))
DEFAULT_DATASET     = "benjamin-paine/imagenet-1k-256x256"
DEFAULT_OUT_DIR     = Path(
    os.environ.get("IMAGENET_OUT_DIR", REPO_ROOT / "datasets" / "imagenet" / "train")
)
DEFAULT_MAX_RETRIES = 8
DEFAULT_BASE_WAIT   = 10   # seconds; doubles each retry (capped at 120 s)

# ── HF token discovery ─────────────────────────────────────────────────────────

def _find_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Return an HF token, trying (in order): explicit arg → env var → cached login."""
    if explicit:
        return explicit
    if (env := os.environ.get("HF_TOKEN")):
        return env
    # huggingface-cli stores the token here after `huggingface-cli login`
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        token = token_file.read_text().strip()
        if token:
            return token
    return None

# ── Already-downloaded index ───────────────────────────────────────────────────

def _build_existing_index(out_dir: Path) -> set[int]:
    """Return the set of integer indices already saved as JPEG files."""
    existing = set()
    for f in out_dir.glob("*.jpg"):
        try:
            existing.add(int(f.stem))
        except ValueError:
            pass
    return existing

# ── Optional corruption check ──────────────────────────────────────────────────

def _verify_images(out_dir: Path, indices: set[int]) -> set[int]:
    """Open every saved JPEG and remove corrupt ones. Returns surviving index set."""
    from PIL import Image, UnidentifiedImageError
    from tqdm import tqdm

    bad: list[Path] = []
    paths = sorted(out_dir.glob("*.jpg"))
    for path in tqdm(paths, desc="Verifying images", unit="img"):
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, Exception):
            bad.append(path)

    if bad:
        warn(f"Found {len(bad)} corrupt image(s) — removing and re-queuing.")
        for p in bad:
            try:
                idx = int(p.stem)
                indices.discard(idx)
                p.unlink()
            except Exception:
                pass
    else:
        ok("All existing images passed verification.")
    return indices

# ── Core download logic ────────────────────────────────────────────────────────

def _stream_and_save(
    dataset_name: str,
    out_dir: Path,
    num_images: int,
    existing: set[int],
    hf_token: Optional[str],
) -> int:
    """Stream the HF dataset and save missing images. Returns count saved this run."""
    from datasets import load_dataset
    from tqdm import tqdm

    need_count = num_images - len(existing)
    info(f"Need to download {need_count:,} more image(s).")

    ds_kwargs: dict = dict(split="train", streaming=True)
    if hf_token:
        ds_kwargs["token"] = hf_token

    ds = load_dataset(dataset_name, **ds_kwargs)

    saved_this_run = 0
    with tqdm(
        total=num_images,
        initial=len(existing),
        desc="Downloading",
        unit="img",
        dynamic_ncols=True,
    ) as pbar:
        for i, example in enumerate(ds):
            if i >= num_images:
                break
            if i in existing:
                continue

            out_path = out_dir / f"{i:07d}.jpg"
            example["image"].convert("RGB").save(out_path, format="JPEG", quality=95)
            existing.add(i)
            saved_this_run += 1
            pbar.update(1)

            if len(existing) >= num_images:
                break

    return saved_this_run

# ── Main with retry loop ───────────────────────────────────────────────────────

def download_data(
    num_images:  int            = DEFAULT_NUM_IMAGES,
    dataset_name: str           = DEFAULT_DATASET,
    out_dir:     Path           = DEFAULT_OUT_DIR,
    max_retries: int            = DEFAULT_MAX_RETRIES,
    hf_token:    Optional[str]  = None,
    verify:      bool           = False,
) -> Path:
    """
    Download up to `num_images` images from `dataset_name` into `out_dir`.

    Resumes automatically if interrupted; retries on transient network errors
    with exponential backoff.

    Returns the absolute path to the output directory.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dependency check ──────────────────────────────────────────────────────
    missing = []
    for pkg in ("datasets", "tqdm", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        error(f"Missing packages: {', '.join(missing)}")
        error("Run:  pip install datasets tqdm Pillow")
        sys.exit(1)

    # ── Token ─────────────────────────────────────────────────────────────────
    token = _find_hf_token(hf_token)
    if token:
        ok(f"HF token found ({token[:8]}…)")
    else:
        warn("No HF token found — unauthenticated requests may be rate-limited.")
        warn("Run  huggingface-cli login  or set HF_TOKEN to get higher limits.")

    header(f"ImageNet Download  ·  target: {num_images:,} images")
    info(f"Dataset  : {dataset_name}")
    info(f"Output   : {out_dir}")

    # ── Existing images ───────────────────────────────────────────────────────
    existing = _build_existing_index(out_dir)
    info(f"Already saved: {len(existing):,} image(s).")

    if verify and existing:
        header("Image Verification")
        existing = _verify_images(out_dir, existing)

    if len(existing) >= num_images:
        ok(f"All {num_images:,} images already present. Nothing to do.")
        return out_dir

    # ── Retry loop ────────────────────────────────────────────────────────────
    total_saved = 0
    for attempt in range(1, max_retries + 1):
        try:
            header(f"Download  (attempt {attempt}/{max_retries})")
            saved = _stream_and_save(dataset_name, out_dir, num_images, existing, token)
            total_saved += saved

            # Re-check after the stream ends
            existing = _build_existing_index(out_dir)
            if len(existing) >= num_images:
                break
            else:
                warn(f"Stream ended early — have {len(existing):,}/{num_images:,}. Retrying…")

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            existing = _build_existing_index(out_dir)
            error(f"Error: {exc}")
            info(f"Progress saved: {len(existing):,}/{num_images:,} images on disk.")

            if attempt == max_retries:
                error("Maximum retries reached.")
                error("Re-run this script to continue from where it left off.")
                sys.exit(1)

            wait = min(DEFAULT_BASE_WAIT * (2 ** (attempt - 1)), 120)
            warn(f"Retrying in {wait}s… (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    # ── Final summary ─────────────────────────────────────────────────────────
    final_count = len(_build_existing_index(out_dir))
    header("Summary")
    ok(f"Images on disk : {final_count:,} / {num_images:,}")
    ok(f"Saved this run : {total_saved:,}")
    ok(f"Output dir     : {out_dir}")
    if final_count < num_images:
        warn(f"Still missing {num_images - final_count:,} images — re-run to continue.")

    return out_dir

# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust ImageNet-1k subset downloader via Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-images", type=int, default=DEFAULT_NUM_IMAGES,
        metavar="N",
        help="Total number of images to download.",
    )
    p.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        metavar="REPO_ID",
        help="Hugging Face dataset repository ID.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help="Directory where images will be saved.",
    )
    p.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        metavar="N",
        help="Maximum number of retry attempts on network errors.",
    )
    p.add_argument(
        "--hf-token", type=str, default=None,
        metavar="TOKEN",
        help="Hugging Face access token (overrides env var / cached login).",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Verify existing images for corruption before resuming.",
    )
    return p.parse_args()


def _sigint_handler(sig, frame):  # noqa: ANN001
    print()
    warn("Interrupted by user.  Progress is saved — re-run to continue.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint_handler)
    args = _parse_args()
    download_data(
        num_images=args.num_images,
        dataset_name=args.dataset,
        out_dir=args.out_dir,
        max_retries=args.max_retries,
        hf_token=args.hf_token,
        verify=args.verify,
    )
