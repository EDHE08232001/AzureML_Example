"""
Environment check for the MCUCoder project.

Verifies that all dependencies, directories, and configuration are in order
before running preprocessing (option 3), training (option 1), or evaluation
(option 2).  Each check prints [OK], [WARN], or [FAIL] and a final summary
is printed at the end.  The script exits with code 1 if any FAIL is found.

Usage:
    python env_check.py
"""

import importlib
import os
import sys
import tempfile

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS  = "[OK]  "
WARN  = "[WARN]"
FAIL  = "[FAIL]"

_fails = 0
_warns = 0


def _ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def _warn(msg: str) -> None:
    global _warns
    _warns += 1
    print(f"  {WARN} {msg}")


def _fail(msg: str) -> None:
    global _fails
    _fails += 1
    print(f"  {FAIL} {msg}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Python version ──────────────────────────────────────────────────────────

def check_python() -> None:
    _section("Python version")
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 8):
        _ok(f"Python {ver}")
    else:
        _fail(f"Python {ver} — need >= 3.8")


# ── 2. Required packages ───────────────────────────────────────────────────────

_REQUIRED_PACKAGES = [
    # (import_name, display_name, version_attr)
    ("torch",         "torch",         "__version__"),
    ("torchvision",   "torchvision",   "__version__"),
    ("torchmetrics",  "torchmetrics",  "__version__"),
    ("compressai",    "compressai",    "__version__"),
    ("numpy",         "numpy",         "__version__"),
    ("PIL",           "Pillow",        "__version__"),
    ("cv2",           "opencv-python", "__version__"),
    ("tqdm",          "tqdm",          "__version__"),
    ("matplotlib",    "matplotlib",    "__version__"),
]


def check_packages() -> None:
    _section("Required packages")
    for import_name, display_name, version_attr in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, version_attr, "unknown")
            _ok(f"{display_name} {ver}")
        except ImportError as exc:
            _fail(f"{display_name} not found — {exc}")


# ── 3. Device / CUDA ──────────────────────────────────────────────────────────

def check_device() -> None:
    _section("Compute device")
    try:
        import torch
        _ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "unknown"
            _ok(f"CUDA available — {n} GPU(s), device 0: {name}, CUDA {cuda_ver}")
        else:
            _warn("CUDA not available — training will run on CPU (slow)")

        if torch.backends.mps.is_available():
            _ok("Apple MPS available")

        # Show the device that get_device() would pick.
        from src.utils import get_device
        device = get_device()
        _ok(f"Active device: {device}")

    except Exception as exc:
        _fail(f"Device check error: {exc}")


# ── 4. Config sanity ──────────────────────────────────────────────────────────

def check_config() -> None:
    _section("CONFIG sanity")
    try:
        from src.config import CONFIG

        # Numeric range checks.
        checks = [
            ("latent_channels",  1,  64),
            ("decoder_channels", 1, 512),
            ("batch_size",       1, 4096),
            ("num_workers",      0,   32),
            ("num_epochs",       1, 10000),
            ("image_size",      16,  2048),
        ]
        for key, lo, hi in checks:
            val = CONFIG.get(key)
            if val is None:
                _fail(f"CONFIG['{key}'] is missing")
            elif not (lo <= val <= hi):
                _warn(f"CONFIG['{key}'] = {val} is outside expected range [{lo}, {hi}]")
            else:
                _ok(f"CONFIG['{key}'] = {val}")

        # Loss function.
        loss = CONFIG.get("loss", "")
        if loss in ("msssim", "mse"):
            _ok(f"CONFIG['loss'] = '{loss}'")
        else:
            _fail(f"CONFIG['loss'] = '{loss}' — expected 'msssim' or 'mse'")

        # Learning rate.
        lr = CONFIG.get("learning_rate", 0)
        if 0 < lr < 1:
            _ok(f"CONFIG['learning_rate'] = {lr}")
        else:
            _warn(f"CONFIG['learning_rate'] = {lr} looks unusual")

    except Exception as exc:
        _fail(f"Config check error: {exc}")


# ── 5. Output directories (writable) ─────────────────────────────────────────

def check_output_dirs() -> None:
    _section("Output directories (writable)")
    try:
        from src.config import CHECKPOINT_DIR, RESULTS_DIR

        for label, path in [("checkpoints", CHECKPOINT_DIR), ("results", RESULTS_DIR)]:
            os.makedirs(path, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile(dir=path, delete=True):
                    pass
                _ok(f"{label}: {path}")
            except OSError as exc:
                _fail(f"{label} not writable ({path}): {exc}")

    except Exception as exc:
        _fail(f"Output-dir check error: {exc}")


# ── 6. Data directories ───────────────────────────────────────────────────────

def _count_images(directory: str) -> int:
    """Return the number of image files found recursively under directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for _, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count


def check_data_dirs() -> None:
    _section("Data directories")
    try:
        from src.config import (
            IMAGENET_TRAIN_DIR,
            IMAGENET_OUT_DIR,
            KODAK_DIR,
            CONFIG,
        )

        # ── Preprocessing input (raw ImageNet) ────────────────────────────────
        print("  [Preprocessing — raw ImageNet]")
        if os.path.isdir(IMAGENET_TRAIN_DIR):
            n = _count_images(IMAGENET_TRAIN_DIR)
            if n > 0:
                _ok(f"ImageNet raw dir found with {n:,} images: {IMAGENET_TRAIN_DIR}")
            else:
                _warn(f"ImageNet raw dir exists but contains no images: {IMAGENET_TRAIN_DIR}")
        else:
            _warn(f"ImageNet raw dir not found (needed for preprocessing): {IMAGENET_TRAIN_DIR}")

        # ── Training input: prepared ImageNet (preferred) or raw ──────────────
        print("  [Training — ImageNet source]")
        train_dir = CONFIG["train_data_dir"]
        if os.path.isdir(train_dir):
            n = _count_images(train_dir)
            if n > 0:
                label = "prepared" if train_dir == IMAGENET_OUT_DIR else "raw"
                _ok(f"Train dir ({label}) found with {n:,} images: {train_dir}")
            else:
                _fail(f"Train dir exists but contains no images: {train_dir}")
        else:
            _fail(
                f"Train dir not found: {train_dir}\n"
                f"         Run preprocessing (option 3) first, or place ImageNet images there."
            )

        # ── Validation / evaluation input (Kodak) ─────────────────────────────
        print("  [Training & Evaluation — Kodak]")
        if os.path.isdir(KODAK_DIR):
            n = _count_images(KODAK_DIR)
            if n > 0:
                _ok(f"Kodak dir found with {n} images: {KODAK_DIR}")
            else:
                _fail(f"Kodak dir exists but contains no images: {KODAK_DIR}")
        else:
            _fail(
                f"Kodak dir not found: {KODAK_DIR}\n"
                f"         Download the Kodak dataset and place images there."
            )

        # ── Evaluation checkpoint ──────────────────────────────────────────────
        print("  [Evaluation — model checkpoint]")
        ckpt = os.path.abspath(CONFIG["model_save_path"])
        if os.path.isfile(ckpt):
            size_mb = os.path.getsize(ckpt) / (1024 ** 2)
            _ok(f"Checkpoint found ({size_mb:.1f} MB): {ckpt}")
        else:
            _warn(
                f"Checkpoint not found: {ckpt}\n"
                f"         Train the model first (option 1) before evaluating."
            )

    except Exception as exc:
        _fail(f"Data-dir check error: {exc}")


# ── 7. Model instantiation + dummy forward pass ───────────────────────────────

def check_model() -> None:
    _section("Model instantiation & forward pass")
    try:
        import torch
        from src.config import CONFIG
        from src.model import MCUCoder

        model = MCUCoder(
            latent_channels=CONFIG["latent_channels"],
            decoder_channels=CONFIG["decoder_channels"],
        )
        _ok(f"MCUCoder instantiated (latent_ch={CONFIG['latent_channels']}, "
            f"decoder_ch={CONFIG['decoder_channels']})")

        # Count parameters.
        n_params = sum(p.numel() for p in model.parameters())
        _ok(f"Total parameters: {n_params:,}")

        # Tiny dummy forward pass on CPU (keeps the check fast).
        model.eval()
        dummy = torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"])
        with torch.no_grad():
            recon, latent, frac = model(dummy)

        expected_spatial = CONFIG["image_size"]  # encoder strides: 2 × 4 = 8 → 224/8 = 28
        _ok(f"Forward pass OK — input {list(dummy.shape)}, "
            f"output {list(recon.shape)}, latent {list(latent.shape)}, "
            f"keep_fraction={frac:.3f}")

        # Sanity: reconstruction shape must match input.
        if recon.shape != dummy.shape:
            _fail(f"Output shape {list(recon.shape)} != input shape {list(dummy.shape)}")
        else:
            _ok("Reconstruction shape matches input")

    except Exception as exc:
        _fail(f"Model check error: {exc}")


# ── 8. Loss functions ─────────────────────────────────────────────────────────

def check_losses() -> None:
    _section("Loss functions")
    try:
        import torch
        from src.losses import MSELoss, ProgressiveLoss, compute_msssim_db, compute_psnr

        dummy = torch.rand(1, 3, 64, 64)

        mse_fn = MSELoss()
        mse_val = mse_fn(dummy, dummy).item()
        _ok(f"MSELoss OK (self-loss={mse_val:.6f}, expect ~0)")

        dummy_224 = torch.rand(1, 3, 224, 224)
        prog_fn = ProgressiveLoss(lambda_msssim=0.9, device=torch.device("cpu"))
        prog_val = prog_fn(dummy_224, dummy_224).item()
        _ok(f"ProgressiveLoss OK (self-loss={prog_val:.6f}, expect ~0)")

        psnr = compute_psnr(dummy, dummy)
        _ok(f"compute_psnr OK (self-PSNR={psnr:.1f} dB, expect 99.0)")

    except Exception as exc:
        _fail(f"Loss check error: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MCUCoder — Environment Check")
    print("=" * 60)

    # Ensure repo root is importable.
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    check_python()
    check_packages()
    check_device()
    check_config()
    check_output_dirs()
    check_data_dirs()
    check_model()
    check_losses()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    if _fails == 0 and _warns == 0:
        print("  All checks passed. Environment is ready.")
    elif _fails == 0:
        print(f"  {_warns} warning(s), 0 failures.")
        print("  Environment is usable but review the warnings above.")
    else:
        print(f"  {_fails} failure(s), {_warns} warning(s).")
        print("  Fix the issues marked [FAIL] before running the pipeline.")
    print("=" * 60)

    if _fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
