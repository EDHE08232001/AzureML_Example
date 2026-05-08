"""
Project-wide paths and hyperparameters for MCUCoder progressive image compression.

All paths are absolute and resolved relative to the repository root so the
project can be run from any working directory.
"""

import os

# ── Path resolution ────────────────────────────────────────────────────────────
# This file lives in  <repo_root>/src/, so one level up is the repo root.
_HERE     = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))

# Datasets — user must place raw data here (see README for download commands).
DATASETS_DIR       = os.path.join(REPO_ROOT, "datasets")
IMAGENET_TRAIN_DIR = os.path.join(DATASETS_DIR, "imagenet", "train")
IMAGENET_OUT_DIR   = os.path.join(DATASETS_DIR, "imagenet_prepared")  # flat PNG output
KODAK_DIR          = os.path.join(DATASETS_DIR, "kodak")

# Outputs — auto-created at runtime.
OUTPUTS_DIR    = os.path.join(REPO_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(OUTPUTS_DIR, "results")

for _d in [CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Use the prepared (flat) ImageNet directory if it exists, otherwise fall back
# to the raw directory (which may contain class-subdirectories).
_train_dir = IMAGENET_OUT_DIR if os.path.isdir(IMAGENET_OUT_DIR) else IMAGENET_TRAIN_DIR

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data paths ──────────────────────────────────────────────────────────
    "train_data_dir": _train_dir,
    "val_data_dir":   KODAK_DIR,
    "image_size":     224,              # center-crop size used for all images

    # ── DataLoader ──────────────────────────────────────────────────────────
    "batch_size":  16,
    "num_workers": 4,

    # ── Model architecture ──────────────────────────────────────────────────
    "latent_channels":  12,             # 12 latent channels → rates 1/12 … 12/12
    "decoder_channels": 196,            # internal channel width of the deep decoder

    # ── Training ────────────────────────────────────────────────────────────
    "num_epochs":    10,                # increase (e.g. 100+) for production runs
    "learning_rate": 1e-4,
    "lr_decay_epoch": 9,               # epoch at which LR is multiplied by lr_gamma
    "lr_gamma":       0.1,
    "loss":          "msssim",         # "msssim" or "mse"
    "lambda_msssim": 0.9,              # weight of MS-SSIM term in the combined loss

    # ── Evaluation ──────────────────────────────────────────────────────────
    "eval_filter_counts": list(range(1, 13)),   # test with 1 … 12 active channels
    "jpeg_qualities":     [10, 20, 35, 50, 65, 80],
    "quant_step":         4,           # uniform quantization step (64 levels)
    "quant_bits":         6,           # bits per quantized symbol (ceil(log2(256/step)))
    "num_visualizations": 4,           # sample reconstruction images saved per run

    # ── Checkpointing ───────────────────────────────────────────────────────
    "model_save_path": os.path.join(CHECKPOINT_DIR, "mcucoder.pth"),

    # ── ImageNet preparation ─────────────────────────────────────────────────
    "imagenet_raw_dir":     IMAGENET_TRAIN_DIR,
    "imagenet_out_dir":     IMAGENET_OUT_DIR,
    "num_images_to_select": 300_000,   # select the N highest-resolution images
}
