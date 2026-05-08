# MCUCoder — Progressive Learned Image Compression

A convolutional autoencoder that supports **multiple bitrate–quality operating points from a single trained model**. During training, random *tail-dropout* of latent channels forces early channels to carry the most important image information. At inference time, any prefix of 1–12 channels can be decoded independently — trading bitrate for reconstruction quality without re-encoding.

This project is inspired by and evaluated against the MCUCoder paper (Hojjat et al., NeurIPS 2024).

---

## Table of Contents

1. [Project Goals & Implementation Status](#1-project-goals--implementation-status)
2. [Repository Layout](#2-repository-layout)
3. [Architecture Overview](#3-architecture-overview)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Download](#5-dataset-download)
6. [How to Run](#6-how-to-run)
7. [Running the Full Experiment](#7-running-the-full-experiment)
8. [Configuration Reference](#8-configuration-reference)
9. [Output Files](#9-output-files)
10. [Implementation Notes & Known Gaps](#10-implementation-notes--known-gaps)
11. [References](#11-references)

---

## 1. Project Goals & Implementation Status

| # | Proposal Objective | Status | Where |
|---|---|---|---|
| 1 | Implement a convolutional autoencoder in PyTorch | ✅ Done | `src/model.py` |
| 2 | Train with stochastic tail-dropout to order latent channels by importance | Done | `src/model.py`, `src/train.py` |
| 3 | Evaluate rate-distortion curves (bpp vs MS-SSIM and PSNR) | Done | `src/evaluate.py` |
| 3 | Compare against JPEG baseline at multiple quality levels | Done | `src/evaluate.py` |
| 3 | Compare against a non-progressive autoencoder baseline | Done | Use `k=12` (full quality) as the non-progressive point |
| 4 | Benchmark on edge device (DCT/DWT comparison, power efficiency) | ⏳ If time permits | Not yet implemented |

### What the code accomplishes relative to the proposal

**Objective 1 — Autoencoder:** `src/model.py` defines an `Encoder` (3-layer lightweight CNN: 7×7→5×5→3×3) that maps `(B, 3, 224, 224)` to a `(B, 12, 28, 28)` latent, and a deep residual `Decoder` built with `compressai` `AttentionBlock`s and `ResidualBottleneckBlock`s that maps back to `(B, 3, 224, 224)`. This directly matches the architecture described in the proposal.

**Objective 2 — Stochastic tail-dropout:** During each training forward pass, `MCUCoder.forward()` draws a random `k ~ Uniform{1, …, 12}` and zeros channels `k+1 … 12` before decoding. This matches Equation 1 in the proposal and the MCUCoder paper exactly. The loss function `L = λ·(1 − MS-SSIM) + (1 − λ)·MSE` is implemented in `src/losses.py` as `ProgressiveLoss`.

**Objective 3 — Evaluation:** `src/evaluate.py` evaluates the model for each of 12 channel-count levels on the full Kodak dataset, applies per-channel uniform quantization (`step=4`, `bits=6`) to produce realistic bitrate estimates using the proposal's formula `bpp = (Hz·Wz·k·b) / (Hx·Wx)`, and runs a JPEG baseline at 6 quality levels. It produces both a two-panel PDF plot and a JSON summary.

**Objective 4 — Edge benchmarking:** Not yet implemented. The `requirements.txt` includes commented-out TensorFlow/TFLite dependencies for a future deployment step.

---

## 2. Repository Layout

```
ELG5378_Project/
│
├── main.py                        ← sole entry point (always run from here)
├── env_check.py                   ← verify device (CUDA / MPS / CPU)
├── requirements.txt               ← CPU / macOS / generic install
├── requirementsCUDA.txt           ← GPU install (CUDA 12.4, no Conda needed)
├── README.md
├── .gitignore
│
├── src/                           ← source package
│   ├── config.py                  ← all paths + hyperparameters (edit here)
│   ├── data.py                    ← RecursiveImageDataset + DataLoaders
│   ├── model.py                   ← Encoder, Decoder, MCUCoder
│   ├── losses.py                  ← ProgressiveLoss, PSNR, MS-SSIM helpers
│   ├── train.py                   ← training loop, LR schedule, checkpointing
│   ├── evaluate.py                ← RD evaluation, quantization, JPEG baseline, plots
│   ├── prepare_data.py            ← ImageNet high-res selection & preprocessing
│   └── utils.py                   ← set_seed, get_device, format_metrics
│
├── download_scripts/
│   ├── download_kodak.sh          ← Kodak dataset (macOS/Linux)
│   ├── download_kodak.bat         ← Kodak dataset (Windows)
│   └── download_imagenet.py       ← ImageNet subset via Hugging Face
│
├── datasets/                      ← place raw data here (you create these)
│   ├── kodak/                     ← 24 lossless Kodak PNG images
│   ├── imagenet/train/            ← raw ImageNet class subdirectories
│   └── imagenet_prepared/         ← flat PNG output from prepare step (auto)
│
└── outputs/                       ← auto-created at runtime
    ├── checkpoints/mcucoder.pth   ← best model checkpoint
    └── results/
        ├── eval_summary.json      ← numeric RD results
        ├── rd_curves.pdf          ← rate-distortion plot
        └── *.png                  ← sample reconstructions
```

---

## 3. Architecture Overview

```
Input (B, 3, 224, 224)
        │
        ▼
 ┌──────────────────────────────────┐
 │           Encoder                │
 │  Conv 7×7 s=2 → ReLU            │  224 → 112
 │  Conv 5×5 s=4 → ReLU            │  112 →  28
 │  Conv 3×3 s=1 (no activation)   │   28 →  28
 │  Output: (B, 12, 28, 28)        │
 └──────────────┬───────────────────┘
                │
                │  ← Tail-dropout during training: zero channels k+1 … 12
                │     (k drawn from Uniform{1, …, 12} each forward pass)
                │  ← Uniform quantization during evaluation
                │
        ▼
 ┌──────────────────────────────────┐
 │           Decoder                │
 │  AttentionBlock                  │
 │  DeConv s=2 → 3×(ResBottle+Att) │   28 →  56
 │  DeConv s=2 → 3×(ResBottle+Att) │   56 → 112
 │  DeConv s=2 → sigmoid           │  112 → 224
 │  Output: (B, 3, 224, 224)       │
 └──────────────────────────────────┘

Loss:     L = 0.9 × (1 − MS-SSIM) + 0.1 × MSE
Bitrate:  bpp = 28 × 28 × k × 6 / (224 × 224)  ≈ 0.094 × k
```

**Key design choices:**
- No activation on the final encoder layer — preserves full value range before quantization, avoiding ReLU's negative truncation.
- The decoder is intentionally much heavier (~3M parameters) than the encoder (10.5K), an asymmetric design well-suited for a cloud decoder / edge encoder split.
- The `compressai` library provides the `AttentionBlock` and `deconv` modules used in the decoder.

---

## 4. Environment Setup

### Requirements

- Python 3.9+
- PyTorch 2.3+ (CPU, CUDA 12.4, or Apple MPS)

> **No standalone CUDA Toolkit or Conda installation required.** The CUDA-enabled PyTorch wheel bundles everything it needs — just use `requirementsCUDA.txt` (see below).

### Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

Choose the requirements file that matches your hardware:

#### Option A — NVIDIA GPU (CUDA 12.4) — Recommended for training

```bash
pip install --upgrade pip
pip install -r requirementsCUDA.txt
```

This pulls the CUDA-enabled PyTorch wheel directly from PyTorch's index (`cu124`). **No separate CUDA Toolkit installation or Conda environment is needed** — `pip` handles everything.

> Requires an NVIDIA GPU with CUDA Compute Capability 6.1 or higher (GTX 10-series / RTX 20, 30, 40-series and newer). Driver version 520+ is sufficient.

#### Option B — CPU / Apple Silicon (MPS)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The code auto-detects CUDA → MPS → CPU at runtime. Apple MPS (M1/M2/M3) is used automatically on macOS when available.

### Verify your device

```bash
python env_check.py
# Expected output examples:
#   Using device: cuda   (NVIDIA GPU)
#   Using device: mps    (Apple Silicon)
#   Using device: cpu    (fallback)
```

---

## 5. Dataset Download

Create the required directories first:

```bash
mkdir -p datasets/imagenet/train
mkdir -p datasets/kodak
```

### Kodak Validation Set (required for both training and evaluation)

The Kodak dataset is 24 lossless PNG images used as the validation and evaluation set.

**macOS / Linux:**
```bash
zsh download_scripts/download_kodak.sh
```

**Windows:**
```bat
download_scripts\download_kodak.bat
```

Downloads ~35 MB to `datasets/kodak/`. Verify with:
```bash
ls datasets/kodak/ | wc -l   # should print 24
```

### ImageNet Training Subset (required for training only)

**Option A — Download via Hugging Face (~300k images, ~30 GB):**

Requires a free Hugging Face account. Log in first, then run:
```bash
huggingface-cli login
python download_scripts/download_imagenet.py
```

This streams the `imagenet-1k` dataset and saves up to 300,000 images as JPEGs into `datasets/imagenet/train/`. Expect several hours depending on your connection.

**Option B — Use an existing ImageNet copy:**

Symlink or place the `ILSVRC/train/` directory at `datasets/imagenet/train/`. The `RecursiveImageDataset` loader scans subdirectories automatically.

**Option C — Use a smaller subset for quick experiments:**

Point `train_data_dir` in `src/config.py` at any directory of images. Even 5,000–10,000 images are sufficient to verify the training pipeline end-to-end (quality will be lower than a full run).

---

## 6. How to Run

> **All commands must be run from the repository root** (`ELG5378_Project/`). Do not `cd` into `src/`.

The project has a single interactive entry point:

```bash
python main.py
```

```
============================================================
  MCUCoder — Progressive Learned Image Compression
============================================================

Select an action:
  1) Train    — train the model on ImageNet / validate on Kodak
  2) Evaluate — run rate-distortion evaluation and plot RD curves
  3) Prepare  — pre-process raw ImageNet images for training

Enter 1, 2, or 3:
```

### Option 1 — Train

Trains the model with stochastic tail-dropout. After each epoch, it validates at three representative bitrate levels (2 / 6 / 12 active channels) on the Kodak set and saves the best checkpoint.

**Before training**, make sure `src/config.py` is configured correctly. Key settings:

| Key | Default | Notes |
|-----|---------|-------|
| `num_epochs` | `10` | Use `50`–`150` for production-quality results |
| `batch_size` | `16` | Reduce to `8` or `4` if you run out of GPU/RAM |
| `learning_rate` | `1e-4` | Adam LR |
| `lr_decay_epoch` | `9` | Epoch at which LR decays by ×0.1 |
| `loss` | `"msssim"` | `"msssim"` (recommended) or `"mse"` |
| `lambda_msssim` | `0.9` | Weight of the MS-SSIM term |
| `decoder_channels` | `196` | Reduce to `128` or `96` for faster iteration |

During training you will see per-epoch output like:
```
Epoch 3: train_loss: 0.0412  train_psnr: 26.34  val_loss: 0.0389  val_psnr_2ch: 22.11  val_psnr_6ch: 25.77  val_psnr_12ch: 28.43
  ✓ Saved checkpoint → outputs/checkpoints/mcucoder.pth
```

### Option 2 — Evaluate

Loads the checkpoint from `outputs/checkpoints/mcucoder.pth` and runs the full rate-distortion evaluation. Produces:

- `outputs/results/rd_curves.pdf` — two-panel RD plot (PSNR and MS-SSIM vs bpp)
- `outputs/results/eval_summary.json` — numeric results for all channel counts and JPEG quality levels
- `outputs/results/model_recon_k01_img*.png` — sample reconstructions at minimum bitrate
- `outputs/results/jpeg_q10_img*.png` — JPEG reconstructions at lowest quality for comparison

### Option 3 — Prepare ImageNet

Scans the raw ImageNet directory, selects the 300k highest-resolution images, optionally halves very large images (bicubic), adds small uniform noise, and saves flat PNGs to `datasets/imagenet_prepared/`. Run once before training; subsequent `train` runs will automatically detect and use the prepared directory.

---

## 7. Running the Full Experiment

This section describes the complete workflow to reproduce the project's rate-distortion results, from scratch to final plots.

### Step 1 — Set up the environment

Install dependencies for your hardware (see [Section 4](#4-environment-setup)):

```bash
# NVIDIA GPU (recommended)
pip install -r requirementsCUDA.txt

# CPU / Apple Silicon
pip install -r requirements.txt
```

Verify the correct device is detected:
```bash
python env_check.py
```

### Step 2 — Download the Kodak validation set

```bash
zsh download_scripts/download_kodak.sh
```

Verify: 24 PNG files appear in `datasets/kodak/`.

### Step 3 — Obtain training data

Choose one of the options in [Section 5](#5-dataset-download). For a quick sanity-check run, you can use 5,000–10,000 images from any public image dataset.

### Step 4 — (Optional) Preprocess ImageNet

If you downloaded raw ImageNet images, run the preparation step once:

```bash
python main.py   # → enter 3
```

The prepared flat-PNG directory (`datasets/imagenet_prepared/`) will be used automatically on subsequent training runs.

### Step 5 — Configure your experiment

Open `src/config.py` and adjust:

```python
CONFIG = {
    "num_epochs":       100,     # 100 epochs ≈ production quality
    "batch_size":       16,      # reduce if memory-limited
    "decoder_channels": 196,     # reduce to 128 for faster iteration
    "loss":             "msssim",
    "lambda_msssim":    0.9,
    ...
}
```

For a quick smoke-test before a full run:
```python
"num_epochs":           5,
"decoder_channels":     128,
"num_images_to_select": 10_000,
```

### Step 6 — Train the model

```bash
python main.py   # → enter 1
```

Training progress is printed every epoch. The best checkpoint (lowest validation loss) is saved automatically to `outputs/checkpoints/mcucoder.pth`.

> **Expected runtime:** ~30 min/epoch on a modern GPU with 300k images and `batch_size=16`. On Apple M-series (MPS), roughly 3–5× slower. On CPU only, training is not recommended beyond smoke-testing.

### Step 7 — Evaluate and generate RD curves

```bash
python main.py   # → enter 2
```

The evaluation loop runs all 12 channel-count levels × 24 Kodak images, then the JPEG baseline at 6 quality levels. On a GPU this takes 1–3 minutes; on CPU, 5–15 minutes.

**Expected output in `outputs/results/rd_curves.pdf`:**

The RD plot shows two curves on each panel:
- **MCUCoder (ours):** 12 points, one per channel count (k=1 to k=12), with bpp increasing from ~0.094 to ~1.125.
- **JPEG baseline:** 6 points, one per quality level.

A well-trained model should show the MCUCoder curve sitting above the JPEG curve (higher MS-SSIM at the same bpp), especially at low bitrates.

### Step 8 — Interpret the JSON summary

Open `outputs/results/eval_summary.json`. The structure is:

```json
{
  "model": [
    { "active_channels": 1, "bpp": 0.094, "psnr": 22.1, "msssim_db": 10.3 },
    ...
    { "active_channels": 12, "bpp": 1.125, "psnr": 29.4, "msssim_db": 18.7 }
  ],
  "jpeg": [
    { "quality": 10, "bpp": 0.18, "psnr": 24.5, "msssim_db": 11.2 },
    ...
  ]
}
```

Use `msssim_db` values for comparison against the MCUCoder paper (which reports MS-SSIM in dB). Higher is better on both axes.

### Step 9 — (Optional) Ablation: non-progressive baseline

To approximate the non-progressive autoencoder baseline, re-run evaluation after modifying `src/evaluate.py` to fix `k=12` for all images (i.e., always use all channels). Comparing this single-point result to the k=12 progressive curve shows the cost of tail-dropout training vs. training a fixed-rate model.

---

## 8. Configuration Reference

All settings live in `src/config.py` inside the `CONFIG` dict. **Do not hardcode paths in other files** — always import from `config.py`.

```python
CONFIG = {
    # Paths (auto-resolved relative to repo root)
    "train_data_dir": ...,         # imagenet_prepared/ if it exists, else imagenet/train/
    "val_data_dir":   ...,         # datasets/kodak/
    "image_size":     224,         # center-crop size for all images

    # DataLoader
    "batch_size":   16,
    "num_workers":  4,             # set to 0 on Windows if you get multiprocessing errors

    # Model
    "latent_channels":  12,        # number of progressive quality levels
    "decoder_channels": 196,       # internal width of the deep decoder

    # Training
    "num_epochs":     10,          # increase to 50–150 for publication-quality results
    "learning_rate":  1e-4,
    "lr_decay_epoch": 9,           # epoch at which LR multiplies by lr_gamma
    "lr_gamma":       0.1,
    "loss":           "msssim",    # "msssim" (proposed) or "mse"
    "lambda_msssim":  0.9,         # weight of MS-SSIM in the combined loss

    # Evaluation
    "eval_filter_counts": list(range(1, 13)),  # k = 1 … 12
    "jpeg_qualities":     [10, 20, 35, 50, 65, 80],
    "quant_step":         4,       # uniform quantization step (64 levels = 256/4)
    "quant_bits":         6,       # bits per symbol: ceil(log2(256/step)) = 6
    "num_visualizations": 4,       # sample images saved per evaluation run

    # Checkpointing
    "model_save_path": "outputs/checkpoints/mcucoder.pth",

    # ImageNet preparation
    "imagenet_raw_dir":     "datasets/imagenet/train/",
    "imagenet_out_dir":     "datasets/imagenet_prepared/",
    "num_images_to_select": 300_000,
}
```

---

## 9. Output Files

| File | Description |
|------|-------------|
| `outputs/checkpoints/mcucoder.pth` | Best model weights (saved when val loss improves) |
| `outputs/results/eval_summary.json` | PSNR, MS-SSIM (dB), and bpp for all 12 MCUCoder levels + 6 JPEG qualities |
| `outputs/results/rd_curves.pdf` | Two-panel RD plot: PSNR vs bpp (left), MS-SSIM vs bpp (right) |
| `outputs/results/model_recon_k01_img*.png` | Sample reconstructions at minimum bitrate (k=1) |
| `outputs/results/jpeg_q10_img*.png` | Corresponding JPEG reconstructions at quality 10 |

---

## 10. Implementation Notes & Known Gaps

### Requirements files at a glance

| File | Use case | PyTorch source |
|------|----------|----------------|
| `requirementsCUDA.txt` | NVIDIA GPU (CUDA 12.4) — no Conda, no separate toolkit | PyTorch CUDA wheel (`cu124`) |
| `requirements.txt` | CPU, Apple MPS, or any non-CUDA setup | PyPI default (CPU wheel) |

Both files pin the same library versions. The only difference is the `--extra-index-url` that causes `pip` to prefer the GPU-enabled PyTorch binary.

### Differences from the MCUCoder paper

| Aspect | MCUCoder paper | This implementation |
|--------|---------------|---------------------|
| Target hardware | nRF5340 / STM32F7 MCUs | PC (CUDA / MPS / CPU) |
| Encoder quantization | INT8 (hardware-aware) | Simulated uniform quantization at eval time |
| Training data | 300K largest ImageNet images | Same (configurable) |
| Training iterations | 1M | 10 epochs default (~187K steps with 300K images, configurable) |
| Decoder | ELIC-inspired attention + residual | Same architecture via `compressai` |
| Adaptive bitrate | Hardware bitrate control module | Evaluated offline at all 12 levels |

### Known limitations

- **PSNR trade-off:** Because the loss is optimized for MS-SSIM, JPEG may achieve better PSNR at high bitrates. This is expected and consistent with the paper (Appendix A/B). The MS-SSIM RD curve is the primary comparison metric.
- **Bitrate estimation:** The formula `bpp = Hz·Wz·k·b / (Hx·Wx)` assumes raw (uncompressed) channel transmission. Actual arithmetic coding would yield lower bpp. This is a conservative (pessimistic) estimate.
- **Non-progressive baseline:** Not a separate model; approximate it by evaluating with `keep_fraction=1.0` fixed.
- **Edge deployment (Objective 4):** TFLite/CMSIS-NN export and DCT/DWT baseline comparison are not yet implemented.

### Tips for getting good results

- **More epochs matter:** The model trained for 10 epochs will show a valid RD curve but may underperform JPEG at high bitrates. At 50–100 epochs the curve typically crosses JPEG around 0.2–0.3 bpp.
- **MS-SSIM loss is important:** Switching to pure MSE (`"loss": "mse"`) will improve PSNR numbers but hurt the MS-SSIM curve significantly.
- **Reduce `decoder_channels` for faster experiments:** Setting `decoder_channels=128` cuts training time roughly in half with modest quality loss.
- **`num_workers=0` on Windows:** If you encounter multiprocessing issues, set `num_workers` to `0` in `src/config.py`.
- **CUDA out-of-memory:** Reduce `batch_size` to `8` or `4`. Each step requires roughly `batch_size × 3 × 224 × 224 × 4` bytes for inputs alone, plus decoder activations.

---

## 11. References

1. Hojjat, A., Haberer, J., & Landsiedel, O. (2024). MCUCoder: Adaptive Bitrate Learned Video Compression for IoT Devices. *NeurIPS 2024*.
2. Lu, G. et al. (2019). DVC: An End-to-end Deep Video Compression Framework. *CVPR 2019*.
3. Maghari, A. (2019). A Comparative Study of DCT and DWT Image Compression. *JJCIT*.
4. Ballé, J. et al. (2018). Variational Image Compression with a Scale Hyperprior. *ICLR 2018*.
