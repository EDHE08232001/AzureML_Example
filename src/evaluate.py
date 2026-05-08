"""
Rate-distortion evaluation: MCUCoder vs JPEG baseline.

For each number of active latent channels (1–12) the model is evaluated on the
Kodak dataset with uniform quantization applied to simulate realistic bitrates.
JPEG is evaluated at multiple quality levels for comparison.

Results are saved as:
  outputs/results/rd_curves.pdf    — rate-distortion plot
  outputs/results/eval_summary.json — numeric summary
  outputs/results/recon_*.png       — sample reconstructions

Run via the repo-root entry point:
    python main.py  →  choose option 2
"""

import io
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from .config import CONFIG, RESULTS_DIR
from .data import RecursiveImageDataset
from .losses import compute_msssim_db, compute_psnr
from .model import MCUCoder
from .utils import get_device


# ── Quantization helpers ───────────────────────────────────────────────────────

def _quantize_dequantize(channel: torch.Tensor, step: int = 4) -> torch.Tensor:
    """Simulate uniform quantization + dequantization on a single channel map.

    Normalizes values to [0, 255], rounds to multiples of `step`, then maps
    back to the original float range.  This approximates the quantization noise
    a receiver would introduce after entropy decoding.
    """
    ch_min = channel.min()
    ch_max = channel.max()
    span   = ch_max - ch_min

    if span < 1e-8:
        return channel.clone()

    # Normalize → quantize → dequantize → denormalize.
    normalized  = (channel - ch_min) / span * 255.0
    quantized   = (normalized / step).to(torch.uint8).float() * step
    dequantized = quantized / 255.0 * span + ch_min
    return dequantized


def _apply_quantization(latent: torch.Tensor, keep: int, step: int) -> torch.Tensor:
    """Apply per-channel quantization to the first `keep` channels of a single image."""
    z = latent.clone()
    for c in range(keep):
        z[0, c] = _quantize_dequantize(z[0, c], step)
    return z


# ── Bitrate estimation ─────────────────────────────────────────────────────────

def _estimate_bpp(
    image: torch.Tensor,
    latent: torch.Tensor,
    keep: int,
    quant_bits: int,
) -> float:
    """Compute bpp via the proposal formula: bpp = (Hz·Wz·k·b) / (Hx·Wx).

    Args:
        image:      Input image tensor  (1, 3, Hx, Wx).
        latent:     Full latent tensor  (1, C, Hz, Wz).
        keep:       Number of active channels (k).
        quant_bits: Bits per quantized symbol (b).
    """
    _, _, Hx, Wx = image.shape
    _, _,  Hz, Wz = latent.shape
    return (Hz * Wz * keep * quant_bits) / float(Hx * Wx)


# ── JPEG baseline ──────────────────────────────────────────────────────────────

def _jpeg_stats(
    image_tensor: torch.Tensor, quality: int
) -> Tuple[float, float, float]:
    """Compress/decompress with JPEG in-memory, return (bpp, psnr, msssim_db)."""
    pil_img = Image.fromarray(
        (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    )
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)

    bpp = len(buf.getvalue()) * 8 / float(pil_img.width * pil_img.height)

    buf.seek(0)
    recon_np = np.asarray(Image.open(buf).convert("RGB")).astype("float32") / 255.0
    recon_t  = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0)

    psnr   = compute_psnr(recon_t,   image_tensor.cpu())
    msssim = compute_msssim_db(recon_t, image_tensor.cpu())
    return bpp, psnr, msssim


# ── RD curve plot ──────────────────────────────────────────────────────────────

def _plot_rd_curves(summary: dict, out_path: str) -> None:
    """Save a two-panel rate-distortion figure (bpp vs PSNR and bpp vs MS-SSIM)."""
    model_bpp    = [p["bpp"]       for p in summary["model"]]
    model_psnr   = [p["psnr"]      for p in summary["model"]]
    model_msssim = [p["msssim_db"] for p in summary["model"]]
    jpeg_bpp     = [p["bpp"]       for p in summary["jpeg"]]
    jpeg_psnr    = [p["psnr"]      for p in summary["jpeg"]]
    jpeg_msssim  = [p["msssim_db"] for p in summary["jpeg"]]

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"]  = 42

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR panel.
    axes[0].plot(model_bpp, model_psnr,   "o-",  label="MCUCoder (ours)", color="steelblue")
    axes[0].plot(jpeg_bpp,  jpeg_psnr,    "s--", label="JPEG baseline",   color="darkorange")
    axes[0].set_xlabel("Bits per pixel (bpp)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Rate–Distortion: PSNR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # MS-SSIM panel.
    axes[1].plot(model_bpp, model_msssim, "o-",  label="MCUCoder (ours)", color="steelblue")
    axes[1].plot(jpeg_bpp,  jpeg_msssim,  "s--", label="JPEG baseline",   color="darkorange")
    axes[1].set_xlabel("Bits per pixel (bpp)")
    axes[1].set_ylabel("MS-SSIM (dB)")
    axes[1].set_title("Rate–Distortion: MS-SSIM")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RD curves → {out_path}")


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate_model() -> str:
    """Evaluate MCUCoder vs JPEG on the Kodak dataset and generate RD curves.

    Returns:
        Absolute path to the JSON summary file.
    """
    device     = get_device()
    model_path = os.path.abspath(CONFIG["model_save_path"])
    step       = CONFIG["quant_step"]
    bits       = CONFIG["quant_bits"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No checkpoint found at {model_path}.\n"
            f"Please train the model first (option 1)."
        )

    # ── Load model ────────────────────────────────────────────────────────────
    model = MCUCoder(
        latent_channels=CONFIG["latent_channels"],
        decoder_channels=CONFIG["decoder_channels"],
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded checkpoint: {model_path}")

    # ── Kodak dataset ─────────────────────────────────────────────────────────
    dataset = RecursiveImageDataset(CONFIG["val_data_dir"], image_size=CONFIG["image_size"])
    if len(dataset) == 0:
        raise RuntimeError(f"No images found at: {CONFIG['val_data_dir']}")
    print(f"Kodak images: {len(dataset)}")

    summary: Dict[str, List[dict]] = {"model": [], "jpeg": []}

    # ── MCUCoder at each channel count ────────────────────────────────────────
    for k in tqdm(CONFIG["eval_filter_counts"], desc="Model"):
        total_bpp, total_psnr, total_msssim = 0.0, 0.0, 0.0

        for idx in range(len(dataset)):
            image = dataset[idx].unsqueeze(0).to(device)

            with torch.no_grad():
                latent = model.encoder(image)

                # Zero tail channels, then quantize the k active ones.
                z = latent.clone()
                z[:, k:, :, :] = 0.0
                z = _apply_quantization(z, k, step)

                recon = model.decoder(z)

            total_bpp    += _estimate_bpp(image, latent, k, bits)
            total_psnr   += compute_psnr(recon, image)
            total_msssim += compute_msssim_db(recon, image)

            # Save sample reconstructions at the lowest bitrate level.
            if idx < CONFIG["num_visualizations"] and k == CONFIG["eval_filter_counts"][0]:
                out_path = os.path.abspath(
                    os.path.join(RESULTS_DIR, f"model_recon_k{k:02d}_img{idx}.png")
                )
                save_image(recon.cpu(), out_path)

        n = len(dataset)
        summary["model"].append({
            "active_channels": k,
            "bpp":             total_bpp    / n,
            "psnr":            total_psnr   / n,
            "msssim_db":       total_msssim / n,
        })
        print(f"  k={k:2d} | bpp={total_bpp/n:.4f} | "
              f"PSNR={total_psnr/n:.2f} dB | MS-SSIM={total_msssim/n:.2f} dB")

    # ── JPEG baseline ─────────────────────────────────────────────────────────
    for quality in tqdm(CONFIG["jpeg_qualities"], desc="JPEG"):
        total_bpp, total_psnr, total_msssim = 0.0, 0.0, 0.0

        for idx in range(len(dataset)):
            image = dataset[idx].unsqueeze(0)
            bpp, psnr, msssim = _jpeg_stats(image, quality)
            total_bpp    += bpp
            total_psnr   += psnr
            total_msssim += msssim

            # Save sample JPEG reconstructions at the lowest quality.
            if idx < CONFIG["num_visualizations"] and quality == CONFIG["jpeg_qualities"][0]:
                pil_src = Image.fromarray(
                    (image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
                )
                buf = io.BytesIO()
                pil_src.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                out_path = os.path.abspath(
                    os.path.join(RESULTS_DIR, f"jpeg_q{quality}_img{idx}.png")
                )
                Image.open(buf).convert("RGB").save(out_path)

        n = len(dataset)
        summary["jpeg"].append({
            "quality":   quality,
            "bpp":       total_bpp    / n,
            "psnr":      total_psnr   / n,
            "msssim_db": total_msssim / n,
        })
        print(f"  JPEG q={quality:3d} | bpp={total_bpp/n:.4f} | "
              f"PSNR={total_psnr/n:.2f} dB | MS-SSIM={total_msssim/n:.2f} dB")

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = os.path.abspath(os.path.join(RESULTS_DIR, "eval_summary.json"))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved summary → {json_path}")

    plot_path = os.path.abspath(os.path.join(RESULTS_DIR, "rd_curves.pdf"))
    _plot_rd_curves(summary, plot_path)

    return json_path
