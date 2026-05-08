"""
Training loop for the MCUCoder progressive compression model.

Run via the repo-root entry point:
    python main.py   →  choose option 1
"""

import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .config import CONFIG
from .data import build_dataloaders
from .losses import MSELoss, ProgressiveLoss, compute_msssim_db, compute_psnr
from .model import MCUCoder
from .utils import format_metrics, get_device, set_seed


# ── Validation helper ──────────────────────────────────────────────────────────

def _validate(model: MCUCoder, val_loader, criterion, device: torch.device) -> dict:
    """Evaluate the model at three representative bitrate levels.

    Returns a dict with loss, PSNR, and MS-SSIM averaged across Kodak images
    for channel counts 2 / 6 / 12  (low / mid / high quality).
    """
    model.eval()
    metrics = {f"val_psnr_{k}ch": 0.0 for k in [2, 6, 12]}
    metrics.update({f"val_msssim_{k}ch": 0.0 for k in [2, 6, 12]})
    metrics["val_loss"] = 0.0
    n = 0

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)

            for k in [2, 6, 12]:
                frac = k / model.latent_channels
                recon, _, _ = model(images, keep_fraction=frac)
                metrics[f"val_psnr_{k}ch"]   += compute_psnr(recon, images)
                metrics[f"val_msssim_{k}ch"] += compute_msssim_db(recon, images)

            # Loss is measured at full quality (all channels).
            recon_full, _, _ = model(images, keep_fraction=1.0)
            metrics["val_loss"] += criterion(recon_full, images).item()
            n += 1

    # Average over all validation images.
    for key in metrics:
        metrics[key] /= max(1, n)
    return metrics


# ── Main training function ─────────────────────────────────────────────────────

def train_model() -> str:
    """Train MCUCoder and save the best checkpoint.

    Returns:
        Absolute path to the saved checkpoint file.
    """
    set_seed(42)
    device = get_device()

    print(f"Device: {device}")
    print(f"Train dir: {CONFIG['train_data_dir']}")
    print(f"Val   dir: {CONFIG['val_data_dir']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_dir=CONFIG["train_data_dir"],
        val_dir=CONFIG["val_data_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"Training images: {len(train_loader.dataset)} | "
          f"Validation images: {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MCUCoder(
        latent_channels=CONFIG["latent_channels"],
        decoder_channels=CONFIG["decoder_channels"],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    if CONFIG["loss"] == "msssim":
        criterion = ProgressiveLoss(lambda_msssim=CONFIG["lambda_msssim"], device=device)
    else:
        criterion = MSELoss()

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = StepLR(
        optimizer,
        step_size=CONFIG["lr_decay_epoch"],
        gamma=CONFIG["lr_gamma"],
    )

    save_path     = os.path.abspath(CONFIG["model_save_path"])
    best_val_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, CONFIG["num_epochs"] + 1):

        # Training pass — stochastic tail-dropout applied inside model.forward().
        model.train()
        train_loss, train_psnr, n_batches = 0.0, 0.0, 0

        for images in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']} [train]"):
            images = images.to(device)
            recon, _, _ = model(images)          # random keep_fraction per batch
            loss        = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += compute_psnr(recon.detach(), images)
            n_batches  += 1

        train_loss /= max(1, n_batches)
        train_psnr /= max(1, n_batches)

        # Validation pass — three representative bitrate levels.
        val_metrics = _validate(model, val_loader, criterion, device)

        # Print epoch summary.
        summary = {
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            **val_metrics,
        }
        print(f"Epoch {epoch}: {format_metrics(summary)}")

        # Save if validation loss improved.
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved checkpoint → {save_path}")

        scheduler.step()

    return save_path
