"""
Training loop for the MCUCoder progressive compression model.

Run via the repo-root entry point:
    python main.py   →  choose option 1
    or non-interactively:
    python main.py --mode train

Logging
-------
Metrics, parameters, and the best checkpoint are logged via MLflow.

When this script runs inside an Azure ML job, the MLflow tracking URI is set
automatically by Azure ML to the workspace's MLflow server, so all logs land
under the job's "Metrics" / "Outputs + logs" tabs in Azure ML Studio. When run
locally MLflow falls back to a local ``./mlruns/`` directory.

Best-checkpoint files are also written under ``./outputs/checkpoints/`` so
Azure ML's automatic output upload picks them up regardless of MLflow.
"""

import logging
import os
from typing import Dict

import mlflow
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .config import CONFIG
from .data import build_dataloaders
from .losses import MSELoss, ProgressiveLoss, compute_msssim_db, compute_psnr
from .model import MCUCoder
from .utils import format_metrics, get_device, set_seed


# ── Logger ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcucoder.train")


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def _safe_log_metrics(metrics: Dict[str, float], step: int) -> None:
    """Log a metrics dict to MLflow, ignoring failures (e.g. if no active run)."""
    try:
        # MLflow only accepts numeric values for metrics.
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric, step=step)
    except Exception as exc:  # pragma: no cover — logging must never crash training
        logger.warning("MLflow log_metrics failed at step %d: %s", step, exc)


def _safe_log_params(params: Dict[str, object]) -> None:
    """Log a flat params dict to MLflow, ignoring failures."""
    try:
        # MLflow params must be strings; cast everything for safety.
        mlflow.log_params({k: str(v) for k, v in params.items()})
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log_params failed: %s", exc)


def _safe_log_artifact(path: str, artifact_path: str = None) -> None:
    """Log a single file as an MLflow artifact, ignoring failures."""
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log_artifact failed for %s: %s", path, exc)


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

    logger.info("Device: %s", device)
    logger.info("Train dir: %s", CONFIG["train_data_dir"])
    logger.info("Val   dir: %s", CONFIG["val_data_dir"])

    # ── Start (or reuse) an MLflow run ────────────────────────────────────────
    # When running under an Azure ML job, MLflow autoconfigures the tracking
    # URI and an active run already exists — start_run(nested=True) reuses it
    # cleanly. When running locally, a new run is created in ./mlruns/.
    active = mlflow.active_run()
    run_ctx = (
        mlflow.start_run(nested=True) if active is not None else mlflow.start_run()
    )

    with run_ctx:
        # Log the full hyperparameter set.
        _safe_log_params({
            "image_size":       CONFIG["image_size"],
            "batch_size":       CONFIG["batch_size"],
            "num_workers":      CONFIG["num_workers"],
            "latent_channels":  CONFIG["latent_channels"],
            "decoder_channels": CONFIG["decoder_channels"],
            "num_epochs":       CONFIG["num_epochs"],
            "learning_rate":    CONFIG["learning_rate"],
            "lr_decay_epoch":   CONFIG["lr_decay_epoch"],
            "lr_gamma":         CONFIG["lr_gamma"],
            "loss":             CONFIG["loss"],
            "lambda_msssim":    CONFIG["lambda_msssim"],
            "device":           str(device),
        })

        # ── Data ──────────────────────────────────────────────────────────────
        train_loader, val_loader = build_dataloaders(
            train_dir=CONFIG["train_data_dir"],
            val_dir=CONFIG["val_data_dir"],
            image_size=CONFIG["image_size"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
        )
        logger.info(
            "Training images: %d | Validation images: %d",
            len(train_loader.dataset), len(val_loader.dataset),
        )
        _safe_log_params({
            "train_size": len(train_loader.dataset),
            "val_size":   len(val_loader.dataset),
        })

        # ── Model ─────────────────────────────────────────────────────────────
        model = MCUCoder(
            latent_channels=CONFIG["latent_channels"],
            decoder_channels=CONFIG["decoder_channels"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %d", n_params)
        _safe_log_params({"model_parameters": n_params})

        # ── Loss ──────────────────────────────────────────────────────────────
        if CONFIG["loss"] == "msssim":
            criterion = ProgressiveLoss(
                lambda_msssim=CONFIG["lambda_msssim"], device=device,
            )
        else:
            criterion = MSELoss()

        # ── Optimizer + LR schedule ───────────────────────────────────────────
        optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
        scheduler = StepLR(
            optimizer,
            step_size=CONFIG["lr_decay_epoch"],
            gamma=CONFIG["lr_gamma"],
        )

        save_path     = os.path.abspath(CONFIG["model_save_path"])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_val_loss = float("inf")

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(1, CONFIG["num_epochs"] + 1):

            # Training pass — stochastic tail-dropout applied inside model.forward().
            model.train()
            train_loss, train_psnr, n_batches = 0.0, 0.0, 0

            for images in tqdm(
                train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']} [train]"
            ):
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

            # Log learning rate (current, before scheduler.step()).
            current_lr = optimizer.param_groups[0]["lr"]

            summary = {
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "lr":         current_lr,
                **val_metrics,
            }
            logger.info("Epoch %d: %s", epoch, format_metrics(summary))
            _safe_log_metrics(summary, step=epoch)

            # Save if validation loss improved.
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(model.state_dict(), save_path)
                logger.info("Saved checkpoint -> %s", save_path)
                _safe_log_metrics({"best_val_loss": best_val_loss}, step=epoch)

            scheduler.step()

        # Log the final best checkpoint as an artifact.
        if os.path.exists(save_path):
            _safe_log_artifact(save_path, artifact_path="checkpoints")

        logger.info("Training complete. Best val loss: %.6f", best_val_loss)

    return save_path
