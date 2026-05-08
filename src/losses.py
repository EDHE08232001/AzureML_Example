"""
Loss functions and quality metrics.

ProgressiveLoss   — combined MS-SSIM + MSE loss as defined in the proposal.
compute_psnr      — batch-averaged PSNR in dB.
compute_msssim_db — batch-averaged MS-SSIM converted to dB scale.
"""

import math

import torch
import torch.nn.functional as F
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


# ── Training loss ──────────────────────────────────────────────────────────────

class ProgressiveLoss:
    """Combined MS-SSIM and MSE loss from the project proposal.

    L = λ · (1 − MS-SSIM(x, x̂))  +  (1 − λ) · MSE(x, x̂)

    A higher λ weights the perceptual (MS-SSIM) term more heavily.
    """

    def __init__(self, lambda_msssim: float = 0.9, device: torch.device = None):
        if device is None:
            device = torch.device("cpu")
        self.lambda_msssim = lambda_msssim
        # Keep the metric on the same device as the model.
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ms_ssim_score = self.msssim(prediction, target)
        mse_score     = F.mse_loss(prediction, target)
        return (
            self.lambda_msssim       * (1.0 - ms_ssim_score)
            + (1.0 - self.lambda_msssim) * mse_score
        )


class MSELoss:
    """Plain mean-squared-error loss (alternative to ProgressiveLoss)."""

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)


# ── Quality metrics (used during validation and evaluation) ────────────────────

def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Batch-averaged Peak Signal-to-Noise Ratio in dB.

    Returns 99.0 dB for numerically perfect reconstructions.
    """
    mse = F.mse_loss(prediction.detach(), target.detach()).item()
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_msssim_db(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Batch-averaged MS-SSIM converted to dB:  −10 · log₁₀(1 − MS-SSIM).

    Moves tensors to CPU to avoid requiring a persistent device reference.
    """
    metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    score  = metric(prediction.detach().cpu(), target.detach().cpu()).item()
    # Clamp to avoid log10(0).
    score  = min(score, 1.0 - 1e-8)
    return -10.0 * math.log10(1.0 - score)
