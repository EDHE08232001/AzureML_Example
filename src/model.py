"""
MCUCoder: lightweight encoder + deep residual decoder with tail-dropout.

Architecture is adapted from the MCUCoder paper (reference [8]).  The encoder
produces a 12-channel latent tensor at 1/8 the input spatial resolution.
During training a random prefix of 1-12 channels is kept active (the rest are
zeroed), forcing early channels to encode the most important information so that
any prefix yields a valid reconstruction at the corresponding bitrate.
"""

import numpy as np
import torch
import torch.nn as nn

from compressai.layers import AttentionBlock, conv1x1, conv3x3
from compressai.models.utils import deconv


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualBottleneckBlock(nn.Module):
    """Bottleneck residual block: 1×1 → 3×3 → 1×1 convolutions (He et al., 2016)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch     = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        # learned identity shortcut when channel widths differ
        self.skip  = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return self.conv3(out) + identity


# ── Encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Lightweight 3-layer convolutional encoder.

    Spatial path for a 224×224 input:
        224 → 112  (7×7 conv, stride 2)
        112 →  28  (5×5 conv, stride 4)
         28 →  28  (3×3 conv, stride 1)  ← latent output
    """

    def __init__(self, latent_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=7, stride=2, padding=3),           # 224 → 112
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=4, padding=1),           # 112 →  28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, latent_channels, kernel_size=3, stride=1, padding=1),  # 28 →  28
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Decoder ────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Deep residual decoder with attention blocks.

    Spatial path for a 28×28 latent:
        28 →  56  (deconv, stride 2) + residual refinement
        56 → 112  (deconv, stride 2) + residual refinement
       112 → 224  (deconv, stride 2)
    """

    def __init__(self, latent_channels: int = 12, N: int = 196):
        super().__init__()

        # Helper: 3 residual bottleneck blocks followed by one attention block.
        def _rbb3_att(ch):
            return [
                ResidualBottleneckBlock(ch, ch),
                ResidualBottleneckBlock(ch, ch),
                ResidualBottleneckBlock(ch, ch),
                AttentionBlock(ch),
            ]

        # Helper: just 3 residual bottleneck blocks (no trailing attention).
        def _rbb3(ch):
            return [
                ResidualBottleneckBlock(ch, ch),
                ResidualBottleneckBlock(ch, ch),
                ResidualBottleneckBlock(ch, ch),
            ]

        self.net = nn.Sequential(
            # ── Stage 1: 28 → 56 ────────────────────────────────────────────
            AttentionBlock(latent_channels),
            deconv(latent_channels, N, kernel_size=5, stride=2),
            *_rbb3_att(N),   # group 1
            *_rbb3_att(N),   # group 2
            *_rbb3(N),       # group 3  (no trailing attention before deconv)

            # ── Stage 2: 56 → 112 ───────────────────────────────────────────
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            *_rbb3_att(N),   # group 1
            *_rbb3_att(N),   # group 2
            *_rbb3(N),       # group 3

            # ── Stage 3: 112 → 224 ──────────────────────────────────────────
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ── Full Model ─────────────────────────────────────────────────────────────────

class MCUCoder(nn.Module):
    """Progressive image compression via stochastic latent tail-dropout.

    Training:
        A random integer k ∈ [1, C] is drawn each forward pass; channels
        k … C are zeroed in the latent tensor before decoding.  This forces
        channel 1 to carry the most important image information, channel 2 the
        second-most important, etc., enabling any prefix of channels to
        reconstruct the image at the corresponding quality level.

    Inference:
        Call forward(x, keep_fraction=k/C) to decode at a specific bitrate.
        k=C (keep_fraction=1.0) uses all channels for highest quality.
    """

    def __init__(self, latent_channels: int = 12, decoder_channels: int = 196):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels, N=decoder_channels)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mask(self, z: torch.Tensor, keep: int) -> torch.Tensor:
        """Zero out channels beyond index `keep` (in-place on a clone)."""
        z_masked = z.clone()
        if keep < self.latent_channels:
            z_masked[:, keep:, :, :] = 0.0
        return z_masked

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, keep_fraction: float = None):
        """Encode → tail-dropout → decode.

        Args:
            x:             Input image batch  (B, 3, H, W).
            keep_fraction: Fraction of channels to keep in [0, 1].
                           - Training with None: sample k ~ Uniform{1, …, C}.
                           - Eval with None:     use all C channels.
                           - Any float:          deterministic, use round(f*C) channels.

        Returns:
            reconstruction: Reconstructed image   (B, 3, H, W).
            latent:         Full latent tensor     (B, C, H_z, W_z).
            used_fraction:  Fraction of channels actually used.
        """
        z = self.encoder(x)

        if keep_fraction is not None:
            keep = max(1, int(round(keep_fraction * self.latent_channels)))
        elif self.training:
            keep = int(np.random.randint(1, self.latent_channels + 1))
        else:
            keep = self.latent_channels  # full quality during eval by default

        z_masked       = self._mask(z, keep)
        reconstruction = self.decoder(z_masked)
        return reconstruction, z, keep / self.latent_channels
