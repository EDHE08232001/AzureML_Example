"""
Dataset and DataLoader factories for training and evaluation.

RecursiveImageDataset  — scans all images in a directory tree (for ImageNet).
build_dataloaders      — returns (train_loader, val_loader) ready to use.
"""

import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Dataset ────────────────────────────────────────────────────────────────────

class RecursiveImageDataset(Dataset):
    """Loads every image found recursively under root_dir.

    Works for both flat directories (e.g. Kodak) and class-subdirectory trees
    (e.g. raw ImageNet).  Images are converted to RGB so grayscale and RGBA
    files are handled transparently.
    """

    def __init__(self, root_dir: str, image_size: int = 224):
        self.root_dir    = os.path.abspath(root_dir)
        self.image_paths = self._scan(self.root_dir)
        # Resize shortest edge to image_size, then take a square center crop.
        self.transform   = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _scan(root_dir: str):
        """Walk root_dir and collect all image file paths, sorted for reproducibility."""
        paths = []
        for dirpath, _, files in os.walk(root_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in VALID_EXTENSIONS:
                    paths.append(os.path.join(dirpath, fname))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    train_dir: str,
    val_dir:   str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Training loader:   shuffled, drops the last incomplete batch.
    Validation loader: sequential, batch_size=1 (images differ in size).
    """
    train_dataset = RecursiveImageDataset(train_dir, image_size=image_size)
    val_dataset   = RecursiveImageDataset(val_dir,   image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,           # Kodak images may vary; keep batch=1 for safety
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
