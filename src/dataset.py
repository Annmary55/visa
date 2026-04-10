"""VisA dataset loader for PatchCore anomaly detection.

Expected on-disk layout::

    <root>/<category>/Data/Images/Anomaly/*.JPG
    <root>/<category>/Data/Images/Normal/*.JPG
    <root>/<category>/Data/Masks/Anomaly/*.png

Normal images are split 80 % train / 20 % val (deterministic, fixed seed).
Anomaly images are split 20 % val / 80 % test (deterministic, fixed seed).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constants ──────────────────────────────────────────────────────────────────

CATEGORIES: List[str] = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

# ImageNet normalisation statistics
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Split seed – never change so that train/val/test sets are reproducible
_SPLIT_SEED = 42


# ── Dataset class ──────────────────────────────────────────────────────────────


class VisADataset(Dataset):
    """PyTorch Dataset for the VisA benchmark.

    Parameters
    ----------
    root:
        Path to the VisA dataset root that contains one sub-directory per
        category (e.g. ``<root>/candle/Data/Images/Normal/``).
    category:
        One of the 12 VisA categories (see ``CATEGORIES``).
    split:
        One of ``'train'``, ``'val'``, or ``'test'``.
    image_size:
        Both spatial dimensions of the output image/mask tensors.
    use_mask:
        When ``True``, the ``'mask'`` key contains a binarised ground-truth
        mask tensor (``1×H×W``).  When ``False`` (or no mask file exists), an
        all-zero tensor of the same shape is returned.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_size: int = 256,
        use_mask: bool = True,
    ) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Choose from {CATEGORIES}.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'.")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.use_mask = use_mask

        self._image_transform = _build_image_transform(image_size)
        self._mask_transform = _build_mask_transform(image_size)

        self.samples: List[Dict] = self._collect_samples()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[Dict]:
        """Collect (image_path, mask_path, label) tuples for the requested split."""
        cat_root = self.root / self.category / "Data"
        normal_dir = cat_root / "Images" / "Normal"
        anomaly_dir = cat_root / "Images" / "Anomaly"
        mask_dir = cat_root / "Masks" / "Anomaly"

        samples: List[Dict] = []

        # ── Normal images (label = 0) ──────────────────────────────────
        normal_paths = sorted(
            p for p in normal_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        train_normal, val_normal = _split_normal(normal_paths, train_ratio=0.8, seed=_SPLIT_SEED)

        if self.split == "train":
            for p in train_normal:
                samples.append({"image_path": str(p), "mask_path": None, "label": 0})
        elif self.split == "val":
            for p in val_normal:
                samples.append({"image_path": str(p), "mask_path": None, "label": 0})

        # ── Anomaly images (label = 1) ─────────────────────────────────
        if self.split in ("val", "test"):
            anomaly_paths = sorted(
                p for p in anomaly_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            val_anomaly, test_anomaly = _split_anomaly(
                anomaly_paths, val_ratio=0.2, seed=_SPLIT_SEED
            )

            chosen_anomaly = val_anomaly if self.split == "val" else test_anomaly
            for p in chosen_anomaly:
                mask_path = _find_mask(p, mask_dir)
                samples.append(
                    {"image_path": str(p), "mask_path": mask_path, "label": 1}
                )

        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        img_tensor: torch.Tensor = self._image_transform(image)

        # Build mask tensor
        h = w = self.image_size
        if self.use_mask and sample["mask_path"] is not None:
            mask_img = Image.open(sample["mask_path"]).convert("L")
            mask_tensor: torch.Tensor = self._mask_transform(mask_img)
        else:
            mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": sample["label"],
            "mask": mask_tensor,
            "image_path": sample["image_path"],
        }


# ── Transforms ────────────────────────────────────────────────────────────────


def _build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _build_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # → [0, 1] float
            # Binarise: any pixel > 0.5 → 1.0
            transforms.Lambda(lambda t: (t > 0.5).float()),
        ]
    )


# ── Split helpers ──────────────────────────────────────────────────────────────


def _split_normal(
    paths: List[Path], train_ratio: float = 0.8, seed: int = _SPLIT_SEED
) -> tuple[List[Path], List[Path]]:
    """Deterministically split normal images into train/val."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def _split_anomaly(
    paths: List[Path], val_ratio: float = 0.2, seed: int = _SPLIT_SEED
) -> tuple[List[Path], List[Path]]:
    """Deterministically split anomaly images into val/test."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[:n_val], shuffled[n_val:]


def _find_mask(image_path: Path, mask_dir: Path) -> Optional[str]:
    """Look up the corresponding mask file for an anomaly image.

    The mask typically shares the same stem but may use a ``.png`` extension.
    """
    stem = image_path.stem
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg"):
        candidate = mask_dir / (stem + ext)
        if candidate.exists():
            return str(candidate)
    return None


# ── DataLoader helper ──────────────────────────────────────────────────────────


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Wrap a ``Dataset`` in a ``DataLoader`` with sensible defaults.

    Parameters
    ----------
    dataset:
        Any PyTorch ``Dataset`` instance.
    batch_size:
        Number of samples per mini-batch.
    shuffle:
        Whether to shuffle the dataset every epoch (use ``True`` for training).
    num_workers:
        Number of worker processes for data loading.
    pin_memory:
        Pin host memory for faster GPU transfers.

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ── Auto-discovery helper ──────────────────────────────────────────────────────


def find_visa_root(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """Try multiple common Kaggle / local paths to locate the VisA dataset root.

    The root is considered valid when it contains at least one of the 12
    expected category sub-directories.

    Parameters
    ----------
    search_paths:
        Additional paths to search.  If ``None``, only the built-in list of
        Kaggle-style paths is tried.

    Returns
    -------
    str or None
        First valid root path, or ``None`` if none is found.
    """
    default_paths = [
        "/kaggle/input/visa",
        "/kaggle/input/visa-dataset",
        "/kaggle/input/visual-anomaly-visa",
        "/kaggle/input/visa-anomaly-detection",
        "/kaggle/input/visadataset",
        os.path.expanduser("~/datasets/visa"),
        "./data/visa",
        "./visa",
    ]
    candidates = list(default_paths)
    if search_paths:
        candidates = list(search_paths) + candidates

    for path in candidates:
        p = Path(path)
        if not p.is_dir():
            continue
        # Validate by checking for at least one expected category directory
        for cat in CATEGORIES:
            if (p / cat).is_dir():
                return str(p)

    return None
