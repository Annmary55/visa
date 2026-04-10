"""Heatmap generation and visualisation utilities for PatchCore results.

All functions operate on BGR ``uint8`` numpy arrays (OpenCV convention) so
that results can be saved directly with ``cv2.imwrite``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ImageNet de-normalisation statistics (matches dataset.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Prefer TURBO colourmap; fall back to JET if the OpenCV version is old
_CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


# ── Tensor ↔ numpy conversion ──────────────────────────────────────────────────


def tensor_to_numpy_img(tensor: Tensor) -> np.ndarray:
    """Convert a normalised ``C×H×W`` float tensor to ``H×W×3 uint8 BGR``.

    Reverses the ImageNet normalisation applied by ``dataset.py`` so that the
    recovered image matches the original file.

    Parameters
    ----------
    tensor:
        A float32 tensor of shape ``(3, H, W)`` normalised with ImageNet
        mean/std.

    Returns
    -------
    np.ndarray of shape ``(H, W, 3)`` dtype ``uint8`` in BGR channel order.
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) RGB float
    img = img * _IMAGENET_STD + _IMAGENET_MEAN   # de-normalise
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Heatmap overlay ────────────────────────────────────────────────────────────


def make_heatmap_overlay(
    img_bgr_uint8: np.ndarray,
    anomaly_map: np.ndarray,
    thr_px: Optional[float] = None,
    alpha: float = 0.4,
    cmap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend an anomaly map onto an image as a colour heatmap.

    Parameters
    ----------
    img_bgr_uint8:
        Original image in BGR ``uint8`` format ``(H, W, 3)``.
    anomaly_map:
        2-D float array of per-pixel anomaly scores (any range).
    thr_px:
        If provided, only pixels with ``anomaly_map > thr_px`` are coloured;
        the rest retain the original image pixels.  Pass ``None`` for a full
        heatmap.
    alpha:
        Blending factor: ``0`` = original image only, ``1`` = heatmap only.
    cmap:
        OpenCV colourmap constant.  Defaults to TURBO (or JET as fallback).

    Returns
    -------
    overlay_bgr:
        Blended image ``(H, W, 3) uint8``.
    heatmap_bgr:
        Coloured heatmap image ``(H, W, 3) uint8`` (before blending).
    anomaly_map_resized:
        Anomaly map resized to ``img_bgr_uint8`` spatial dimensions, float32.
    """
    if cmap is None:
        cmap = _CMAP

    h, w = img_bgr_uint8.shape[:2]

    # Resize anomaly map to image resolution
    amap = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalise to [0, 255]
    amap_min, amap_max = amap.min(), amap.max()
    if amap_max > amap_min:
        amap_norm = ((amap - amap_min) / (amap_max - amap_min) * 255.0).astype(np.uint8)
    else:
        amap_norm = np.zeros((h, w), dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(amap_norm, cmap)

    if thr_px is not None:
        # Only colour anomalous regions
        fg_mask = (amap > thr_px).astype(np.uint8)
        # Blend only where fg_mask = 1
        overlay_bgr = img_bgr_uint8.copy()
        fg_idx = fg_mask == 1
        blended = cv2.addWeighted(img_bgr_uint8, 1.0 - alpha, heatmap_bgr, alpha, 0)
        overlay_bgr[fg_idx] = blended[fg_idx]
    else:
        overlay_bgr = cv2.addWeighted(img_bgr_uint8, 1.0 - alpha, heatmap_bgr, alpha, 0)

    return overlay_bgr, heatmap_bgr, amap


# ── GT mask boundary overlay ───────────────────────────────────────────────────


def _draw_gt_contours(
    img: np.ndarray,
    gt_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ground-truth mask boundaries on an image."""
    out = img.copy()
    binary = (gt_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


# ── Per-image result saver ────────────────────────────────────────────────────


def save_image_results(
    output_dir: str | Path,
    image_path: str,
    category: str,
    variant: str,
    anomaly_map: np.ndarray,
    image_score: float,
    label: int,
    pred_label: int,
    thr_img: float,
    thr_px: Optional[float] = None,
    gt_mask: Optional[np.ndarray] = None,
) -> None:
    """Save visualisation artefacts for a single test image.

    Directory structure::

        <output_dir>/<variant>/<category>/<image_stem>/
            anomaly_map.npy
            heatmap.png
            overlay.png
            overlay_with_gt.png   (only when gt_mask is not None)

    Parameters
    ----------
    output_dir:
        Root output directory.
    image_path:
        Absolute or relative path to the original image file.
    category:
        VisA category name (used for sub-directory and annotation text).
    variant:
        Model variant name (used for sub-directory).
    anomaly_map:
        2-D float array of per-pixel anomaly scores.
    image_score:
        Scalar anomaly score for the whole image.
    label:
        Ground-truth label (0 = normal, 1 = anomaly).
    pred_label:
        Predicted label (0 = normal, 1 = anomaly) at ``thr_img``.
    thr_img:
        Image-level decision threshold used for prediction.
    thr_px:
        Optional pixel-level threshold for masked overlay.
    gt_mask:
        Optional ``(H, W)`` ground-truth binary mask.
    """
    image_path = str(image_path)
    stem = Path(image_path).stem
    out_dir = Path(output_dir) / variant / category / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.warning("Could not load image: %s", image_path)
        return

    # Save anomaly map as numpy array
    np.save(str(out_dir / "anomaly_map.npy"), anomaly_map)

    # Create heatmap and overlay
    overlay_bgr, heatmap_bgr, amap_resized = make_heatmap_overlay(
        img_bgr, anomaly_map, thr_px=thr_px
    )

    # Annotate overlay with metadata
    gt_str = "anomaly" if label == 1 else "normal"
    pred_str = "anomaly" if pred_label == 1 else "normal"
    correct = "✓" if label == pred_label else "✗"
    annotation_lines = [
        f"{category} [{variant}]  {correct}",
        f"score={image_score:.4f}  thr={thr_img:.4f}",
        f"GT={gt_str}  Pred={pred_str}",
    ]
    _annotate_image(overlay_bgr, annotation_lines)

    cv2.imwrite(str(out_dir / "heatmap.png"), heatmap_bgr)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay_bgr)

    # Optional GT contour overlay
    if gt_mask is not None:
        overlay_gt = _draw_gt_contours(overlay_bgr, gt_mask)
        cv2.imwrite(str(out_dir / "overlay_with_gt.png"), overlay_gt)


# ── Text annotation helper ─────────────────────────────────────────────────────


def _annotate_image(
    img: np.ndarray,
    lines: list,
    font_scale: float = 0.45,
    thickness: int = 1,
    start_y: int = 18,
    line_spacing: int = 16,
) -> None:
    """Draw annotation text lines on ``img`` in-place (top-left corner)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y = start_y + i * line_spacing
        # Dark shadow for readability on any background
        cv2.putText(img, line, (9, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
