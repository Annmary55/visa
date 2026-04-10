"""Evaluation metrics for PatchCore anomaly detection.

All functions operate on plain Python lists or NumPy arrays and have no
dependency on the rest of the project, so they can be imported and used
independently.

Image-level metrics treat each image as a sample; pixel-level metrics treat
each pixel of every anomaly map as a sample.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Type aliases
_Array = Union[np.ndarray, List[float]]


# ── Image-level metrics ────────────────────────────────────────────────────────


def compute_image_metrics(
    scores: _Array,
    labels: _Array,
    fpr_target: float = 0.05,
) -> Dict[str, float]:
    """Compute image-level anomaly detection metrics.

    Parameters
    ----------
    scores:
        Per-image anomaly scores (higher = more anomalous).
    labels:
        Ground-truth binary labels: 0 = normal, 1 = anomaly.
    fpr_target:
        FPR level at which TPR is reported (default 5 %).

    Returns
    -------
    dict with keys:
        * ``'image_auroc'``      – Area Under the ROC Curve
        * ``'image_auprc'``      – Area Under the Precision-Recall Curve
        * ``'fpr_at_tpr95'``     – FPR when TPR ≥ 95 %
        * ``'tpr_at_fpr_target'``– TPR at the requested ``fpr_target``
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    results: Dict[str, float] = {}

    if len(np.unique(labels)) < 2:
        # Can't compute ROC metrics with a single class
        results["image_auroc"] = float("nan")
        results["image_auprc"] = float("nan")
        results["fpr_at_tpr95"] = float("nan")
        results["tpr_at_fpr_target"] = float("nan")
        return results

    results["image_auroc"] = float(roc_auc_score(labels, scores))
    results["image_auprc"] = float(average_precision_score(labels, scores))
    results["fpr_at_tpr95"] = float(_fpr_at_tpr(scores, labels, tpr_target=0.95))
    results["tpr_at_fpr_target"] = float(
        _tpr_at_fpr(scores, labels, fpr_target=fpr_target)
    )
    return results


def _fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95) -> float:
    """Return FPR at the lowest threshold where TPR ≥ ``tpr_target``."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find the first index where TPR reaches the target
    idx = np.searchsorted(tpr_arr, tpr_target)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def _tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float = 0.05) -> float:
    """Return TPR at the threshold where FPR ≤ ``fpr_target``."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Last index where fpr ≤ target
    valid = fpr_arr <= fpr_target
    if not valid.any():
        return 0.0
    return float(tpr_arr[valid][-1])


# ── Pixel-level metrics ────────────────────────────────────────────────────────


def compute_pixel_metrics(
    anomaly_maps: List[np.ndarray],
    masks: List[np.ndarray],
) -> Dict[str, float]:
    """Compute pixel-level anomaly segmentation metrics.

    Parameters
    ----------
    anomaly_maps:
        List of ``(H, W)`` float arrays – per-pixel anomaly scores.
    masks:
        List of ``(H, W)`` binary arrays – ground-truth pixel labels.

    Returns
    -------
    dict with keys ``'pixel_auroc'`` and ``'pixel_auprc'``.
    """
    all_scores = np.concatenate([m.ravel() for m in anomaly_maps]).astype(np.float64)
    all_labels = np.concatenate([m.ravel() for m in masks]).astype(np.int32)

    results: Dict[str, float] = {}
    if len(np.unique(all_labels)) < 2:
        results["pixel_auroc"] = float("nan")
        results["pixel_auprc"] = float("nan")
        return results

    results["pixel_auroc"] = float(roc_auc_score(all_labels, all_scores))
    results["pixel_auprc"] = float(average_precision_score(all_labels, all_scores))
    return results


# ── Threshold search helpers ───────────────────────────────────────────────────


def find_best_f1_threshold(
    scores: _Array,
    labels: _Array,
) -> Dict[str, float]:
    """Find the decision threshold that maximises F1 score.

    Parameters
    ----------
    scores:
        Per-image anomaly scores.
    labels:
        Ground-truth binary labels.

    Returns
    -------
    dict with keys:
        ``threshold``, ``f1``, ``precision``, ``recall``,
        ``tp``, ``fp``, ``tn``, ``fn``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns arrays of length N+1; thresholds has length N
    # The last element corresponds to the highest recall (lowest threshold boundary)
    f1_arr = _safe_f1(precision_arr[:-1], recall_arr[:-1])

    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx])

    result = compute_confusion_at_threshold(scores, labels, best_thr)
    result["threshold"] = best_thr
    return result


def find_fpr_target_threshold(
    scores: _Array,
    labels: _Array,
    fpr_target: float = 0.01,
) -> Dict[str, float]:
    """Find the smallest threshold such that FPR ≤ ``fpr_target``.

    A lower threshold includes more positives (higher TPR) but also more FP.
    We scan from high to low to find the last threshold where FPR stays at or
    below the target.

    Parameters
    ----------
    scores:
        Per-image anomaly scores.
    labels:
        Ground-truth binary labels.
    fpr_target:
        Maximum acceptable FPR on normal samples.

    Returns
    -------
    dict with keys: ``threshold``, ``actual_fpr``, ``tpr_at_threshold``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    # fpr_arr is monotonically increasing; thresholds are sorted descending
    # Find the last position where FPR ≤ fpr_target
    valid = fpr_arr <= fpr_target
    if not valid.any():
        # No threshold achieves the desired FPR; return the most conservative one
        idx = 0
    else:
        idx = int(np.where(valid)[0][-1])

    thr = float(thresholds[idx]) if idx < len(thresholds) else float(thresholds[-1])
    return {
        "threshold": thr,
        "actual_fpr": float(fpr_arr[idx]),
        "tpr_at_threshold": float(tpr_arr[idx]),
    }


def find_best_pixel_f1_threshold(
    anomaly_maps: List[np.ndarray],
    masks: List[np.ndarray],
) -> Dict[str, float]:
    """Find the pixel-level decision threshold that maximises F1 score.

    Parameters
    ----------
    anomaly_maps:
        List of ``(H, W)`` float anomaly score arrays.
    masks:
        List of ``(H, W)`` binary ground-truth mask arrays.

    Returns
    -------
    Same dict as ``find_best_f1_threshold`` but operating at pixel level.
    """
    all_scores = np.concatenate([m.ravel() for m in anomaly_maps]).astype(np.float64)
    all_labels = np.concatenate([m.ravel() for m in masks]).astype(np.int32)
    return find_best_f1_threshold(all_scores, all_labels)


# ── Confusion matrix at a given threshold ─────────────────────────────────────


def compute_confusion_at_threshold(
    scores: _Array,
    labels: _Array,
    threshold: float,
) -> Dict[str, float]:
    """Compute confusion-matrix-derived metrics at a fixed threshold.

    Parameters
    ----------
    scores:
        Per-image anomaly scores.
    labels:
        Ground-truth binary labels.
    threshold:
        Decision boundary.  Samples with ``score ≥ threshold`` are predicted
        positive (anomaly).

    Returns
    -------
    dict with keys: ``tp``, ``fp``, ``tn``, ``fn``,
    ``precision``, ``recall``, ``f1``, ``accuracy``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    preds = (scores >= threshold).astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _safe_f1_scalar(precision, recall)
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


# ── All-in-one convenience function ───────────────────────────────────────────


def compute_all_thresholds(
    scores: _Array,
    labels: _Array,
    anomaly_maps: Optional[List[np.ndarray]] = None,
    masks: Optional[List[np.ndarray]] = None,
    fpr_target: float = 0.01,
) -> Dict[str, object]:
    """Compute every threshold and metric in one call.

    Parameters
    ----------
    scores:
        Per-image anomaly scores.
    labels:
        Ground-truth binary labels.
    anomaly_maps:
        Optional list of per-pixel anomaly maps for pixel-level metrics.
    masks:
        Optional list of ground-truth pixel masks.
    fpr_target:
        FPR target for ``find_fpr_target_threshold``.

    Returns
    -------
    dict with all image and pixel metrics and thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    out: Dict[str, object] = {}

    # Image-level AUROC / AUPRC
    out.update(compute_image_metrics(scores, labels))

    # Best F1 threshold (image level)
    f1_info = find_best_f1_threshold(scores, labels)
    out["thr_img_f1"] = f1_info["threshold"]
    out["img_f1"] = f1_info["f1"]
    out["img_precision"] = f1_info["precision"]
    out["img_recall"] = f1_info["recall"]

    # FPR-controlled threshold (image level)
    fpr_info = find_fpr_target_threshold(scores, labels, fpr_target=fpr_target)
    out["thr_img_fpr"] = fpr_info["threshold"]
    out["img_fpr_actual"] = fpr_info["actual_fpr"]
    out["img_tpr_at_fpr"] = fpr_info["tpr_at_threshold"]

    # Pixel-level metrics
    if anomaly_maps is not None and masks is not None:
        out.update(compute_pixel_metrics(anomaly_maps, masks))
        px_f1_info = find_best_pixel_f1_threshold(anomaly_maps, masks)
        out["thr_px_f1"] = px_f1_info["threshold"]
        out["px_f1"] = px_f1_info["f1"]
        out["px_precision"] = px_f1_info["precision"]
        out["px_recall"] = px_f1_info["recall"]

        all_scores_px = np.concatenate([m.ravel() for m in anomaly_maps]).astype(np.float64)
        all_labels_px = np.concatenate([m.ravel() for m in masks]).astype(np.int32)
        px_fpr_info = find_fpr_target_threshold(all_scores_px, all_labels_px, fpr_target=fpr_target)
        out["thr_px_fpr"] = px_fpr_info["threshold"]

    return out


# ── Internal helpers ───────────────────────────────────────────────────────────


def _safe_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Vectorised F1 with zero-division safety."""
    denom = precision + recall
    # Avoid dividing by zero: write result into a pre-allocated array so that
    # NumPy does not evaluate the division at positions where denom == 0.
    f1 = np.zeros_like(denom)
    mask = denom > 0
    f1[mask] = 2.0 * precision[mask] * recall[mask] / denom[mask]
    return f1


def _safe_f1_scalar(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0
