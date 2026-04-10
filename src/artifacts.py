"""Export / import utilities for PatchCore model artefacts.

Canonical on-disk layout for a single variant::

    <export_root>/
    └── <variant>/
        ├── config.json
        ├── thresholds.json
        ├── calibrators/
        │   ├── candle.pkl
        │   ├── capsules.pkl
        │   └── …
        └── <category>/
            ├── memory_bank.npz
            ├── config.json
            └── pca.pkl          (FRPatchCore only)

A summary of metrics across all variants and categories is stored at::

    <export_root>/metrics_summary.json

A ZIP bundle of the entire export root is provided for easy download from
Kaggle notebooks.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Public constants ───────────────────────────────────────────────────────────

VARIANTS: List[str] = ["standard", "fr", "mask", "fr_mask"]

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


# ── Threshold I/O ──────────────────────────────────────────────────────────────


def save_thresholds(thresholds_dict: Dict[str, Any], output_path: str | Path) -> None:
    """Persist the threshold dictionary as a JSON file.

    Expected structure::

        {
          "fpr_target": 0.01,
          "categories": {
            "candle": {
              "thr_img_f1":    float,
              "thr_img_fpr":   float,
              "thr_px_f1":     float,
              "thr_px_fpr":    float,
              "img_auroc":     float,
              "img_auprc":     float,
              "pixel_auroc":   float,
              "pixel_auprc":   float,
              "img_f1":        float,
              "img_precision": float,
              "img_recall":    float
            },
            …
          }
        }

    Parameters
    ----------
    thresholds_dict:
        Dictionary conforming to the structure above.
    output_path:
        Destination file path (e.g. ``exports/standard/thresholds.json``).
        Parent directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds_dict, indent=2, default=_json_default))
    logger.info("Saved thresholds to %s", output_path)


def load_thresholds(path: str | Path) -> Dict[str, Any]:
    """Load and return the thresholds dictionary from a JSON file.

    Parameters
    ----------
    path:
        Path to a JSON file previously written by ``save_thresholds``.

    Returns
    -------
    dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")
    return json.loads(path.read_text())


# ── Metrics summary ────────────────────────────────────────────────────────────


def save_metrics_summary(
    metrics_dict: Dict[str, Any], output_path: str | Path
) -> None:
    """Persist a full metrics summary (all variants × all categories) as JSON.

    Parameters
    ----------
    metrics_dict:
        Arbitrarily nested dict of metrics.  Values that are not JSON-native
        (e.g. ``np.float32``) are coerced to plain Python floats.
    output_path:
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_dict, indent=2, default=_json_default))
    logger.info("Saved metrics summary to %s", output_path)


# ── ZIP packaging ──────────────────────────────────────────────────────────────


def package_artifacts(
    export_root: str | Path,
    output_zip_path: str | Path,
) -> None:
    """Create a ZIP archive of all artefacts for easy download from Kaggle.

    The archive preserves the relative directory structure rooted at
    ``export_root``.  Only the following file types are included to keep
    the archive size manageable:

    * ``*.npz``  – memory banks
    * ``*.pkl``  – calibrators / PCA models
    * ``*.json`` – configuration and thresholds
    * ``*.npy``  – anomaly map arrays (optional artefacts)

    Parameters
    ----------
    export_root:
        Root directory that holds all variant sub-directories.
    output_zip_path:
        Destination ``.zip`` file path.
    """
    export_root = Path(export_root)
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    _INCLUDE_SUFFIXES = {".npz", ".pkl", ".json", ".npy"}
    file_count = 0

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(export_root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _INCLUDE_SUFFIXES:
                continue
            arcname = file_path.relative_to(export_root)
            zf.write(file_path, arcname)
            file_count += 1

    logger.info(
        "Packaged %d files from %s → %s", file_count, export_root, output_zip_path
    )


# ── Full model loader ──────────────────────────────────────────────────────────


def load_model_artifacts(
    model_dir: str | Path,
    variant: str,
    category: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a fitted PatchCore model, thresholds, and calibrator for inference.

    Assumes the layout described in the module docstring.

    Parameters
    ----------
    model_dir:
        Root export directory (parent of the variant directory).
    variant:
        One of ``VARIANTS``.
    category:
        VisA category name.
    device:
        Torch device string (e.g. ``'cpu'``, ``'cuda'``).

    Returns
    -------
    dict with keys:

    * ``'model'``      – loaded ``PatchCoreBase`` instance
    * ``'thresholds'`` – category-level threshold dict (from thresholds.json)
    * ``'calibrator'`` – ``ScoreCalibrator`` instance (or ``None`` if not found)
    """
    from .patchcore import create_model  # local import to avoid circular deps
    from .calibration import ScoreCalibrator

    model_dir = Path(model_dir)
    variant_dir = model_dir / variant
    category_dir = variant_dir / category

    if not category_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {category_dir}")

    # Load model
    model = create_model(variant)
    model.load(str(category_dir))

    # Load thresholds
    thr_path = variant_dir / "thresholds.json"
    thresholds: Optional[Dict] = None
    if thr_path.exists():
        all_thresholds = load_thresholds(thr_path)
        thresholds = all_thresholds.get("categories", {}).get(category)

    # Load calibrator
    calibrator: Optional[ScoreCalibrator] = None
    cal_path = variant_dir / "calibrators" / f"{category}.pkl"
    if cal_path.exists():
        calibrator = ScoreCalibrator.load(cal_path)

    return {
        "model": model,
        "thresholds": thresholds,
        "calibrator": calibrator,
    }


# ── JSON serialisation helper ──────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """Coerce non-JSON-native objects to serialisable Python types."""
    import numpy as np  # lazy import

    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")
