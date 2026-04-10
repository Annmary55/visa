#!/usr/bin/env python3
"""PatchCore VisA Training & Evaluation Script – Kaggle-ready.

Usage examples
--------------
Run all variants on all categories (default):

    python train_kaggle.py --config config.yaml

Quick smoke-test (one category, one variant):

    python train_kaggle.py --dry-run --variants standard

Custom output directory and specific categories:

    python train_kaggle.py \\
        --output-dir /kaggle/working/exports \\
        --categories candle capsules cashew \\
        --variants standard fr

Pipeline summary
----------------
For each (variant, category) pair:
  1. Build train / val / test DataLoaders from the VisA dataset.
  2. Fit the PatchCore memory bank on the *train* split.
  3. Score the *val* split → compute decision thresholds & calibrator.
  4. Score the *test* split → compute final evaluation metrics.
  5. Optionally save heatmap visualisations for sampled images.
  6. Persist model artefacts, thresholds, calibrators, and metrics.

After all categories of a variant are done, a per-variant thresholds.json and
metrics summary are written.  Finally all artefacts are zipped for download.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Ensure the repo root is on the Python path ────────────────────────────────
# Works whether the script lives in <repo>/kaggle/ or is copied elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── Third-party (must be available in the Kaggle environment) ─────────────────
try:
    import torch
    import yaml
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Missing required package: {exc}.  "
        "Run: pip install torch pyyaml tqdm"
    )

# ── Internal modules ───────────────────────────────────────────────────────────
from src.artifacts import (
    VARIANTS,
    package_artifacts,
    save_metrics_summary,
    save_thresholds,
)
from src.calibration import fit_and_save_calibrators
from src.dataset import CATEGORIES, VisADataset, find_visa_root, get_dataloader
from src.metrics import compute_all_thresholds, compute_image_metrics
from src.patchcore import create_model
from src.visualization import save_image_results

# ── Logging configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_kaggle")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PatchCore VisA training and evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--visa-root",
        default=None,
        help="Explicit path to the VisA dataset root.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root export directory.  Overrides export.output_dir in config.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=None,
        help=f"Variants to train.  Defaults to all: {VARIANTS}.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        default=None,
        help="Categories to process.  Defaults to all 12 VisA categories.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Torch device.  Overrides training.device in config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Smoke-test mode: process only the *first* category of each "
            "selected variant, then exit."
        ),
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Configuration helpers
# ══════════════════════════════════════════════════════════════════════════════


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration and return as a nested dict."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config from %s", cfg_path)
    return cfg


def resolve_device(cfg_device: str, cli_device: Optional[str]) -> torch.device:
    """Choose the Torch device from CLI override → config → auto-detection."""
    choice = cli_device or cfg_device  # CLI wins
    if choice == "auto" or choice is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(choice)
    logger.info("Using device: %s", device)
    return device


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════


def _build_dataloaders(
    visa_root: str,
    category: str,
    cfg: Dict[str, Any],
    use_mask: bool,
) -> tuple:
    """Return (train_loader, val_loader, test_loader) for one category."""
    ds_cfg = cfg["dataset"]
    image_size: int = ds_cfg["image_size"]
    batch_size: int = ds_cfg["batch_size"]
    num_workers: int = cfg["training"]["num_workers"]

    def _make_loader(split: str, shuffle: bool) -> Any:
        ds = VisADataset(
            root=visa_root,
            category=category,
            split=split,
            image_size=image_size,
            use_mask=use_mask,
        )
        return get_dataloader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = _make_loader("train", shuffle=False)
    val_loader = _make_loader("val", shuffle=False)
    test_loader = _make_loader("test", shuffle=False)

    logger.info(
        "  %s splits → train=%d  val=%d  test=%d",
        category,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════


def _save_heatmaps(
    results: Dict[str, Any],
    output_dir: Path,
    category: str,
    variant: str,
    thr_img: float,
    thr_px: Optional[float],
    max_images: int,
    split_label: str,
) -> None:
    """Save heatmap visualisations for a subset of scored images.

    Parameters
    ----------
    results:
        Dict returned by ``model.score()``.
    output_dir:
        Root export directory.
    category, variant:
        Used for sub-directory naming and image annotation.
    thr_img:
        Image-level decision threshold.
    thr_px:
        Pixel-level threshold (may be ``None`` if no masks available).
    max_images:
        Hard cap on the number of images saved.
    split_label:
        ``'val'`` or ``'test'`` – appended to the output sub-directory name.
    """
    image_scores = results["image_scores"]
    anomaly_maps = results["anomaly_maps"]
    labels = results["labels"]
    masks = results["masks"]
    image_paths = results["image_paths"]

    n_save = min(len(image_paths), max_images)
    # Prefer an equal split of normal / anomaly images where possible
    anom_idx = [i for i, lb in enumerate(labels) if lb == 1]
    norm_idx = [i for i, lb in enumerate(labels) if lb == 0]
    half = max(1, n_save // 2)
    selected = (anom_idx[:half] + norm_idx[:half])[:n_save]

    # Output sub-directory includes the split name to avoid collisions
    split_output = output_dir / "heatmaps" / split_label

    for i in selected:
        score = image_scores[i]
        pred_label = 1 if score >= thr_img else 0
        gt_mask = masks[i] if (masks[i] is not None and masks[i].sum() > 0) else None

        try:
            save_image_results(
                output_dir=split_output,
                image_path=image_paths[i],
                category=category,
                variant=variant,
                anomaly_map=anomaly_maps[i],
                image_score=score,
                label=labels[i],
                pred_label=pred_label,
                thr_img=thr_img,
                thr_px=thr_px,
                gt_mask=gt_mask,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("    Heatmap save failed for %s: %s", image_paths[i], exc)


# ══════════════════════════════════════════════════════════════════════════════
# Per-category training logic
# ══════════════════════════════════════════════════════════════════════════════


def train_category(
    visa_root: str,
    category: str,
    variant: str,
    cfg: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Full pipeline for one (variant, category) pair.

    Returns
    -------
    dict
        Contains ``'thresholds'`` and ``'metrics'`` sub-dicts for the category,
        or ``None`` if the run failed.
    """
    t0 = time.perf_counter()
    model_cfg = cfg["model"]
    thr_cfg = cfg["thresholds"]
    training_cfg = cfg["training"]

    # Masked variants need ground-truth masks for foreground filtering
    use_mask = variant in ("mask", "fr_mask")

    logger.info("  → %s / %s", variant, category)

    # ── Build dataloaders ──────────────────────────────────────────────────────
    try:
        train_loader, val_loader, test_loader = _build_dataloaders(
            visa_root, category, cfg, use_mask=use_mask
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("    DataLoader build failed: %s", exc)
        return None

    # ── Create and fit model ───────────────────────────────────────────────────
    try:
        model = create_model(
            variant,
            coreset_ratio=model_cfg["coreset_ratio"],
            k=model_cfg["k_neighbors"],
            image_size=cfg["dataset"]["image_size"],
            pca_components=model_cfg.get("fr_pca_components", 256),
            alpha=model_cfg.get("fr_alpha", 0.5),
        )
        logger.info("    Fitting memory bank …")
        model.fit(train_loader, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.error("    Model fit failed: %s", exc)
        return None

    # ── Score validation split ─────────────────────────────────────────────────
    logger.info("    Scoring val split …")
    try:
        with torch.no_grad():
            val_results = model.score(val_loader, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.error("    Val scoring failed: %s", exc)
        return None

    val_scores = np.asarray(val_results["image_scores"])
    val_labels = np.asarray(val_results["labels"])
    val_maps = val_results["anomaly_maps"]
    val_masks = val_results["masks"]

    # Only pass pixel data to threshold computation when masks are meaningful
    has_px_data = use_mask and any(m.sum() > 0 for m in val_masks)

    # ── Compute thresholds on val split ───────────────────────────────────────
    try:
        thr_dict = compute_all_thresholds(
            scores=val_scores,
            labels=val_labels,
            anomaly_maps=val_maps if has_px_data else None,
            masks=val_masks if has_px_data else None,
            fpr_target=thr_cfg["fpr_target"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("    Threshold computation failed: %s", exc)
        return None

    thr_img = float(thr_dict.get("thr_img_fpr", thr_dict.get("thr_img_f1", 0.5)))
    thr_px = float(thr_dict["thr_px_fpr"]) if "thr_px_fpr" in thr_dict else None

    # ── Score test split ───────────────────────────────────────────────────────
    logger.info("    Scoring test split …")
    try:
        with torch.no_grad():
            test_results = model.score(test_loader, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.error("    Test scoring failed: %s", exc)
        return None

    test_scores = np.asarray(test_results["image_scores"])
    test_labels = np.asarray(test_results["labels"])
    test_maps = test_results["anomaly_maps"]
    test_masks = test_results["masks"]

    has_test_px = use_mask and any(m.sum() > 0 for m in test_masks)

    # ── Compute final test metrics ─────────────────────────────────────────────
    try:
        test_metrics = compute_all_thresholds(
            scores=test_scores,
            labels=test_labels,
            anomaly_maps=test_maps if has_test_px else None,
            masks=test_masks if has_test_px else None,
            fpr_target=thr_cfg["fpr_target"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("    Test metrics failed: %s", exc)
        test_metrics = {}

    # ── Save model artefacts ───────────────────────────────────────────────────
    model_save_dir = output_dir / variant / category
    try:
        model.save(str(model_save_dir))
        logger.info("    Model saved to %s", model_save_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("    Model save failed: %s", exc)

    # ── Heatmap visualisations ─────────────────────────────────────────────────
    max_hm = training_cfg.get("max_heatmaps_per_category", 20)

    if training_cfg.get("save_heatmaps_val", True):
        _save_heatmaps(
            results=val_results,
            output_dir=output_dir / variant / category,
            category=category,
            variant=variant,
            thr_img=thr_img,
            thr_px=thr_px,
            max_images=max_hm,
            split_label="val",
        )

    if training_cfg.get("save_heatmaps_test", True):
        _save_heatmaps(
            results=test_results,
            output_dir=output_dir / variant / category,
            category=category,
            variant=variant,
            thr_img=thr_img,
            thr_px=thr_px,
            max_images=max_hm,
            split_label="test",
        )

    elapsed = time.perf_counter() - t0
    img_auroc = test_metrics.get("image_auroc", float("nan"))
    logger.info(
        "    Done in %.1f s  |  test image-AUROC = %.4f",
        elapsed,
        img_auroc,
    )

    return {
        "thresholds": thr_dict,
        "metrics": test_metrics,
        # Raw scores needed for calibrator fitting (val split only)
        "val_scores": val_scores.tolist(),
        "val_labels": val_labels.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-variant orchestration
# ══════════════════════════════════════════════════════════════════════════════


def run_variant(
    variant: str,
    categories: List[str],
    visa_root: str,
    cfg: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    dry_run: bool,
) -> Dict[str, Any]:
    """Train and evaluate all categories for a single variant.

    Returns
    -------
    dict
        Nested ``{category: {thresholds, metrics}}`` mapping.
    """
    logger.info("═══ Variant: %s ═══", variant.upper())
    variant_results: Dict[str, Any] = {}

    # Accumulate val scores/labels for per-variant calibrator fitting
    all_val_scores: Dict[str, List[float]] = {}
    all_val_labels: Dict[str, List[int]] = {}

    cats_to_run = categories[:1] if dry_run else categories

    for category in tqdm(cats_to_run, desc=variant, unit="cat"):
        try:
            result = train_category(
                visa_root=visa_root,
                category=category,
                variant=variant,
                cfg=cfg,
                output_dir=output_dir,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  Category %s FAILED with unexpected error: %s", category, exc)
            result = None

        if result is None:
            variant_results[category] = {"thresholds": {}, "metrics": {}}
            continue

        variant_results[category] = {
            "thresholds": result["thresholds"],
            "metrics": result["metrics"],
        }
        all_val_scores[category] = result["val_scores"]
        all_val_labels[category] = result["val_labels"]

        # Free GPU memory between categories to avoid OOM on the next one
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Persist thresholds.json for this variant ───────────────────────────────
    thresholds_payload: Dict[str, Any] = {
        "fpr_target": cfg["thresholds"]["fpr_target"],
        "pixel_fpr_target": cfg["thresholds"]["pixel_fpr_target"],
        "categories": {
            cat: data["thresholds"]
            for cat, data in variant_results.items()
        },
    }
    save_thresholds(
        thresholds_payload,
        output_dir / variant / "thresholds.json",
    )

    # ── Fit and save calibrators (one per category) ────────────────────────────
    if all_val_scores:
        try:
            fit_and_save_calibrators(
                category_scores=all_val_scores,
                category_labels=all_val_labels,
                output_dir=output_dir,
                variant=variant,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  Calibrator fitting failed for variant %s: %s", variant, exc)

    # ── Per-variant metrics summary ────────────────────────────────────────────
    variant_metrics = {
        cat: data["metrics"] for cat, data in variant_results.items()
    }
    save_metrics_summary(
        variant_metrics,
        output_dir / variant / "metrics_summary.json",
    )

    return variant_results


# ══════════════════════════════════════════════════════════════════════════════
# Results table printer
# ══════════════════════════════════════════════════════════════════════════════


def _print_metrics_table(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a compact ASCII metrics table to stdout."""
    header = f"{'Variant':<12}  {'Category':<14}  {'ImgAUROC':>9}  {'PxAUROC':>9}  {'ImgF1':>7}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    for variant, cats in all_results.items():
        for cat, data in cats.items():
            m = data.get("metrics", {})
            img_auroc = m.get("image_auroc", float("nan"))
            px_auroc = m.get("pixel_auroc", float("nan"))
            img_f1 = m.get("img_f1", float("nan"))
            print(
                f"{variant:<12}  {cat:<14}  {img_auroc:>9.4f}  "
                f"{px_auroc:>9.4f}  {img_f1:>7.4f}"
            )
    print("═" * len(header) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── Load configuration ─────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Resolve settings (CLI overrides config) ────────────────────────────────
    output_dir = Path(args.output_dir or cfg["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants or VARIANTS
    categories = args.categories or cfg["dataset"]["categories"]

    device = resolve_device(
        cfg_device=cfg["training"]["device"],
        cli_device=args.device,
    )

    # ── Locate VisA dataset ────────────────────────────────────────────────────
    kaggle_search_paths = [
        "/kaggle/input/visa/visa_pytorch/",
        "/kaggle/input/visa/",
        "/kaggle/input/visad/",
        "/kaggle/input/visa-dataset/",
        str(Path.cwd()),
    ]
    visa_root = args.visa_root or find_visa_root(search_paths=kaggle_search_paths)
    if visa_root is None:
        logger.error(
            "Could not find the VisA dataset.  "
            "Pass --visa-root or place the data in a recognised directory."
        )
        sys.exit(1)
    logger.info("VisA dataset root: %s", visa_root)

    if args.dry_run:
        logger.warning("DRY-RUN mode: processing only the first category of each variant.")

    # ── Main training loop ─────────────────────────────────────────────────────
    t_start = time.perf_counter()
    all_results: Dict[str, Dict[str, Any]] = {}

    for variant in tqdm(variants, desc="Variants", unit="variant"):
        variant_results = run_variant(
            variant=variant,
            categories=categories,
            visa_root=visa_root,
            cfg=cfg,
            output_dir=output_dir,
            device=device,
            dry_run=args.dry_run,
        )
        all_results[variant] = variant_results

    # ── Overall metrics summary ────────────────────────────────────────────────
    overall_metrics: Dict[str, Any] = {
        variant: {
            cat: data["metrics"]
            for cat, data in cats.items()
        }
        for variant, cats in all_results.items()
    }
    save_metrics_summary(overall_metrics, output_dir / "metrics_summary.json")

    # ── ZIP packaging ──────────────────────────────────────────────────────────
    zip_name = cfg["export"]["zip_name"]
    zip_path = output_dir / zip_name
    try:
        package_artifacts(export_root=output_dir, output_zip_path=zip_path)
        logger.info("ZIP bundle: %s  (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    except Exception as exc:  # noqa: BLE001
        logger.error("Packaging failed: %s", exc)

    elapsed_total = time.perf_counter() - t_start
    logger.info("Total pipeline time: %.1f s (%.1f min)", elapsed_total, elapsed_total / 60)

    # ── Print summary table ────────────────────────────────────────────────────
    _print_metrics_table(all_results)


if __name__ == "__main__":
    main()
