"""PatchCore VisA Inspector – Streamlit GUI.

Run from the repo root::

    streamlit run ui/app.py

Or from the ``ui/`` directory::

    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – make sure src/ is importable whether we run from repo root or
# from inside the ui/ directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # …/visa/ui
_REPO_ROOT = _THIS_DIR.parent                        # …/visa
for _p in (_REPO_ROOT, str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Streamlit import (must come before any st.* calls)
# ---------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="PatchCore VisA Inspector",
    layout="wide",
    page_icon="🔍",
)

# ---------------------------------------------------------------------------
# Optional heavy imports – surface friendly errors when packages are missing
# ---------------------------------------------------------------------------
_IMPORT_ERROR: str | None = None

try:
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image
    import cv2
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
except ImportError as exc:
    _IMPORT_ERROR = (
        f"Missing Python package: {exc}. "
        "Please install requirements before running the app."
    )

try:
    from src.artifacts import VARIANTS, load_thresholds, load_model_artifacts
    from src.calibration import load_calibrators
    from src.dataset import CATEGORIES
    from src.visualization import make_heatmap_overlay, tensor_to_numpy_img
except ImportError as exc:
    _IMPORT_ERROR = (
        f"Could not import from src/: {exc}\n"
        "Make sure you are running from the repo root or that src/ is on PYTHONPATH."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_VARIANT_LABELS = {
    "standard": "Standard PatchCore",
    "fr": "FR-PatchCore",
    "mask": "Masked PatchCore",
    "fr_mask": "FR+Masked PatchCore",
}
_LABEL_TO_VARIANT = {v: k for k, v in _VARIANT_LABELS.items()}

# Image sub-directory names used in the VisA test split
_IMG_TYPE_DIRS = {
    "Normal": "good",
    "Anomaly": "bad",
    "All": None,          # sentinel – scan both
}


# ===========================================================================
# Preprocessing & inference helpers
# ===========================================================================

def preprocess_image(pil_image: "Image.Image", image_size: int = 256) -> "torch.Tensor":
    """Convert a PIL image to a normalised (1, C, H, W) float tensor.

    Parameters
    ----------
    pil_image:
        Source image (any mode; converted to RGB internally).
    image_size:
        Target spatial resolution (both sides).

    Returns
    -------
    Tensor of shape ``(1, 3, image_size, image_size)`` ready for the backbone.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    img_rgb = pil_image.convert("RGB")
    tensor = transform(img_rgb)          # (C, H, W)
    return tensor.unsqueeze(0)           # (1, C, H, W)


@torch.no_grad()
def run_inference(
    model: "PatchCoreBase",  # type: ignore[name-defined]
    image_tensor: "torch.Tensor",
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Score a single pre-processed image tensor.

    The function bypasses the DataLoader-based ``score()`` method by directly
    calling the feature extractor and patch-distance computation that live on
    the model.

    Parameters
    ----------
    model:
        Fitted ``PatchCoreBase`` instance.
    image_tensor:
        ``(1, C, H, W)`` float tensor (output of :func:`preprocess_image`).
    device:
        Torch device string.

    Returns
    -------
    anomaly_map:
        2-D float32 numpy array of shape ``(image_size, image_size)``.
    image_score:
        Scalar anomaly score (max of the anomaly map).
    """
    dev = torch.device(device)
    image_tensor = image_tensor.to(dev)

    extractor = model._get_extractor(dev)
    f2, f3 = extractor(image_tensor)

    from src.patchcore import _patches_from_batch  # local helper
    patches, feat_h, feat_w = _patches_from_batch(f2, f3)
    img_patches = patches.cpu().numpy().astype(np.float32)  # (H*W, C)

    # Use the model's own distance method (handles FRPatchCore projection etc.).
    # image_path is only used by MaskedPatchCore to load a foreground mask from
    # disk.  For single-image upload inference we pass an empty string; masked
    # variants will fall back to all-foreground (the base-class behaviour) which
    # is safe but means foreground masking is bypassed for that path.
    image_path = ""
    patch_dists = model._patch_distances(img_patches, image_path, feat_h, feat_w)

    fg_mask = model._compute_foreground_mask(image_path, feat_h, feat_w)
    patch_dists = patch_dists * fg_mask.astype(np.float32)

    dist_map = patch_dists.reshape(feat_h, feat_w)
    anomaly_map = model._upsample_map(dist_map)   # (image_size, image_size)
    image_score = float(anomaly_map.max())

    return anomaly_map, image_score


def build_heatmap_overlay(
    pil_image: "Image.Image",
    anomaly_map: np.ndarray,
    thr_px: float | None,
    display_size: int = 256,
) -> np.ndarray:
    """Create a coloured heatmap overlay suitable for ``st.image``.

    Parameters
    ----------
    pil_image:
        Original PIL image.
    anomaly_map:
        2-D float anomaly map (output of :func:`run_inference`).
    thr_px:
        Pixel-level threshold.  Pixels below the threshold keep the original
        colour; pixels above are tinted with the heatmap colour.
    display_size:
        Resize the image to this square size before blending.

    Returns
    -------
    RGB uint8 numpy array of shape ``(display_size, display_size, 3)``.
    """
    img_rgb = pil_image.convert("RGB").resize((display_size, display_size))
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    overlay_bgr, _, _ = make_heatmap_overlay(img_bgr, anomaly_map, thr_px=thr_px)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


# ===========================================================================
# Cached resource loading
# ===========================================================================

@st.cache_resource(show_spinner="Loading model …")
def cached_load_model(
    artifacts_dir: str,
    variant: str,
    category: str,
    device: str,
) -> dict | None:
    """Load and cache a PatchCore model together with thresholds and calibrator.

    The cache key is ``(artifacts_dir, variant, category, device)``.

    Returns
    -------
    dict with keys ``'model'``, ``'thresholds'``, ``'calibrator'`` on success,
    or ``None`` if the artefact directory does not exist.
    """
    try:
        artifacts = load_model_artifacts(
            model_dir=artifacts_dir,
            variant=variant,
            category=category,
            device=device,
        )
        return artifacts
    except FileNotFoundError as exc:
        st.warning(f"⚠️ Model artefacts not found: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Failed to load model: {exc}")
        return None


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> dict:
    """Render the sidebar and return the user-selected configuration dict.

    Returns
    -------
    dict with keys:
        ``variant``, ``category``, ``thr_mode``, ``artifacts_dir``,
        ``test_data_dir``, ``device``
    """
    with st.sidebar:
        st.title("🔍 PatchCore VisA Inspector")

        # ── Model selection ────────────────────────────────────────────────
        st.header("Model Selection")

        basic_labels = [_VARIANT_LABELS["standard"], _VARIANT_LABELS["fr"]]
        show_advanced = st.checkbox("Show Advanced Models", value=False)
        if show_advanced:
            all_labels = list(_VARIANT_LABELS.values())
        else:
            all_labels = basic_labels

        variant_label = st.radio("Model Variant", all_labels, index=0)
        variant = _LABEL_TO_VARIANT[variant_label]

        # ── Category ──────────────────────────────────────────────────────
        st.header("Category")
        category = st.selectbox("VisA Category", CATEGORIES, index=0)

        # ── Threshold mode ─────────────────────────────────────────────────
        st.header("Threshold Mode")
        thr_mode = st.radio(
            "Threshold Mode",
            ["Best F1 (Recommended)", "Low False Alarm (FPR Target)"],
            index=0,
        )

        # ── Paths ──────────────────────────────────────────────────────────
        st.header("Paths")
        artifacts_dir = st.text_input(
            "Artifacts directory",
            value="./exports",
            help="Root directory that contains per-variant sub-directories.",
        )
        test_data_dir = st.text_input(
            "Test data directory",
            value="./data/visa/test",
            help="Root of VisA test split: <dir>/<category>/<good|bad>/…",
        )

        # ── Device ────────────────────────────────────────────────────────
        st.header("Compute")
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        device = st.selectbox("Device", device_options, index=0)

        # ── About ─────────────────────────────────────────────────────────
        st.divider()
        with st.expander("ℹ️ About"):
            st.markdown(
                """
**PatchCore** is a memory-bank based anomaly detection method.
It stores representative patch descriptors from normal training images
and scores test images by their distance to the nearest neighbours
in the bank.

This inspector supports four variants:
- **Standard** – baseline cosine-distance PatchCore
- **FR** – dimensionality-reduced via PCA
- **Masked** – foreground-only patch selection
- **FR+Masked** – both reductions combined

Scores are optionally calibrated to produce probabilities via
isotonic regression.
                """
            )

    return {
        "variant": variant,
        "category": category,
        "thr_mode": thr_mode,
        "artifacts_dir": artifacts_dir,
        "test_data_dir": test_data_dir,
        "device": device,
    }


# ===========================================================================
# Result display helper
# ===========================================================================

def display_inference_result(
    pil_image: "Image.Image",
    anomaly_map: np.ndarray,
    image_score: float,
    thresholds: dict | None,
    calibrator,
    thr_mode: str,
    gt_mask: np.ndarray | None = None,
) -> None:
    """Render the inference results (images + metric cards) in the main area.

    Parameters
    ----------
    pil_image:
        Original PIL image (used for overlay rendering).
    anomaly_map:
        2-D float anomaly map returned by :func:`run_inference`.
    image_score:
        Scalar anomaly score.
    thresholds:
        Category-level threshold dict from ``thresholds.json``, or ``None``.
    calibrator:
        Fitted ``ScoreCalibrator`` or ``None``.
    thr_mode:
        One of the two threshold mode strings shown in the sidebar.
    gt_mask:
        Optional ground-truth binary mask (same spatial resolution as the
        anomaly map) for test-set images.
    """
    # Select thresholds --------------------------------------------------------
    if thresholds:
        if "Best F1" in thr_mode:
            thr_img = thresholds.get("thr_img_f1")
            thr_px = thresholds.get("thr_px_f1")
        else:
            thr_img = thresholds.get("thr_img_fpr")
            thr_px = thresholds.get("thr_px_fpr")
    else:
        thr_img = thr_px = None

    # Prediction ----------------------------------------------------------------
    if thr_img is not None:
        predicted_anomaly = image_score > thr_img
    else:
        predicted_anomaly = None

    # Calibrated confidence ----------------------------------------------------
    p_anom: float | None = None
    if calibrator is not None:
        try:
            p_anom = float(calibrator.predict_proba([image_score])[0])
        except Exception:  # noqa: BLE001
            p_anom = None

    # ── Visual output ─────────────────────────────────────────────────────────
    n_cols = 3 if gt_mask is not None else 2
    img_cols = st.columns(n_cols)

    with img_cols[0]:
        st.subheader("Original")
        st.image(pil_image.convert("RGB").resize((256, 256)), use_container_width=True)

    with img_cols[1]:
        st.subheader("Heatmap Overlay")
        overlay_rgb = build_heatmap_overlay(pil_image, anomaly_map, thr_px=thr_px)
        st.image(overlay_rgb, use_container_width=True)

    if gt_mask is not None and len(img_cols) > 2:
        with img_cols[2]:
            st.subheader("GT Mask")
            mask_vis = (gt_mask * 255).astype(np.uint8)
            st.image(mask_vis, use_container_width=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.divider()
    metric_cols = st.columns(4)

    with metric_cols[0]:
        if predicted_anomaly is None:
            st.info("**Prediction**\nNo threshold available")
        elif predicted_anomaly:
            st.error("🚨 **ANOMALY**")
        else:
            st.success("✅ **NORMAL**")

    with metric_cols[1]:
        st.metric("Anomaly Score", f"{image_score:.4f}")
        if thr_img is not None:
            st.caption(f"Image Threshold: {thr_img:.4f}")
        if thr_px is not None:
            st.caption(f"Pixel Threshold: {thr_px:.4f}")

    with metric_cols[2]:
        if p_anom is not None:
            st.metric("P(Anomaly)", f"{p_anom:.1%}")
            st.metric("P(Normal)", f"{1.0 - p_anom:.1%}")
        else:
            st.caption("No calibrator loaded")

    with metric_cols[3]:
        if p_anom is not None:
            st.write("**Confidence**")
            st.write("Anomaly")
            st.progress(p_anom)
            st.write("Normal")
            st.progress(1.0 - p_anom)


# ===========================================================================
# Tab 1 – Browse Test Set
# ===========================================================================

def _collect_test_images(test_data_dir: str, category: str, img_type: str) -> list[Path]:
    """Return a sorted list of image paths for the selected image type.

    Parameters
    ----------
    test_data_dir:
        Root test directory (contains per-category sub-directories).
    category:
        VisA category name (must be one of the 12 known categories).
    img_type:
        One of ``"Normal"``, ``"Anomaly"``, or ``"All"``.
    """
    # Validate category against the known list to prevent directory traversal
    if category not in CATEGORIES:
        return []

    sub_dirs_map = {
        "Normal": ["good"],
        "Anomaly": ["bad"],
        "All": ["good", "bad"],
    }
    sub_dirs = sub_dirs_map.get(img_type, ["good", "bad"])

    base = Path(test_data_dir).resolve() / category
    if not base.is_dir():
        return []

    images: list[Path] = []
    for sub in sub_dirs:
        # sub is always from the fixed map above, so no traversal risk
        sub_path = base / sub
        if sub_path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                images.extend(sorted(sub_path.glob(ext)))
    return sorted(images)


def tab_browse_test_set(cfg: dict, artifacts: dict | None) -> None:
    """Render the 'Browse Test Set' tab.

    Parameters
    ----------
    cfg:
        Sidebar configuration dict.
    artifacts:
        Loaded model artefacts dict, or ``None`` when not available.
    """
    st.header("Browse Test Set")

    img_type = st.radio(
        "Image Type",
        ["Normal", "Anomaly", "All"],
        horizontal=True,
        key="browse_img_type",
    )

    images = _collect_test_images(cfg["test_data_dir"], cfg["category"], img_type)

    if not images:
        st.info(
            f"No images found in `{cfg['test_data_dir']}/{cfg['category']}/`. "
            "Check the **Test data directory** path in the sidebar."
        )
        return

    st.caption(f"Found **{len(images)}** images.")

    # Selectbox shows relative path from category dir for readability
    rel_labels = [str(p.relative_to(Path(cfg["test_data_dir"]) / cfg["category"])) for p in images]
    selected_idx = st.selectbox(
        "Selected Image",
        range(len(rel_labels)),
        format_func=lambda i: rel_labels[i],
        key="browse_selected",
    )
    selected_path = images[selected_idx]

    # Thumbnail preview
    try:
        thumb = Image.open(selected_path).convert("RGB")
        thumb.thumbnail((128, 128))
        st.image(thumb, caption=rel_labels[selected_idx])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not open image: {exc}")
        return

    if st.button("🔍 Analyze Image", key="browse_analyze", type="primary"):
        if artifacts is None:
            st.error("Model not loaded – cannot run inference.")
            return

        pil_image = Image.open(selected_path)

        with st.spinner("Running inference …"):
            try:
                tensor = preprocess_image(pil_image, image_size=artifacts["model"].image_size)
                anomaly_map, image_score = run_inference(
                    artifacts["model"], tensor, cfg["device"]
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Inference failed: {exc}")
                return

        # Persist results in session state so they survive re-renders
        st.session_state["browse_result"] = {
            "pil_image": pil_image,
            "anomaly_map": anomaly_map,
            "image_score": image_score,
            "image_path": selected_path,
        }

    # Display persisted result -------------------------------------------------
    if "browse_result" in st.session_state:
        res = st.session_state["browse_result"]
        st.divider()
        st.subheader("Inference Result")

        # Attempt to load GT mask (VisA convention: masks/ sibling of test/)
        gt_mask = _try_load_gt_mask(res["image_path"], cfg["test_data_dir"])

        display_inference_result(
            pil_image=res["pil_image"],
            anomaly_map=res["anomaly_map"],
            image_score=res["image_score"],
            thresholds=artifacts["thresholds"] if artifacts else None,
            calibrator=artifacts["calibrator"] if artifacts else None,
            thr_mode=cfg["thr_mode"],
            gt_mask=gt_mask,
        )


def _try_load_gt_mask(image_path: Path, test_data_dir: str) -> np.ndarray | None:
    """Try to locate and load the ground-truth mask for a VisA test image.

    VisA masks live under ``<data_root>/masks/<category>/<subdir>/<stem>.png``
    or next to the image with a ``_mask`` suffix.  Returns ``None`` when not
    found.
    """
    try:
        # Common VisA layout: data/visa/ground_truth/<category>/…/<stem>.png
        test_root = Path(test_data_dir).parent  # one level above 'test/'
        rel = image_path.relative_to(Path(test_data_dir))  # category/bad/stem.ext
        mask_candidates = [
            test_root / "ground_truth" / rel.with_suffix(".png"),
            test_root / "masks" / rel.with_suffix(".png"),
            image_path.parent / (image_path.stem + "_mask.png"),
        ]
        for cand in mask_candidates:
            if cand.exists():
                mask_pil = Image.open(cand).convert("L")
                mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
                return mask_arr
    except Exception:  # noqa: BLE001
        pass
    return None


# ===========================================================================
# Tab 2 – Upload Image
# ===========================================================================

def tab_upload_image(cfg: dict, artifacts: dict | None) -> None:
    """Render the 'Upload Image' tab.

    Parameters
    ----------
    cfg:
        Sidebar configuration dict.
    artifacts:
        Loaded model artefacts dict, or ``None`` when not available.
    """
    st.header("Upload Image")

    uploaded = st.file_uploader(
        "Choose an image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        key="upload_file",
    )

    if uploaded is None:
        st.info("Upload an image above to run anomaly detection.")
        return

    try:
        pil_image = Image.open(uploaded).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not open uploaded file: {exc}")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(pil_image.resize((128, 128)), caption="Uploaded image")
    with col2:
        st.write(f"**Size:** {pil_image.width} × {pil_image.height} px")
        st.write(f"**Mode:** {pil_image.mode}")

    if st.button("🔍 Analyze", key="upload_analyze", type="primary"):
        if artifacts is None:
            st.error("Model not loaded – cannot run inference.")
            return

        with st.spinner("Running inference …"):
            try:
                tensor = preprocess_image(pil_image, image_size=artifacts["model"].image_size)
                anomaly_map, image_score = run_inference(
                    artifacts["model"], tensor, cfg["device"]
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Inference failed: {exc}")
                return

        st.session_state["upload_result"] = {
            "pil_image": pil_image,
            "anomaly_map": anomaly_map,
            "image_score": image_score,
        }

    if "upload_result" in st.session_state:
        res = st.session_state["upload_result"]
        st.divider()
        st.subheader("Inference Result")
        display_inference_result(
            pil_image=res["pil_image"],
            anomaly_map=res["anomaly_map"],
            image_score=res["image_score"],
            thresholds=artifacts["thresholds"] if artifacts else None,
            calibrator=artifacts["calibrator"] if artifacts else None,
            thr_mode=cfg["thr_mode"],
        )


# ===========================================================================
# Tab 3 – Batch Analysis
# ===========================================================================

def tab_batch_analysis(cfg: dict, artifacts: dict | None) -> None:
    """Render the 'Batch Analysis' tab.

    Runs inference over every image in the selected category's test split and
    summarises the results.  If ground-truth labels are inferred from the
    directory structure (good / bad), simple performance metrics are shown.

    Parameters
    ----------
    cfg:
        Sidebar configuration dict.
    artifacts:
        Loaded model artefacts dict, or ``None`` when not available.
    """
    st.header("Batch Analysis")
    st.caption(
        "Run inference on all test images for the selected category. "
        "This may take a while depending on dataset size and hardware."
    )

    if artifacts is None:
        st.warning("Load a model first (check the Artifacts directory in the sidebar).")
        return

    if st.button("▶️ Run Batch Inference", key="batch_run", type="primary"):
        all_images = _collect_test_images(cfg["test_data_dir"], cfg["category"], "All")
        if not all_images:
            st.error("No test images found. Check the test data directory in the sidebar.")
            return

        progress = st.progress(0.0, text="Processing images …")
        scores: list[float] = []
        labels: list[int] = []
        paths: list[Path] = []
        anomaly_maps: list[np.ndarray] = []

        for idx, img_path in enumerate(all_images):
            try:
                pil_img = Image.open(img_path).convert("RGB")
                tensor = preprocess_image(pil_img, artifacts["model"].image_size)
                amap, score = run_inference(artifacts["model"], tensor, cfg["device"])
                label = 0 if img_path.parent.name == "good" else 1
            except Exception:  # noqa: BLE001
                continue

            scores.append(score)
            labels.append(label)
            paths.append(img_path)
            anomaly_maps.append(amap)
            progress.progress((idx + 1) / len(all_images), text=f"{idx + 1}/{len(all_images)}")

        progress.empty()

        st.session_state["batch_results"] = {
            "scores": scores,
            "labels": labels,
            "paths": paths,
            "anomaly_maps": anomaly_maps,
        }
        st.success(f"Done – processed **{len(scores)}** images.")

    if "batch_results" not in st.session_state:
        return

    res = st.session_state["batch_results"]
    scores = res["scores"]
    labels = res["labels"]
    paths = res["paths"]
    anomaly_maps = res["anomaly_maps"]

    if not scores:
        st.warning("No results to display.")
        return

    # ── Select thresholds ─────────────────────────────────────────────────────
    thr_img: float | None = None
    thr_px: float | None = None
    if artifacts["thresholds"]:
        thr_key_img = "thr_img_f1" if "Best F1" in cfg["thr_mode"] else "thr_img_fpr"
        thr_key_px = "thr_px_f1" if "Best F1" in cfg["thr_mode"] else "thr_px_fpr"
        thr_img = artifacts["thresholds"].get(thr_key_img)
        thr_px = artifacts["thresholds"].get(thr_key_px)

    predictions = [1 if s > thr_img else 0 for s in scores] if thr_img is not None else None

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Score Distribution")
    score_df = pd.DataFrame({
        "Image": [p.name for p in paths],
        "GT Label": ["Anomaly" if l == 1 else "Normal" for l in labels],
        "Score": scores,
        "Predicted": (
            ["Anomaly" if p == 1 else "Normal" for p in predictions]
            if predictions else ["–"] * len(scores)
        ),
    })
    st.dataframe(score_df, use_container_width=True, height=250)

    # ── Performance metrics ───────────────────────────────────────────────────
    if predictions and any(l == 1 for l in labels):
        st.subheader("Performance (image-level)")
        from sklearn.metrics import (  # type: ignore[import]
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score,
        )
        try:
            acc = accuracy_score(labels, predictions)
            prec = precision_score(labels, predictions, zero_division=0)
            rec = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")

            m_cols = st.columns(5)
            m_cols[0].metric("Accuracy", f"{acc:.3f}")
            m_cols[1].metric("Precision", f"{prec:.3f}")
            m_cols[2].metric("Recall", f"{rec:.3f}")
            m_cols[3].metric("F1", f"{f1:.3f}")
            m_cols[4].metric("AUROC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        except Exception:  # noqa: BLE001
            st.info("Could not compute performance metrics (sklearn not available).")

    # ── Score histogram ───────────────────────────────────────────────────────
    fig = px.histogram(
        score_df,
        x="Score",
        color="GT Label",
        barmode="overlay",
        opacity=0.7,
        title="Anomaly Score Distribution",
        color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
    )
    if thr_img is not None:
        fig.add_vline(
            x=thr_img,
            line_dash="dash",
            line_color="orange",
            annotation_text="Threshold",
        )
    st.plotly_chart(fig, use_container_width=True)

    # ── Sample heatmap grid ───────────────────────────────────────────────────
    st.subheader("Sample Heatmaps")
    n_samples = min(8, len(paths))
    sample_indices = list(range(0, len(paths), max(1, len(paths) // n_samples)))[:n_samples]
    hm_cols = st.columns(min(4, n_samples))

    for col_idx, img_idx in enumerate(sample_indices):
        col = hm_cols[col_idx % len(hm_cols)]
        try:
            pil_img = Image.open(paths[img_idx]).convert("RGB")
            overlay = build_heatmap_overlay(pil_img, anomaly_maps[img_idx], thr_px=thr_px)
            gt_str = "Anomaly" if labels[img_idx] == 1 else "Normal"
            score_str = f"{scores[img_idx]:.3f}"
            col.image(overlay, caption=f"{paths[img_idx].name}\n{gt_str} | {score_str}", use_container_width=True)
        except Exception:  # noqa: BLE001
            col.warning("Image unavailable")


# ===========================================================================
# Tab 4 – Metrics Dashboard
# ===========================================================================

def tab_metrics_dashboard(cfg: dict) -> None:
    """Render the 'Metrics Dashboard' tab.

    Loads and visualises the metrics stored in
    ``{artifacts_dir}/{variant}/thresholds.json``.

    Parameters
    ----------
    cfg:
        Sidebar configuration dict.
    """
    st.header("Metrics Dashboard")

    # Validate variant before constructing path (prevents path traversal)
    variant = cfg["variant"]
    if variant not in VARIANTS:
        st.error(f"Unknown model variant: {variant!r}")
        return

    thr_path = Path(cfg["artifacts_dir"]).resolve() / variant / "thresholds.json"

    if not thr_path.is_file():
        st.info(
            f"No thresholds file found at `{thr_path}`. "
            "Train and export models first."
        )
        return

    try:
        all_thr = load_thresholds(thr_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read thresholds: {exc}")
        return

    categories_data = all_thr.get("categories", {})
    if not categories_data:
        st.warning("Thresholds file contains no category data.")
        return

    # ── Build summary table ───────────────────────────────────────────────────
    rows = []
    for cat, data in categories_data.items():
        rows.append({
            "Category": cat,
            "Img AUROC": data.get("auroc_img", float("nan")),
            "Img AUPRC": data.get("auprc_img", float("nan")),
            "Px AUROC": data.get("auroc_px", float("nan")),
            "Px AUPRC": data.get("auprc_px", float("nan")),
            "Best-F1 Thr (img)": data.get("thr_img_f1", float("nan")),
            "FPR Thr (img)": data.get("thr_img_fpr", float("nan")),
            "Best-F1 Thr (px)": data.get("thr_px_f1", float("nan")),
            "FPR Thr (px)": data.get("thr_px_fpr", float("nan")),
        })

    df = pd.DataFrame(rows).set_index("Category")

    # Format numeric columns
    float_cols = [c for c in df.columns if df[c].dtype == float]
    st.dataframe(
        df.style.format({c: "{:.4f}" for c in float_cols}),
        use_container_width=True,
    )

    # ── Bar charts ─────────────────────────────────────────────────────────────
    st.subheader("Image-level AUROC per Category")
    fig_auroc = go.Figure()
    fig_auroc.add_trace(go.Bar(
        x=df.index.tolist(),
        y=df["Img AUROC"].tolist(),
        name="Img AUROC",
        marker_color="steelblue",
    ))
    fig_auroc.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Category",
        yaxis_title="AUROC",
        showlegend=False,
    )
    st.plotly_chart(fig_auroc, use_container_width=True)

    st.subheader("Pixel-level AUROC per Category")
    fig_px = go.Figure()
    fig_px.add_trace(go.Bar(
        x=df.index.tolist(),
        y=df["Px AUROC"].tolist(),
        name="Px AUROC",
        marker_color="darkorange",
    ))
    fig_px.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Category",
        yaxis_title="Pixel AUROC",
        showlegend=False,
    )
    st.plotly_chart(fig_px, use_container_width=True)

    # ── Mean metrics card ─────────────────────────────────────────────────────
    st.subheader("Mean Metrics")
    mean_cols = st.columns(4)
    for col, metric in zip(
        mean_cols,
        ["Img AUROC", "Img AUPRC", "Px AUROC", "Px AUPRC"],
    ):
        val = df[metric].mean()
        col.metric(f"Mean {metric}", f"{val:.4f}" if not np.isnan(val) else "N/A")


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> None:
    """Application entry point – assembles sidebar and tab layout."""

    # Surface import errors prominently before rendering anything else
    if _IMPORT_ERROR:
        st.error(f"❌ Import error\n\n```\n{_IMPORT_ERROR}\n```")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    cfg = render_sidebar()

    # ── Initialise session state keys ─────────────────────────────────────────
    for key in ("browse_result", "upload_result", "batch_results"):
        if key not in st.session_state:
            st.session_state[key] = None

    # Track whether the variant/category changed so we can reload the model
    model_key = (cfg["artifacts_dir"], cfg["variant"], cfg["category"], cfg["device"])
    prev_key = st.session_state.get("_model_key")
    if prev_key != model_key:
        # Clear stale inference results when the model changes
        for key in ("browse_result", "upload_result", "batch_results"):
            st.session_state[key] = None
        st.session_state["_model_key"] = model_key

    # ── Load model (cached) ───────────────────────────────────────────────────
    artifacts = cached_load_model(
        artifacts_dir=cfg["artifacts_dir"],
        variant=cfg["variant"],
        category=cfg["category"],
        device=cfg["device"],
    )

    # Status banner
    variant_label = _VARIANT_LABELS.get(cfg["variant"], cfg["variant"])
    if artifacts is not None:
        st.sidebar.success(f"✅ {variant_label} / {cfg['category']} loaded")
    else:
        st.sidebar.warning("⚠️ No model loaded")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗂️ Browse Test Set",
        "📤 Upload Image",
        "⚡ Batch Analysis",
        "📊 Metrics Dashboard",
    ])

    with tab1:
        tab_browse_test_set(cfg, artifacts)

    with tab2:
        tab_upload_image(cfg, artifacts)

    with tab3:
        tab_batch_analysis(cfg, artifacts)

    with tab4:
        tab_metrics_dashboard(cfg)


if __name__ == "__main__":
    main()
