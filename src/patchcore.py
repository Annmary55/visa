"""PatchCore anomaly detection – four variants for the VisA benchmark.

Architecture overview
---------------------
All variants share a common base:

1. **Feature extraction** – WideResNet-50-2 backbone; intermediate feature maps
   from ``layer2`` (stride-8, 512-ch) and ``layer3`` (stride-16, 1024-ch) are
   captured via forward hooks.
2. **Patch aggregation** – ``layer3`` is upsampled to the spatial size of
   ``layer2``, the two tensors are concatenated along the channel axis, and a
   3×3 average-pool kernel (stride 1, padding 1) is applied.  Each spatial
   location yields one 1536-dimensional patch descriptor.
3. **Memory bank** – All patch descriptors collected over the training set are
   (optionally) sub-sampled with a greedy k-centre coreset algorithm, then
   stored.
4. **Scoring** – For each test patch the distance to its k-th nearest neighbour
   (k=5) in the memory bank is computed.  Distances are reshaped to the spatial
   grid and up-sampled to the original ``image_size``; the per-image anomaly
   score is the maximum pixel-level distance.

The four variants
-----------------
* ``StandardPatchCore``  – vanilla PatchCore, no modifications.
* ``FRPatchCore``        – adds PCA projection + reconstruction-error scoring.
* ``MaskedPatchCore``    – foreground-mask filtering; ignores background patches.
* ``FRMaskedPatchCore``  – combines both FR and masking via multiple inheritance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_BACKBONE_LAYERS = ("layer2", "layer3")
# Output channels: layer2 = 512, layer3 = 1024  →  concat = 1536
_FEATURE_DIM = 1536
# Default coreset sub-sampling ratio (10 % of all patches)
_DEFAULT_CORESET_RATIO = 0.10
# k-NN neighbours for anomaly scoring
_KNN_K = 5
# Avg-pool patch neighbourhood
_POOL_KERNEL = 3
_POOL_PADDING = 1


# ── Feature extraction ─────────────────────────────────────────────────────────


class _FeatureExtractor(torch.nn.Module):
    """Thin wrapper around WideResNet-50-2 that returns multi-scale features.

    Only ``layer2`` and ``layer3`` activations are returned; the remainder of
    the network is not evaluated, which saves ~40 % of compute time.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        # Keep only the layers we need (up to layer3)
        self.layer0 = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        # Freeze all parameters – we only use the backbone as a fixed feature extractor
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)   # (B, 512, H/8,  W/8)
        f3 = self.layer3(f2)  # (B, 1024, H/16, W/16)
        return f2, f3


# ── Patch aggregation ──────────────────────────────────────────────────────────


def _aggregate_patches(f2: Tensor, f3: Tensor) -> Tensor:
    """Upsample f3 → f2 spatial size, concatenate, apply neighbourhood avg-pool.

    Returns
    -------
    Tensor of shape ``(B, 1536, H, W)`` where H, W are the spatial dimensions
    of ``f2``.
    """
    h, w = f2.shape[-2], f2.shape[-1]
    f3_up = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=False)
    combined = torch.cat([f2, f3_up], dim=1)  # (B, 1536, H, W)
    pooled = F.avg_pool2d(combined, kernel_size=_POOL_KERNEL, padding=_POOL_PADDING, stride=1)
    return pooled  # (B, 1536, H, W)


def _patches_from_batch(f2: Tensor, f3: Tensor) -> Tuple[Tensor, int, int]:
    """Return ``(patches, h, w)`` where patches is ``(B*H*W, C)``."""
    pooled = _aggregate_patches(f2, f3)
    b, c, h, w = pooled.shape
    patches = pooled.permute(0, 2, 3, 1).reshape(b * h * w, c)
    return patches, h, w


# ── Greedy coreset sampling ────────────────────────────────────────────────────


def _greedy_coreset(features: np.ndarray, target_size: int, seed: int = 0) -> np.ndarray:
    """k-centre greedy coreset algorithm.

    Iteratively selects the point that is farthest from the current coreset,
    starting from a randomly chosen seed point.

    Parameters
    ----------
    features:
        Array of shape ``(N, D)``.
    target_size:
        Desired number of coreset points.
    seed:
        Random seed for the initial point selection.

    Returns
    -------
    np.ndarray of shape ``(target_size, D)``.
    """
    n = len(features)
    target_size = min(target_size, n)
    if target_size >= n:
        return features

    rng = np.random.default_rng(seed)
    selected: List[int] = [int(rng.integers(0, n))]
    # min distance from each point to the coreset
    min_dist = np.full(n, np.inf, dtype=np.float32)

    for _ in range(target_size - 1):
        last = features[selected[-1]]  # (D,)
        # Update min distances using the most recently added point
        dists = np.linalg.norm(features - last, axis=1).astype(np.float32)
        np.minimum(min_dist, dists, out=min_dist)
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    return features[selected]


# ── Base class ─────────────────────────────────────────────────────────────────


class PatchCoreBase:
    """Base class shared by all PatchCore variants.

    Parameters
    ----------
    image_size:
        Spatial resolution that the dataset delivers (both H and W).
    coreset_ratio:
        Fraction of patches to keep via greedy coreset sub-sampling.
        Set to ``1.0`` to disable sub-sampling.
    knn_k:
        Number of nearest neighbours used for anomaly scoring.
    """

    def __init__(
        self,
        image_size: int = 256,
        coreset_ratio: float = _DEFAULT_CORESET_RATIO,
        knn_k: int = _KNN_K,
    ) -> None:
        self.image_size = image_size
        self.coreset_ratio = coreset_ratio
        self.knn_k = knn_k

        self._extractor: Optional[_FeatureExtractor] = None
        self.memory_bank: Optional[np.ndarray] = None  # (M, C)
        # Spatial grid size inferred during fit
        self._feat_h: Optional[int] = None
        self._feat_w: Optional[int] = None

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _get_extractor(self, device: torch.device) -> _FeatureExtractor:
        if self._extractor is None:
            self._extractor = _FeatureExtractor().to(device).eval()
        return self._extractor.to(device)

    @torch.no_grad()
    def _extract_features(
        self, dataloader: DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, int, int, List[str], List[int], List[np.ndarray]]:
        """Run all images through the backbone and collect patch descriptors.

        Returns
        -------
        all_patches:
            Shape ``(N_total_patches, C)``.
        feat_h, feat_w:
            Spatial grid dimensions.
        image_paths, labels:
            Per-image metadata.
        masks_list:
            List of numpy arrays (``H_img × W_img``) ground-truth masks.
        """
        extractor = self._get_extractor(device)
        all_patches_list: List[np.ndarray] = []
        image_paths: List[str] = []
        labels: List[int] = []
        masks_list: List[np.ndarray] = []
        feat_h = feat_w = 0

        for batch in dataloader:
            images: Tensor = batch["image"].to(device)
            f2, f3 = extractor(images)
            patches, h, w = _patches_from_batch(f2, f3)
            feat_h, feat_w = h, w

            patches_np = patches.cpu().numpy().astype(np.float32)
            b = images.shape[0]
            all_patches_list.append(patches_np)

            image_paths.extend(batch["image_path"])
            labels.extend(batch["label"].tolist())
            for mask_t in batch["mask"]:
                masks_list.append(mask_t.squeeze(0).cpu().numpy())

        all_patches = np.concatenate(all_patches_list, axis=0)  # (N*H*W, C)
        return all_patches, feat_h, feat_w, image_paths, labels, masks_list

    # ------------------------------------------------------------------
    # Foreground masking (override point for MaskedPatchCore)
    # ------------------------------------------------------------------

    def _compute_foreground_mask(self, image_path: str, feat_h: int, feat_w: int) -> np.ndarray:
        """Return a boolean array of shape ``(feat_h * feat_w,)`` – all True by default."""
        return np.ones(feat_h * feat_w, dtype=bool)

    # ------------------------------------------------------------------
    # Memory bank construction
    # ------------------------------------------------------------------

    def _build_memory_bank(
        self, all_patches: np.ndarray, fg_masks_per_image: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """Optionally filter patches by foreground mask and sub-sample.

        Parameters
        ----------
        all_patches:
            All patch descriptors ``(N_images * H * W, C)``.
        fg_masks_per_image:
            If provided, a list of length ``N_images`` each of shape
            ``(H * W,)`` boolean; only ``True`` patches are kept.

        Returns
        -------
        memory_bank:
            Shape ``(M, C)`` after optional filtering and coreset sampling.
        """
        if fg_masks_per_image is not None:
            # Stack foreground masks and filter
            fg_combined = np.concatenate(fg_masks_per_image, axis=0)  # (N*H*W,)
            filtered = all_patches[fg_combined]
        else:
            filtered = all_patches

        if self.coreset_ratio < 1.0:
            target = max(1, int(len(filtered) * self.coreset_ratio))
            logger.info(
                "Coreset sub-sampling: %d → %d patches (ratio=%.2f)",
                len(filtered), target, self.coreset_ratio,
            )
            filtered = _greedy_coreset(filtered, target_size=target)

        return filtered.astype(np.float32)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, dataloader: DataLoader, device: torch.device | str = "cpu") -> None:
        """Build the memory bank from training images.

        Parameters
        ----------
        dataloader:
            DataLoader over the *training* split (``shuffle=False`` recommended
            so that patch order is deterministic for coreset sampling).
        device:
            Torch device string or object.
        """
        device = torch.device(device)
        logger.info("Fitting %s on device=%s", self.__class__.__name__, device)

        all_patches, feat_h, feat_w, image_paths, _, _ = self._extract_features(
            dataloader, device
        )
        self._feat_h = feat_h
        self._feat_w = feat_w

        # Subclass hook: build per-image foreground masks if needed
        fg_masks = self._get_training_fg_masks(all_patches, image_paths, feat_h, feat_w)

        self.memory_bank = self._build_memory_bank(all_patches, fg_masks)
        logger.info("Memory bank shape: %s", self.memory_bank.shape)

        # Subclass hook: post-fit operations (e.g. PCA fitting)
        self._post_fit(self.memory_bank)

    def _get_training_fg_masks(
        self,
        all_patches: np.ndarray,
        image_paths: List[str],
        feat_h: int,
        feat_w: int,
    ) -> Optional[List[np.ndarray]]:
        """Override in MaskedPatchCore to return per-image foreground masks."""
        return None

    def _post_fit(self, memory_bank: np.ndarray) -> None:
        """Override in FRPatchCore to fit PCA after memory bank is built."""
        pass

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(
        self, dataloader: DataLoader, device: torch.device | str = "cpu"
    ) -> Dict[str, Any]:
        """Compute image-level anomaly scores and pixel-level anomaly maps.

        Parameters
        ----------
        dataloader:
            DataLoader over the split to evaluate (val or test).
        device:
            Torch device string or object.

        Returns
        -------
        dict with keys:
            * ``'image_scores'`` – list of float, one per image
            * ``'anomaly_maps'`` – list of np.ndarray ``(image_size, image_size)``
            * ``'labels'`` – list of int (0=normal, 1=anomaly)
            * ``'masks'`` – list of np.ndarray ``(image_size, image_size)``
            * ``'image_paths'`` – list of str
        """
        assert self.memory_bank is not None, "Call fit() before score()."
        device = torch.device(device)

        all_patches, feat_h, feat_w, image_paths, labels, masks_list = (
            self._extract_features(dataloader, device)
        )
        n_images = len(image_paths)
        patches_per_img = feat_h * feat_w

        image_scores: List[float] = []
        anomaly_maps: List[np.ndarray] = []

        for i in range(n_images):
            start = i * patches_per_img
            end = start + patches_per_img
            img_patches = all_patches[start:end]  # (H*W, C)

            # Compute patch-level distances
            patch_dists = self._patch_distances(img_patches, image_paths[i], feat_h, feat_w)

            # Apply foreground masking in output (background → 0)
            fg_mask = self._compute_foreground_mask(image_paths[i], feat_h, feat_w)
            patch_dists = patch_dists * fg_mask.astype(np.float32)

            # Reshape to spatial grid, upsample to image_size
            dist_map = patch_dists.reshape(feat_h, feat_w)
            amap = self._upsample_map(dist_map)  # (image_size, image_size)

            image_scores.append(float(amap.max()))
            anomaly_maps.append(amap)

        return {
            "image_scores": image_scores,
            "anomaly_maps": anomaly_maps,
            "labels": labels,
            "masks": [
                self._upsample_mask(m) for m in masks_list
            ],
            "image_paths": image_paths,
        }

    # ------------------------------------------------------------------
    # Distance computation (override point for FRPatchCore)
    # ------------------------------------------------------------------

    def _patch_distances(
        self,
        img_patches: np.ndarray,
        image_path: str,
        feat_h: int,
        feat_w: int,
    ) -> np.ndarray:
        """Return per-patch anomaly distances using k-NN in memory bank.

        Parameters
        ----------
        img_patches:
            ``(H*W, C)`` patch descriptors for a single image.
        image_path:
            Originating file path (used for masking in subclasses).
        feat_h, feat_w:
            Spatial grid dimensions.

        Returns
        -------
        np.ndarray of shape ``(H*W,)`` – distance to kth nearest neighbour.
        """
        dists = _batch_knn_distances(img_patches, self.memory_bank, k=self.knn_k)
        return dists  # (H*W,)

    # ------------------------------------------------------------------
    # Upsample helpers
    # ------------------------------------------------------------------

    def _upsample_map(self, dist_map: np.ndarray) -> np.ndarray:
        """Upsample ``(H, W)`` distance map to ``(image_size, image_size)``."""
        t = torch.from_numpy(dist_map).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        up = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear",
                           align_corners=False)
        return up.squeeze().numpy()

    def _upsample_mask(self, mask: np.ndarray) -> np.ndarray:
        """Upsample a GT mask to ``(image_size, image_size)`` if needed."""
        if mask.shape == (self.image_size, self.image_size):
            return mask
        t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        up = F.interpolate(t, size=(self.image_size, self.image_size), mode="nearest")
        return (up.squeeze().numpy() > 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the memory bank and configuration.

        Files created:

        * ``<path>/memory_bank.npz`` – compressed numpy array
        * ``<path>/config.json``     – hyperparameters
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.memory_bank is not None, "Nothing to save – call fit() first."
        np.savez_compressed(path / "memory_bank.npz", memory_bank=self.memory_bank)
        config = {
            "class": self.__class__.__name__,
            "image_size": self.image_size,
            "coreset_ratio": self.coreset_ratio,
            "knn_k": self.knn_k,
            "feat_h": self._feat_h,
            "feat_w": self._feat_w,
        }
        config.update(self._extra_config())
        (path / "config.json").write_text(json.dumps(config, indent=2))
        logger.info("Saved %s to %s", self.__class__.__name__, path)

    def load(self, path: str | Path) -> None:
        """Load the memory bank and configuration from a saved directory."""
        path = Path(path)
        data = np.load(path / "memory_bank.npz")
        self.memory_bank = data["memory_bank"]
        config = json.loads((path / "config.json").read_text())
        self.image_size = config["image_size"]
        self.coreset_ratio = config["coreset_ratio"]
        self.knn_k = config["knn_k"]
        self._feat_h = config.get("feat_h")
        self._feat_w = config.get("feat_w")
        self._load_extra(path, config)
        logger.info("Loaded %s from %s", self.__class__.__name__, path)

    def _extra_config(self) -> Dict:
        """Subclasses return additional config entries to serialize."""
        return {}

    def _load_extra(self, path: Path, config: Dict) -> None:
        """Subclasses load additional artifacts (e.g. PCA model)."""
        pass


# ── k-NN distance helper ───────────────────────────────────────────────────────


def _batch_knn_distances(
    queries: np.ndarray, bank: np.ndarray, k: int = 5, chunk_size: int = 1024
) -> np.ndarray:
    """Return distance to the k-th nearest neighbour for each query.

    Computed in chunks to avoid allocating a full ``(N_query, N_bank)`` matrix.

    Parameters
    ----------
    queries:
        ``(Q, D)`` query vectors.
    bank:
        ``(M, D)`` memory bank.
    k:
        Number of nearest neighbours.
    chunk_size:
        Number of queries processed at once.

    Returns
    -------
    np.ndarray of shape ``(Q,)`` – kth-NN distance for each query.
    """
    q = len(queries)
    kth_dists = np.empty(q, dtype=np.float32)

    for start in range(0, q, chunk_size):
        end = min(start + chunk_size, q)
        chunk = queries[start:end]  # (C, D)
        # Squared Euclidean distance via expansion: ||q-b||^2 = ||q||^2 + ||b||^2 - 2<q,b>
        # einsum avoids materialising the full squared arrays, improving numerical stability.
        q_sq = np.einsum("ij,ij->i", chunk, chunk)[:, None]  # (C, 1)
        b_sq = np.einsum("ij,ij->i", bank, bank)[None, :]    # (1, M)
        cross = chunk @ bank.T                                 # (C, M)
        sq_dists = np.maximum(q_sq + b_sq - 2.0 * cross, 0.0)   # (C, M)

        actual_k = min(k, sq_dists.shape[1])
        if actual_k == sq_dists.shape[1]:
            kth_sq = sq_dists.max(axis=1)
        else:
            # Partial sort – O(M) instead of O(M log M)
            kth_sq = np.partition(sq_dists, actual_k - 1, axis=1)[:, actual_k - 1]

        kth_dists[start:end] = np.sqrt(np.maximum(kth_sq, 0.0))

    return kth_dists


# ── Variant 1: StandardPatchCore ──────────────────────────────────────────────


class StandardPatchCore(PatchCoreBase):
    """Vanilla PatchCore with no modifications.

    Uses all patches from training images (subject to coreset sub-sampling).
    """

    pass


# ── Variant 2: FRPatchCore ────────────────────────────────────────────────────


class FRPatchCore(PatchCoreBase):
    """Feature-Reconstruction PatchCore.

    After the memory bank is built, a PCA model is fitted on the stored
    descriptors.  During scoring, each test patch is projected into PCA space
    and reconstructed; the reconstruction error is combined with the k-NN
    distance:

        score = alpha * knn_dist + (1 - alpha) * recon_error

    Parameters
    ----------
    pca_components:
        Number of PCA components (default 256).
    alpha:
        Mixing weight between k-NN distance and reconstruction error (default 0.5).
    """

    def __init__(
        self,
        image_size: int = 256,
        coreset_ratio: float = _DEFAULT_CORESET_RATIO,
        knn_k: int = _KNN_K,
        pca_components: int = 256,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(image_size=image_size, coreset_ratio=coreset_ratio, knn_k=knn_k)
        self.pca_components = pca_components
        self.alpha = alpha
        self._pca = None  # sklearn PCA fitted during _post_fit

    # ------------------------------------------------------------------

    def _post_fit(self, memory_bank: np.ndarray) -> None:
        """Fit PCA on the memory bank descriptors."""
        from sklearn.decomposition import PCA  # lazy import

        n_components = min(self.pca_components, memory_bank.shape[1], memory_bank.shape[0])
        logger.info("Fitting PCA with %d components on %s points", n_components, len(memory_bank))
        self._pca = PCA(n_components=n_components, random_state=0)
        self._pca.fit(memory_bank)

    # ------------------------------------------------------------------

    def _patch_distances(
        self,
        img_patches: np.ndarray,
        image_path: str,
        feat_h: int,
        feat_w: int,
    ) -> np.ndarray:
        """Combine k-NN distance with PCA reconstruction error."""
        knn_dists = super()._patch_distances(img_patches, image_path, feat_h, feat_w)

        if self._pca is not None:
            projected = self._pca.transform(img_patches)           # (H*W, K)
            reconstructed = self._pca.inverse_transform(projected)  # (H*W, C)
            recon_error = np.linalg.norm(img_patches - reconstructed, axis=1).astype(np.float32)
            # Normalise reconstruction error to a comparable scale
            recon_std = recon_error.std() + 1e-8
            recon_norm = recon_error / recon_std
            knn_std = knn_dists.std() + 1e-8
            knn_norm = knn_dists / knn_std
            return (self.alpha * knn_norm + (1.0 - self.alpha) * recon_norm).astype(np.float32)

        return knn_dists

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib  # lazy import
        super().save(path)
        if self._pca is not None:
            joblib.dump(self._pca, Path(path) / "pca.pkl")

    def load(self, path: str | Path) -> None:
        import joblib  # lazy import
        super().load(path)
        pca_path = Path(path) / "pca.pkl"
        if pca_path.exists():
            self._pca = joblib.load(pca_path)

    def _extra_config(self) -> Dict:
        return {"pca_components": self.pca_components, "alpha": self.alpha}

    def _load_extra(self, path: Path, config: Dict) -> None:
        self.pca_components = config.get("pca_components", self.pca_components)
        self.alpha = config.get("alpha", self.alpha)


# ── Variant 3: MaskedPatchCore ────────────────────────────────────────────────


class MaskedPatchCore(PatchCoreBase):
    """Foreground-masked PatchCore.

    For each training image a binary foreground mask is computed on the fly
    (Otsu thresholding + morphological close) and only foreground patches are
    added to the memory bank.  During scoring, patches in background regions
    are zeroed out in the anomaly map to reduce false positives.

    Parameters
    ----------
    morph_kernel_size:
        Size of the structuring element used for morphological closing.
    """

    def __init__(
        self,
        image_size: int = 256,
        coreset_ratio: float = _DEFAULT_CORESET_RATIO,
        knn_k: int = _KNN_K,
        morph_kernel_size: int = 5,
    ) -> None:
        super().__init__(image_size=image_size, coreset_ratio=coreset_ratio, knn_k=knn_k)
        self.morph_kernel_size = morph_kernel_size

    # ------------------------------------------------------------------

    def _get_training_fg_masks(
        self,
        all_patches: np.ndarray,
        image_paths: List[str],
        feat_h: int,
        feat_w: int,
    ) -> List[np.ndarray]:
        """Compute per-image foreground masks for training images."""
        masks = []
        for p in image_paths:
            fg = self._compute_foreground_mask(p, feat_h, feat_w)
            masks.append(fg)
        n_kept = sum(m.sum() for m in masks)
        logger.info(
            "Foreground filtering: %d / %d patches kept (%.1f %%)",
            n_kept, len(all_patches), 100.0 * n_kept / max(len(all_patches), 1),
        )
        return masks

    def _compute_foreground_mask(self, image_path: str, feat_h: int, feat_w: int) -> np.ndarray:
        """Compute Otsu foreground mask downsampled to the feature grid.

        Returns
        -------
        np.ndarray of shape ``(feat_h * feat_w,)`` dtype bool.
        """
        try:
            import cv2
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return np.ones(feat_h * feat_w, dtype=bool)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_kernel_size, self.morph_kernel_size),
            )
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            # Downsample to feature grid
            fg_small = cv2.resize(fg_mask, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
            return (fg_small > 127).flatten()
        except Exception:
            return np.ones(feat_h * feat_w, dtype=bool)

    def _extra_config(self) -> Dict:
        return {"morph_kernel_size": self.morph_kernel_size}

    def _load_extra(self, path: Path, config: Dict) -> None:
        self.morph_kernel_size = config.get("morph_kernel_size", self.morph_kernel_size)


# ── Variant 4: FRMaskedPatchCore ─────────────────────────────────────────────


class FRMaskedPatchCore(FRPatchCore, MaskedPatchCore):
    """Combined Feature-Reconstruction + Foreground-Masking PatchCore.

    Uses Python's MRO to inherit:

    * Foreground masking from ``MaskedPatchCore``
      (``_get_training_fg_masks``, ``_compute_foreground_mask``).
    * PCA-based reconstruction scoring from ``FRPatchCore``
      (``_post_fit``, ``_patch_distances``).

    Parameters
    ----------
    See ``FRPatchCore`` and ``MaskedPatchCore`` for parameter descriptions.
    """

    def __init__(
        self,
        image_size: int = 256,
        coreset_ratio: float = _DEFAULT_CORESET_RATIO,
        knn_k: int = _KNN_K,
        pca_components: int = 256,
        alpha: float = 0.5,
        morph_kernel_size: int = 5,
    ) -> None:
        # Explicit super().__init__ to satisfy MRO and avoid ambiguity
        FRPatchCore.__init__(
            self,
            image_size=image_size,
            coreset_ratio=coreset_ratio,
            knn_k=knn_k,
            pca_components=pca_components,
            alpha=alpha,
        )
        self.morph_kernel_size = morph_kernel_size

    def _extra_config(self) -> Dict:
        config = FRPatchCore._extra_config(self)
        config.update(MaskedPatchCore._extra_config(self))
        return config

    def _load_extra(self, path: Path, config: Dict) -> None:
        FRPatchCore._load_extra(self, path, config)
        MaskedPatchCore._load_extra(self, path, config)


# ── Factory ────────────────────────────────────────────────────────────────────

_VARIANT_MAP = {
    "standard": StandardPatchCore,
    "fr": FRPatchCore,
    "mask": MaskedPatchCore,
    "fr_mask": FRMaskedPatchCore,
}


def create_model(variant: str, **kwargs) -> PatchCoreBase:
    """Instantiate a PatchCore variant by name.

    Parameters
    ----------
    variant:
        One of ``'standard'``, ``'fr'``, ``'mask'``, ``'fr_mask'``.
    **kwargs:
        Forwarded to the class constructor.

    Returns
    -------
    PatchCoreBase instance.
    """
    if variant not in _VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_VARIANT_MAP)}.")
    return _VARIANT_MAP[variant](**kwargs)
