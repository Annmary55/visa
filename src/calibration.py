"""Anomaly score calibration – map raw detector outputs to P(anomaly) ∈ [0, 1].

Two calibration methods are supported:

* **isotonic** – ``sklearn.isotonic.IsotonicRegression`` (monotone, non-parametric).
* **logistic** – ``sklearn.linear_model.LogisticRegression`` (parametric sigmoid).

Typical workflow::

    calibrator = ScoreCalibrator()
    calibrator.fit(val_scores, val_labels, method='isotonic')
    proba = calibrator.predict_proba(test_scores)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Core calibrator class ──────────────────────────────────────────────────────


class ScoreCalibrator:
    """Wrapper around a sklearn calibration model.

    Parameters
    ----------
    method:
        ``'isotonic'`` (default) or ``'logistic'``.  Can also be set at
        ``fit()`` time.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self._model = None  # fitted sklearn estimator

    # ------------------------------------------------------------------

    def fit(
        self,
        scores: np.ndarray | List[float],
        labels: np.ndarray | List[int],
        method: Optional[str] = None,
    ) -> "ScoreCalibrator":
        """Fit the calibration model.

        Parameters
        ----------
        scores:
            Raw anomaly scores (1-D array, higher = more anomalous).
        labels:
            Ground-truth binary labels: 0 = normal, 1 = anomaly.
        method:
            Override the instance ``method`` for this call only.

        Returns
        -------
        self (for chaining).
        """
        if method is not None:
            self.method = method

        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int32)

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            # IsotonicRegression maps sorted scores to probabilities.
            # We use out_of_bounds='clip' to handle scores outside training range.
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(scores.ravel(), labels)
        elif self.method == "logistic":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(scores, labels)
        else:
            raise ValueError(
                f"Unknown calibration method '{self.method}'. "
                "Choose 'isotonic' or 'logistic'."
            )

        self._model = model
        logger.info("ScoreCalibrator fitted (method=%s) on %d samples.", self.method, len(labels))
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, scores: np.ndarray | List[float]) -> np.ndarray:
        """Return calibrated P(anomaly) for each score.

        Parameters
        ----------
        scores:
            1-D array of raw anomaly scores.

        Returns
        -------
        np.ndarray of shape ``(N,)`` with values in ``[0, 1]``.
        """
        if self._model is None:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        scores = np.asarray(scores, dtype=np.float64)

        if self.method == "isotonic":
            proba = self._model.predict(scores.ravel())
        else:
            proba = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]

        return np.clip(proba, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted calibrator to disk using joblib.

        Parameters
        ----------
        path:
            File path (e.g. ``calibrators/candle.pkl``).  Parent directories
            are created automatically.
        """
        import joblib  # lazy import

        if self._model is None:
            raise RuntimeError("Nothing to save – calibrator has not been fitted.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "model": self._model}, path)
        logger.info("Saved calibrator to %s", path)

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCalibrator":
        """Load a previously saved calibrator from disk.

        Parameters
        ----------
        path:
            Path to the ``.pkl`` file written by ``save()``.

        Returns
        -------
        Fitted ``ScoreCalibrator`` instance.
        """
        import joblib  # lazy import

        path = Path(path).resolve()
        # Validate the file exists and has the expected extension before loading
        if not path.is_file():
            raise FileNotFoundError(f"Calibrator file not found: {path}")
        if path.suffix.lower() != ".pkl":
            raise ValueError(
                f"Expected a .pkl calibrator file, got: {path.suffix!r}"
            )
        payload = joblib.load(path)  # noqa: S301 – loading trusted local artifacts
        instance = cls(method=payload["method"])
        instance._model = payload["model"]
        logger.info("Loaded calibrator from %s (method=%s)", path, instance.method)
        return instance


# ── Batch helpers ──────────────────────────────────────────────────────────────


def fit_and_save_calibrators(
    category_scores: Dict[str, List[float]],
    category_labels: Dict[str, List[int]],
    output_dir: str | Path,
    variant: str,
    method: str = "isotonic",
) -> Dict[str, ScoreCalibrator]:
    """Fit one calibrator per category and persist all of them.

    Expected on-disk layout::

        <output_dir>/<variant>/calibrators/<category>.pkl

    Parameters
    ----------
    category_scores:
        Mapping ``{category: [score, ...]}`` for the *validation* split.
    category_labels:
        Mapping ``{category: [label, ...]}`` for the *validation* split.
    output_dir:
        Root export directory (e.g. ``/kaggle/working/exports``).
    variant:
        Model variant name (e.g. ``'standard'``).  Used to construct the
        output sub-directory.
    method:
        Calibration method passed to ``ScoreCalibrator.fit()``.

    Returns
    -------
    dict mapping category name → fitted ``ScoreCalibrator``.
    """
    output_dir = Path(output_dir)
    calibrators: Dict[str, ScoreCalibrator] = {}

    for category in category_scores:
        scores = category_scores[category]
        labels = category_labels[category]

        if len(scores) == 0:
            logger.warning("No scores for category '%s'; skipping calibration.", category)
            continue

        cal = ScoreCalibrator(method=method)
        cal.fit(scores, labels)

        save_path = output_dir / variant / "calibrators" / f"{category}.pkl"
        cal.save(save_path)
        calibrators[category] = cal

    logger.info(
        "Calibrators for variant '%s' saved to %s", variant, output_dir / variant / "calibrators"
    )
    return calibrators


def load_calibrators(
    model_dir: str | Path,
    categories: List[str],
) -> Dict[str, ScoreCalibrator]:
    """Load calibrators from ``<model_dir>/calibrators/<category>.pkl``.

    Parameters
    ----------
    model_dir:
        Directory that contains the ``calibrators/`` sub-directory, typically
        ``<export_root>/<variant>/``.
    categories:
        List of category names to load.

    Returns
    -------
    dict mapping category name → loaded ``ScoreCalibrator``.  Missing files
    are skipped with a warning.
    """
    model_dir = Path(model_dir)
    calibrators: Dict[str, ScoreCalibrator] = {}

    for category in categories:
        pkl_path = model_dir / "calibrators" / f"{category}.pkl"
        if not pkl_path.exists():
            logger.warning("Calibrator file not found: %s", pkl_path)
            continue
        calibrators[category] = ScoreCalibrator.load(pkl_path)

    return calibrators
