"""
dataset_embedding.py
====================
Phase 4.1 — Rich Dataset Embedding Vectors.

Computes fixed-length, FAISS-compatible ``float32`` vectors that
summarise a dataset's statistical fingerprint.  Designed for
meta-learning, dataset similarity search, and pipeline warm-starting.

Public API
----------
- ``compute_dataset_embedding(X, y=None) -> np.ndarray``
- ``build_embedding_matrix(datasets) -> np.ndarray``
- ``save_embeddings(results, path)``

Embedding dimensions (9):
    0  n_samples        — log-scaled row count
    1  n_features       — log-scaled column count
    2  missing_ratio    — fraction of NaN cells
    3  mean_skewness    — avg skewness across numeric columns
    4  mean_kurtosis    — avg kurtosis across numeric columns
    5  entropy          — target entropy (clf) or avg feature entropy
    6  corr_density     — % of pairs with |corr| > 0.5
    7  interaction_score— normalised RF importance variance
    8  task_type         — 0=binary clf, 1=regression, 2=multiclass
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 9
_RF_ESTIMATORS = 50
_RF_MAX_DEPTH = 5
_CORR_THRESHOLD = 0.5
_RANDOM_STATE = 42
_MAX_ROWS_FOR_RF = 10_000          # cap rows fed to the RF probe
_MAX_FEATURES_FOR_CORR = 200       # cap features for correlation matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and label-encode categoricals.

    Returns a fully numeric DataFrame with no NaNs.
    """
    X = X.copy()

    # --- Numeric imputation (median) ---
    num_cols = X.select_dtypes(include="number").columns
    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # --- Categorical imputation (mode) + label encoding ---
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        mode_val = X[col].mode()
        fill = mode_val.iloc[0] if len(mode_val) else "missing"
        X[col] = X[col].fillna(fill)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Convert any remaining non-numeric columns
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            except Exception:
                X[col] = 0

    return X


def _infer_task_type(y: Optional[pd.Series]) -> int:
    """Return 0 (binary classification), 1 (regression), or 2 (multiclass).

    If *y* is ``None`` the task type defaults to regression (1).
    """
    if y is None:
        return 1  # no target → assume regression
    nunique = int(pd.Series(y).nunique())
    if nunique <= 2:
        return 0  # binary classification
    if nunique <= 20 or pd.api.types.is_object_dtype(y):
        return 2  # multiclass
    return 1      # regression


def _safe_entropy(values: np.ndarray) -> float:
    """Shannon entropy in nats, safe for empty / constant arrays."""
    values = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(stats.entropy(counts))


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] and replace any residual NaN with 0."""
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = vec.min(), vec.max()
    if vmax - vmin > 1e-12:
        vec = (vec - vmin) / (vmax - vmin)
    return vec


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def compute_dataset_embedding(
    X: pd.DataFrame,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Compute a fixed-length ``float32`` embedding for a single dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw — missing values and categoricals OK).
    y : array-like, optional
        Target vector.  Used to determine task type, entropy, and
        interaction score.  If ``None``, a target-free embedding is
        produced.

    Returns
    -------
    np.ndarray
        1-D ``float32`` array of length :data:`EMBEDDING_DIM` (9).
        Guaranteed NaN-free, suitable for FAISS indexing.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
        y = np.asarray(y)

    n_samples, n_features = X.shape

    # --- A. Basic statistics ---
    missing_ratio = float(X.isnull().sum().sum()) / max(X.size, 1)

    # --- Preprocess for downstream features ---
    X_clean = _preprocess(X)
    num_cols = X_clean.select_dtypes(include="number").columns

    # --- B. Distribution features ---
    if len(num_cols) > 0:
        skew_vals = X_clean[num_cols].apply(lambda c: stats.skew(c, nan_policy="omit"))
        kurt_vals = X_clean[num_cols].apply(lambda c: stats.kurtosis(c, nan_policy="omit"))
        mean_skewness = float(np.nan_to_num(np.nanmean(skew_vals), nan=0.0))
        mean_kurtosis = float(np.nan_to_num(np.nanmean(kurt_vals), nan=0.0))
    else:
        mean_skewness = 0.0
        mean_kurtosis = 0.0

    # --- C. Information-theoretic feature ---
    task_type = _infer_task_type(y)
    if y is not None and task_type != 1:
        # Classification target entropy
        entropy_val = _safe_entropy(np.asarray(y))
    else:
        # Average feature entropy (discretise numeric cols into 10 bins)
        entropies = []
        for col in num_cols[:50]:  # cap for speed
            try:
                binned = pd.cut(X_clean[col], bins=10, labels=False)
                entropies.append(_safe_entropy(binned.values))
            except Exception:
                entropies.append(0.0)
        entropy_val = float(np.mean(entropies)) if entropies else 0.0

    # --- D. Correlation structure ---
    corr_density = 0.0
    sel_cols = num_cols[:_MAX_FEATURES_FOR_CORR]
    if len(sel_cols) >= 2:
        corr_matrix = X_clean[sel_cols].corr().to_numpy(copy=True)
        np.fill_diagonal(corr_matrix, 0)
        n_pairs = len(sel_cols) * (len(sel_cols) - 1) / 2
        if n_pairs > 0:
            upper = np.triu(np.abs(corr_matrix), k=1)
            corr_density = float(np.sum(upper > _CORR_THRESHOLD) / n_pairs)

    # --- E. Feature interaction score ---
    interaction_score = 0.0
    if y is not None and len(num_cols) >= 2:
        # Subsample for speed
        n_rf = min(n_samples, _MAX_ROWS_FOR_RF)
        rng = np.random.RandomState(_RANDOM_STATE)
        idx = rng.choice(n_samples, size=n_rf, replace=False)
        X_rf = X_clean[num_cols].iloc[idx].values
        y_rf = np.asarray(y)[idx] if isinstance(y, np.ndarray) else pd.Series(y).iloc[idx].values

        try:
            if task_type in (0, 2):
                rf = RandomForestClassifier(
                    n_estimators=_RF_ESTIMATORS,
                    max_depth=_RF_MAX_DEPTH,
                    random_state=_RANDOM_STATE,
                    n_jobs=-1,
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=_RF_ESTIMATORS,
                    max_depth=_RF_MAX_DEPTH,
                    random_state=_RANDOM_STATE,
                    n_jobs=-1,
                )

            # Encode y for classifier if needed
            y_fit = y_rf
            if task_type in (0, 2) and not np.issubdtype(y_rf.dtype, np.number):
                y_fit = LabelEncoder().fit_transform(y_rf.astype(str))

            rf.fit(X_rf, y_fit)
            importances = rf.feature_importances_
            # Normalised variance of importances → proxy for interaction
            imp_std = float(np.std(importances))
            imp_mean = float(np.mean(importances))
            interaction_score = imp_std / imp_mean if imp_mean > 1e-12 else 0.0
        except Exception as exc:
            logger.warning("RF interaction probe failed: %s", exc)
            interaction_score = 0.0

    # --- F. Task type encoding (already computed above) ---

    # --- Assemble raw embedding ---
    raw = np.array(
        [
            np.log1p(n_samples),       # 0
            np.log1p(n_features),      # 1
            missing_ratio,             # 2
            mean_skewness,             # 3
            mean_kurtosis,             # 4
            entropy_val,               # 5
            corr_density,              # 6
            interaction_score,         # 7
            float(task_type),          # 8
        ],
        dtype=np.float64,
    )

    # --- Normalise & cast ---
    embedding = _normalize_embedding(raw).astype(np.float32)

    logger.info(
        "Embedding computed: shape=%s  dtype=%s  range=[%.4f, %.4f]",
        embedding.shape, embedding.dtype, embedding.min(), embedding.max(),
    )
    return embedding


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_embedding_matrix(
    datasets: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """Compute embeddings for multiple datasets at once.

    Parameters
    ----------
    datasets : list of dict
        Each dict must contain ``"X"`` (pd.DataFrame) and optionally
        ``"y"`` (array-like).

    Returns
    -------
    np.ndarray
        2-D ``float32`` array of shape ``(len(datasets), EMBEDDING_DIM)``.
    """
    embeddings = []
    for i, ds in enumerate(datasets):
        X = ds["X"]
        y = ds.get("y", None)
        emb = compute_dataset_embedding(X, y)
        embeddings.append(emb)
        logger.info("Dataset %d/%d embedded.", i + 1, len(datasets))

    matrix = np.vstack(embeddings).astype(np.float32)
    return matrix


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(
    results: List[Dict[str, Any]],
    path: str,
) -> str:
    """Save embedding results with metadata to a JSON file.

    Parameters
    ----------
    results : list of dict
        Each dict should follow the schema::

            {
                "dataset_id": str | int,
                "embedding": list[float],
                "task_type": int,
                "n_samples": int,
                "n_features": int,
            }

    path : str
        Output JSON file path.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure embeddings are plain Python lists (JSON-serialisable)
    serialisable = []
    for r in results:
        entry = dict(r)
        if isinstance(entry.get("embedding"), np.ndarray):
            entry["embedding"] = entry["embedding"].tolist()
        serialisable.append(entry)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    abs_path = os.path.abspath(path)
    logger.info("Embeddings saved to %s (%d entries).", abs_path, len(serialisable))
    return abs_path
