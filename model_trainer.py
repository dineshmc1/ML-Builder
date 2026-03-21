"""
model_trainer.py
Provides model catalogues, baseline screening on a data subsample, and
full training with cross‑validation.  Supports parallel execution and an
optional wall‑clock time budget.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Model catalogue

_CLASSIFICATION_MODELS: Dict[str, Any] = {
    "logistic": lambda: LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    ),
    "gb": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

_REGRESSION_MODELS: Dict[str, Any] = {
    "linear": lambda: LinearRegression(n_jobs=-1),
    "rf": lambda: RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42
    ),
    "gb": lambda: GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
}

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _CLASSIFICATION_MODELS["lightgbm"] = lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    _REGRESSION_MODELS["lightgbm"] = lambda: LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
except ImportError:
    pass

try:
    from xgboost import XGBClassifier, XGBRegressor
    _CLASSIFICATION_MODELS["xgboost"] = lambda: XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    _REGRESSION_MODELS["xgboost"] = lambda: XGBRegressor(n_estimators=100, random_state=42)
except ImportError:
    pass


def get_models(
    problem_type: str,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Return a dict of ``{name: estimator_instance}`` for the requested problem type.
    catalogue = (
        _CLASSIFICATION_MODELS
        if problem_type == "classification"
        else _REGRESSION_MODELS
    )

    if model_names is None or "all" in model_names:
        selected = {k: fn() for k, fn in catalogue.items()}
    else:
        selected = {}
        for name in model_names:
            if name not in catalogue:
                print(
                    f"[Trainer] Warning: model '{name}' not found in "
                    f"{problem_type} catalogue – skipped."
                )
                continue
            selected[name] = catalogue[name]()

    print(f"[Trainer] Selected models: {list(selected.keys())}")
    return selected


# Baseline screening

def baseline_screen(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    sample_frac: float = 0.3,
    cv: int = 5,
    random_state: int = 42,
    max_time_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    # Quick evaluation of all candidate models on a data subsample.
    
    rng = np.random.RandomState(random_state)
    n_sample = max(int(len(X) * sample_frac), 50)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

    scoring = "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"

    scores: Dict[str, float] = {}
    start = time.time()

    print(f"\n[Baseline] Screening on {len(X_sub)} samples ({sample_frac:.0%} subsample)…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[Baseline] Time budget exhausted – skipping '{name}'.")
            break

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        # Check if input is sparse and model supports sparse
        import scipy.sparse
        try:
            X_probe = preprocessor.fit_transform(X_sub.head(5) if hasattr(X_sub, "head") else X_sub[:5])
            if scipy.sparse.issparse(X_probe) and name == "gb":
                print(f"[Trainer] Skipping '{name}' as it requires a dense matrix, but transformations yielded sparse.")
                continue
        except Exception:
            pass

        cv_scores = cross_val_score(
            pipe, X_sub, y_sub, cv=min(cv, len(X_sub)), scoring=scoring,
            n_jobs=-1,
        )
        mean_score = cv_scores.mean()
        scores[name] = mean_score
        print(f"  {name:>12s}  baseline score = {mean_score:.4f}")

    if not scores:
        return models, scores

    # Use range‑based threshold that works for both positive scores
    best_score = max(scores.values())
    worst_score = min(scores.values())
    score_range = best_score - worst_score

    if score_range == 0:
        # All models scored identically – keep them all
        return {n: models[n] for n in scores}, scores

    threshold = best_score - 0.70 * score_range  # keep top 70 % of range

    promising = {
        name: models[name]
        for name, sc in scores.items()
        if sc >= threshold
    }
    dropped = set(scores) - set(promising)
    if dropped:
        print(f"[Baseline] Dropped underperforming model(s): {dropped}")

    return promising, scores


# Full training

def full_train(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int = 5,
    max_time_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    # Train each model on the full training set with cross‑validation.
    scoring = "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"
    trained: Dict[str, Pipeline] = {}
    scores: Dict[str, float] = {}
    start = time.time()

    print(f"\n[FullTrain] Training on {len(X)} samples…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[FullTrain] Time budget exhausted – skipping '{name}'.")
            break

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        # Check if input is sparse and model supports sparse
        import scipy.sparse
        try:
            X_probe = preprocessor.fit_transform(X.head(5) if hasattr(X, "head") else X[:5])
            if scipy.sparse.issparse(X_probe) and name == "gb":
                print(f"[Trainer] Skipping '{name}' as it requires a dense matrix, but transformations yielded sparse.")
                continue
        except Exception:
            pass

        cv_scores = cross_val_score(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        )
        mean_score = cv_scores.mean()
        scores[name] = mean_score

        # Refit on all training data
        pipe.fit(X, y)
        trained[name] = pipe
        print(f"  {name:>12s}  CV score = {mean_score:.4f}")

    return trained, scores
