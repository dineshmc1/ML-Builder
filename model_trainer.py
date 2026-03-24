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
        max_iter=1000, random_state=42, verbose=1
    ),
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42, verbose=1
    ),
    "gb": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42, verbose=1
    ),
}

_REGRESSION_MODELS: Dict[str, Any] = {
    "linear": lambda: LinearRegression(n_jobs=-1),
    "rf": lambda: RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42, verbose=1
    ),
    "gb": lambda: GradientBoostingRegressor(
        n_estimators=100, random_state=42, verbose=1
    ),
}

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _CLASSIFICATION_MODELS["lightgbm"] = lambda: LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=20, random_state=42, verbose=1)
    _REGRESSION_MODELS["lightgbm"] = lambda: LGBMRegressor(n_estimators=100, max_depth=5, num_leaves=20, random_state=42, verbose=1)
except ImportError:
    pass

try:
    from xgboost import XGBClassifier, XGBRegressor
    _CLASSIFICATION_MODELS["xgboost"] = lambda: XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=1)
    _REGRESSION_MODELS["xgboost"] = lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=1)
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


# Helper for training with custom CV and early stopping
def _train_and_evaluate(
    name: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int,
    start: float,
    max_time_seconds: Optional[float],
    refit_full: bool = False,
) -> Tuple[float, Optional[Pipeline]]:
    from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
    from sklearn.base import clone
    from sklearn.metrics import f1_score, mean_squared_error
    from sklearn.pipeline import Pipeline
    import scipy.sparse
    
    cv_scores = []
    is_classification = problem_type == "classification"
    cv_val = min(cv, len(X))
    
    if cv_val <= 1:
        # Single validation split for large datasets (speed and early stopping)
        try:
            splits = [train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=y if is_classification else None)]
        except ValueError:
            splits = [train_test_split(np.arange(len(X)), test_size=0.15, random_state=42)]
    else:
        kf = StratifiedKFold(n_splits=cv_val, shuffle=True, random_state=42) if is_classification else KFold(n_splits=cv_val, shuffle=True, random_state=42)
        try:
            splits = list(kf.split(X, y))
        except ValueError:
            splits = list(KFold(n_splits=cv_val, shuffle=True, random_state=42).split(X, y))
            
    best_pipe = None
    
    print(f"  [Trainer] Commencing training loop for '{name}'...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"    [Timeout] Stopping {name} early due to time budget.")
            break
            
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_va = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        
        prep = clone(preprocessor)
        X_tr_prep = prep.fit_transform(X_tr, y_tr)
        
        if scipy.sparse.issparse(X_tr_prep) and name == "gb":
            print(f"    Skipping '{name}' as transformations yielded sparse matrix.")
            return -float('inf'), None
            
        X_va_prep = prep.transform(X_va)
        
        fit_kwargs = {}
        if name in ["lightgbm", "xgboost"]:
            fit_kwargs["eval_set"] = [(X_va_prep, y_va)]
            if name == "lightgbm":
                try:
                    from lightgbm import early_stopping, log_evaluation
                    fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=10, verbose=True), log_evaluation(period=10)]
                except ImportError:
                    pass
            elif name == "xgboost":
                fit_kwargs["verbose"] = 10
                
        est = clone(estimator)
        print(f"    Fold {fold+1}/{len(splits)} - fitting model...")
        est.fit(X_tr_prep, y_tr, **fit_kwargs)
        
        y_pred = est.predict(X_va_prep)
        if is_classification:
            sc = f1_score(y_va, y_pred, average="weighted", zero_division=0)
        else:
            sc = -np.sqrt(mean_squared_error(y_va, y_pred))
        cv_scores.append(sc)
        
        if cv_val <= 1:
            best_pipe = Pipeline([("preprocessor", prep), ("model", est)])
            
    if not cv_scores:
        return -float('inf'), None
        
    mean_score = float(np.mean(cv_scores))
    
    if cv_val > 1 and refit_full:
        print(f"  [Trainer] Refitting {name} on ALL data...")
        prep = clone(preprocessor)
        X_prep = prep.fit_transform(X, y)
        est = clone(estimator)
        est.fit(X_prep, y)
        best_pipe = Pipeline([("preprocessor", prep), ("model", est)])
        
    return mean_score, best_pipe


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

    scores: Dict[str, float] = {}
    start = time.time()

    print(f"\n[Baseline] Screening on {len(X_sub)} samples ({sample_frac:.0%} subsample)…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[Baseline] Time budget exhausted – skipping '{name}'.")
            break

        mean_score, _ = _train_and_evaluate(
            name, estimator, preprocessor, X_sub, y_sub, problem_type, cv, start, max_time_seconds, refit_full=False
        )
        if mean_score > -float('inf'):
            scores[name] = mean_score
            print(f"  {name:>12s}  baseline score = {mean_score:.4f}")

    if not scores:
        return models, scores

    best_score = max(scores.values())
    worst_score = min(scores.values())
    score_range = best_score - worst_score

    if score_range == 0:
        return {n: models[n] for n in scores}, scores

    threshold = best_score - 0.70 * score_range

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
    # Train each model on the full training set with cross‑validation or single-split.
    trained: Dict[str, Pipeline] = {}
    scores: Dict[str, float] = {}
    start = time.time()

    print(f"\n[FullTrain] Training on {len(X)} samples…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[FullTrain] Time budget exhausted – skipping '{name}'.")
            break

        mean_score, pipe = _train_and_evaluate(
            name, estimator, preprocessor, X, y, problem_type, cv, start, max_time_seconds, refit_full=True
        )
        if pipe is not None and mean_score > -float('inf'):
            scores[name] = mean_score
            trained[name] = pipe
            print(f"  {name:>12s}  training score = {mean_score:.4f}")

    return trained, scores
