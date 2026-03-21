"""
model_selector.py
Evaluates trained models, selects the best one, and provides optional
hyperparameter tuning for the top candidates.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Evaluation

def evaluate_models(
    trained: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
) -> pd.DataFrame:
    
    # Compute metrics for every trained model on the held‑out test set.
    rows: list[dict] = []

    for name, pipe in trained.items():
        y_pred = pipe.predict(X_test)
        row: dict = {"model": name}

        if problem_type == "classification":
            row["accuracy"] = accuracy_score(y_test, y_pred)
            row["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            row["recall"] = recall_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            row["f1"] = f1_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            # ROC‑AUC (only for binary or when predict_proba is available)
            try:
                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        row["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        row["roc_auc"] = roc_auc_score(
                            y_test, y_prob, multi_class="ovr", average="weighted",
                        )
                else:
                    row["roc_auc"] = None
            except Exception:
                row["roc_auc"] = None
        else:
            row["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            row["mae"] = mean_absolute_error(y_test, y_pred)
            row["r2"] = r2_score(y_test, y_pred)

        rows.append(row)

    results = pd.DataFrame(rows)
    return results


# Best model selection

def select_best(
    results: pd.DataFrame,
    problem_type: str,
) -> str:
    # Pick the best model name from the results table.
    if problem_type == "classification":
        best_idx = results["f1"].idxmax()
    else:
        best_idx = results["rmse"].idxmin()

    return results.loc[best_idx, "model"]


# Hyperparameter tuning

# Default param grids (kept small for speed)
_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "rf": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    },
    "gb": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    "linear": {},
}


def tune_top_models(
    trained: Dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str,
    results: pd.DataFrame,
    top_n: int = 2,
    method: str = "randomized",
    n_iter: int = 20,
    cv: int = 5,
) -> Dict[str, Pipeline]:
    
    # Run hyperparameter search on the *top_n* best models.
    if problem_type == "classification":
        sorted_results = results.sort_values("f1", ascending=False)
        scoring = "f1_weighted"
    else:
        sorted_results = results.sort_values("rmse", ascending=True)
        scoring = "neg_root_mean_squared_error"

    top_names = sorted_results["model"].head(top_n).tolist()
    tuned: Dict[str, Pipeline] = {}

    print(f"\n[Tuner] Tuning top {top_n} model(s): {top_names}")

    for name in top_names:
        pipe = trained[name]
        param_grid = _PARAM_GRIDS.get(name, {})

        if not param_grid:
            print(f"  {name}: no param grid defined – skipping tuning.")
            tuned[name] = pipe
            continue

        if method == "grid":
            searcher = GridSearchCV(
                pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1,
                refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1,
                n_iter=min(n_iter, _grid_size(param_grid)),
                refit=True, random_state=42,
            )

        searcher.fit(X_train, y_train)
        tuned[name] = searcher.best_estimator_
        print(
            f"  {name}: best params = {searcher.best_params_}, "
            f"best score = {searcher.best_score_:.4f}"
        )

    return tuned


def _grid_size(param_grid: dict) -> int:
    """Calculate total number of combinations in a param grid."""
    size = 1
    for vals in param_grid.values():
        size *= len(vals)
    return size

# Persistence helpers

def save_model(model: Any, path: str) -> None:
    """Persist a model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Selector] Best model saved → {path}")


def save_metrics(results: pd.DataFrame, path: str) -> None:
    """Save the metrics DataFrame as CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    results.to_csv(path, index=False)
    print(f"[Selector] Metrics saved → {path}")
