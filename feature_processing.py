# feature_processing.py


from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


# Column detection

def detect_column_types(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


# Preprocessor builder

def build_preprocessor(
    X: pd.DataFrame,
    scaler_map: Optional[Dict[str, str]] = None,
    encoding_map: Optional[Dict[str, str]] = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols, categorical_cols = detect_column_types(X)

    transformers = []
    if numeric_cols:
        if scaler_map:
            std_cols = [c for c in numeric_cols
                        if scaler_map.get(c, "standard") == "standard"]
            rob_cols = [c for c in numeric_cols
                        if scaler_map.get(c) == "robust"]
            if std_cols:
                transformers.append(("num_std", StandardScaler(), std_cols))
            if rob_cols:
                transformers.append(("num_rob", RobustScaler(), rob_cols))
            # Any leftover numeric cols not in the map → StandardScaler
            leftover = [c for c in numeric_cols
                        if c not in std_cols and c not in rob_cols]
            if leftover:
                transformers.append(("num_other", StandardScaler(), leftover))
        else:
            transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        onehot_cols = [c for c in categorical_cols if not encoding_map or encoding_map.get(c, 'onehot') == 'onehot']
        hash_cols = [c for c in categorical_cols if encoding_map and encoding_map.get(c) == 'hash']
        
        if onehot_cols:
            transformers.append(
                (
                    "cat_ohe",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    onehot_cols,
                )
            )
            
        if hash_cols:
            from sklearn.feature_extraction import FeatureHasher
            from sklearn.preprocessing import FunctionTransformer
            import pandas as pd
            
            def hash_features(df):
                if isinstance(df, np.ndarray):
                    df = pd.DataFrame(df, columns=[str(i) for i in range(df.shape[1])])
                str_df = df.astype(str).fillna('missing')
                for c in str_df.columns:
                    str_df[c] = c + "=" + str_df[c]
                return FeatureHasher(n_features=2048, input_type='string').transform(str_df.values)

            transformers.append(
                (
                    "cat_hash",
                    FunctionTransformer(hash_features, validate=False),
                    hash_cols,
                )
            )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0, # Disable sparse to support pandas output natively
    )

    n_std = sum(1 for c in numeric_cols
                if not scaler_map or scaler_map.get(c, "standard") == "standard")
    n_rob = len(numeric_cols) - n_std if scaler_map else 0

    scale_info = f"{len(numeric_cols)} numeric"
    if n_rob > 0:
        scale_info += f" ({n_std} Standard, {n_rob} Robust)"
    print(
        f"[Features] {scale_info}, "
        f"{len(categorical_cols)} categorical column(s)."
    )

    # Resolve LightGBM warnings by keeping dataframe format and column names 
    preprocessor.set_output(transform="pandas")

    return preprocessor, numeric_cols, categorical_cols


# Feature selection


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    method: str = "mutual_info",
    k: int = 10,
) -> Tuple[np.ndarray, SelectKBest]:

    k = min(k, X.shape[1])

    if method == "mutual_info":
        score_func = (
            mutual_info_classif
            if problem_type == "classification"
            else mutual_info_regression
        )
    else:
        raise ValueError(
            f"Unsupported feature selection method: '{method}'. "
            "Use 'mutual_info'."
        )

    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    print(
        f"[Features] Selected top {k} features via {method} "
        f"(from {X.shape[1]})."
    )

    return X_selected, selector
