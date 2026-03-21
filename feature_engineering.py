"""
feature_engineering.py
Data‑aware feature engineering engine.  Analyses dataset statistics and
conditionally applies transformations — nothing is done blindly.

Usage

    fe = FeatureEngineer(skew_threshold=1.0, cardinality_threshold=0.05)
    X_train = fe.fit_transform(X_train, y_train, problem_type="classification")
    X_test  = fe.transform(X_test)
    print(fe.log)          # list of human‑readable log strings
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler


# Main class

class FeatureEngineer:
    """Adaptive, data‑aware feature engineering pipeline."""

    def __init__(
        self,
        *,
        cardinality_threshold: float = 0.05,
        skew_threshold: float = 1.0,
        outlier_strategy: str = "cap",        
        encoding_strategy: str = "frequency", 
        interaction_features: int = 0,        
        random_state: int = 42,
        encoding_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.skew_threshold = skew_threshold
        self.outlier_strategy = outlier_strategy
        self.encoding_strategy = encoding_strategy
        self.interaction_features = interaction_features
        self.random_state = random_state
        self.encoding_map = encoding_map or {}

        # Fitted state (populated by fit_transform)
        self._fitted = False
        self._zero_var_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._datetime_features: Dict[str, List[str]] = {}
        self._high_card_cols: List[str] = []
        self._encoding_maps: Dict[str, Dict[Any, float]] = {}
        self._skew_cols: List[str] = []
        self._skew_shifts: Dict[str, float] = {}
        self._outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self._scaler_map: Dict[str, str] = {}  # col → "standard" | "robust"
        self._interaction_pairs: List[Tuple[str, str]] = []
        self.log: List[str] = []

    # Public API

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
    ) -> pd.DataFrame:
        self.log = []
        X = X.copy()

        X = self._fit_zero_variance(X)
        X = self._fit_datetime(X)
        X = self._fit_high_cardinality(X, y, problem_type)
        X = self._fit_skewness(X)
        X = self._fit_outliers(X)
        self._fit_adaptive_scaling(X)
        X = self._fit_interactions(X, y, problem_type)

        self._fitted = True
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")
        X = X.copy()

        X = self._apply_zero_variance(X)
        X = self._apply_datetime(X)
        X = self._apply_high_cardinality(X)
        X = self._apply_skewness(X)
        X = self._apply_outliers(X)
        X = self._apply_interactions(X)

        return X

    # 1. Zero‑variance removal

    def _fit_zero_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include="number")
        variances = numeric.var()
        self._zero_var_cols = variances[variances < 1e-6].index.tolist()
        if self._zero_var_cols:
            X = X.drop(columns=self._zero_var_cols)
            self._log(
                f"Dropped {len(self._zero_var_cols)} zero‑variance feature(s): "
                f"{self._zero_var_cols}"
            )
        return X

    def _apply_zero_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self._zero_var_cols if c in X.columns]
        return X.drop(columns=cols_to_drop) if cols_to_drop else X

    # 2. Datetime feature extraction

    def _detect_datetime_cols(self, X: pd.DataFrame) -> List[str]:
        dt_cols: List[str] = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt_cols.append(col)
            elif X[col].dtype == object:
                try:
                    sample = X[col].dropna().head(50)
                    if len(sample) == 0:
                        continue
                    pd.to_datetime(sample, infer_datetime_format=True)
                    dt_cols.append(col)
                except (ValueError, TypeError):
                    pass
        return dt_cols

    def _fit_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        self._datetime_cols = self._detect_datetime_cols(X)
        if not self._datetime_cols:
            return X

        for col in self._datetime_cols:
            X, new_cols = self._extract_dt_features(X, col)
            self._datetime_features[col] = new_cols
            self._log(
                f"Extracted datetime features from '{col}': {new_cols}"
            )
        return X

    def _apply_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._datetime_cols:
            if col in X.columns:
                X, _ = self._extract_dt_features(X, col)
        return X

    @staticmethod
    def _extract_dt_features(
        X: pd.DataFrame, col: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        dt = pd.to_datetime(X[col], errors="coerce", infer_datetime_format=True)
        new_cols: List[str] = []
        for attr, name in [
            ("year", f"{col}_year"), ("month", f"{col}_month"),
            ("day", f"{col}_day"), ("weekday", f"{col}_weekday"),
        ]:
            X[name] = getattr(dt.dt, attr)
            new_cols.append(name)
        # cyclical month
        X[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        X[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        new_cols += [f"{col}_month_sin", f"{col}_month_cos"]
        # cyclical weekday
        X[f"{col}_wday_sin"] = np.sin(2 * np.pi * dt.dt.weekday / 7)
        X[f"{col}_wday_cos"] = np.cos(2 * np.pi * dt.dt.weekday / 7)
        new_cols += [f"{col}_wday_sin", f"{col}_wday_cos"]
        X = X.drop(columns=[col])
        return X, new_cols

    # 3. High‑cardinality encoding

    def _fit_high_cardinality(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str,
    ) -> pd.DataFrame:
        cat_cols = X.select_dtypes(
            include=["object", "category", "bool"],
        ).columns.tolist()

        self._high_card_cols = []
        for col in cat_cols:
            if self.encoding_map:
                strategy = self.encoding_map.get(col, "onehot")
                if strategy in ["frequency", "target"]:
                    self._high_card_cols.append(col)
            else:
                unique_ratio = X[col].nunique() / len(X) if len(X) > 0 else 0
                if unique_ratio > self.cardinality_threshold:
                    self._high_card_cols.append(col)

        for col in self._high_card_cols:
            strategy = self.encoding_map.get(col, self.encoding_strategy) if self.encoding_map else self.encoding_strategy
            
            if strategy == "target":
                enc_map = self._target_encode_fit(X[col], y)
                strategy_name = "target"
            else:
                enc_map = X[col].value_counts(normalize=True).to_dict()
                strategy_name = "frequency"

            self._encoding_maps[col] = enc_map
            global_fallback = y.mean() if self.encoding_strategy == "target" else 0.0
            X[col] = X[col].map(enc_map).fillna(global_fallback).astype(float)
            self._log(
                f"Applied {strategy_name} encoding to '{col}' "
                f"(unique ratio={X[col].nunique()}/{len(X)})"
            )

        return X

    def _apply_high_cardinality(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._high_card_cols:
            if col not in X.columns:
                continue
            enc_map = self._encoding_maps[col]
            fallback = np.mean(list(enc_map.values()))
            X[col] = X[col].map(enc_map).fillna(fallback).astype(float)
        return X

    @staticmethod
    def _target_encode_fit(
        series: pd.Series, y: pd.Series, n_splits: int = 5,
    ) -> Dict[Any, float]:
        """Cross‑validated target encoding to prevent leakage."""
        from sklearn.model_selection import KFold

        encoding = pd.Series(np.nan, index=series.index, dtype=float)
        kf = KFold(n_splits=min(n_splits, len(series)), shuffle=True,
                    random_state=42)

        for train_idx, val_idx in kf.split(series):
            means = y.iloc[train_idx].groupby(series.iloc[train_idx]).mean()
            encoding.iloc[val_idx] = series.iloc[val_idx].map(means)

        # Fill remaining NaN with global mean
        encoding = encoding.fillna(y.mean())

        # Final map = per‑category average of CV‑encoded values
        final_map = encoding.groupby(series).mean().to_dict()
        return final_map

    # 4. Skewness correction

    def _fit_skewness(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include="number").columns.tolist()
        self._skew_cols = []
        self._skew_shifts = {}

        for col in numeric:
            skew_val = X[col].skew()
            if abs(skew_val) <= self.skew_threshold:
                continue

            col_min = X[col].min()
            if col_min < 0:
                # shift to make all values non‑negative, then log1p
                shift = abs(col_min) + 1.0
            else:
                shift = 0.0

            self._skew_cols.append(col)
            self._skew_shifts[col] = shift
            X[col] = np.log1p(X[col] + shift)
            self._log(
                f"Applied log1p transform to '{col}' (skew={skew_val:.2f}, "
                f"shift={shift:.1f})"
            )

        return X

    def _apply_skewness(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._skew_cols:
            if col not in X.columns:
                continue
            shift = self._skew_shifts[col]
            X[col] = np.log1p(X[col] + shift)
        return X

    # 5. Outlier handling (cap / winsorize)

    def _fit_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_strategy == "none":
            return X

        numeric = X.select_dtypes(include="number").columns.tolist()
        capped_count = 0

        for col in numeric:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()
            if n_outliers > 0:
                self._outlier_bounds[col] = (lower, upper)
                X[col] = X[col].clip(lower, upper)
                capped_count += 1

        if capped_count:
            self._log(
                f"Capped outliers (IQR method) in {capped_count} feature(s)"
            )

        return X

    def _apply_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self._outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        return X

    # 6. Adaptive scaling decision

    def _fit_adaptive_scaling(self, X: pd.DataFrame) -> None:
        """Decide per‑column scaler: StandardScaler vs RobustScaler."""
        numeric = X.select_dtypes(include="number").columns.tolist()
        n_standard = 0
        n_robust = 0

        for col in numeric:
            values = X[col].dropna()
            if len(values) < 20:
                self._scaler_map[col] = "standard"
                n_standard += 1
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_value = stats.normaltest(values)

            if p_value > 0.05:
                self._scaler_map[col] = "standard"
                n_standard += 1
            else:
                self._scaler_map[col] = "robust"
                n_robust += 1

        if n_robust > 0:
            self._log(
                f"Adaptive scaling: {n_standard} StandardScaler, "
                f"{n_robust} RobustScaler column(s)"
            )

    def get_scalers(self) -> Dict[str, str]:
        """Return the fitted scaler map for use in build_preprocessor."""
        return self._scaler_map.copy()

    # 7. Interaction features

    def _fit_interactions(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str,
    ) -> pd.DataFrame:
        k = self.interaction_features
        if k <= 0:
            return X

        numeric = X.select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2:
            return X

        # Rank by absolute correlation with target
        corrs = X[numeric].corrwith(y.astype(float)).abs().sort_values(
            ascending=False,
        )
        top_cols = corrs.head(k).index.tolist()

        self._interaction_pairs = []
        new_cols: List[str] = []
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols[i + 1:]:
                pair_name = f"{c1}__x__{c2}"
                X[pair_name] = X[c1] * X[c2]
                self._interaction_pairs.append((c1, c2))
                new_cols.append(pair_name)

        if new_cols:
            self._log(
                f"Created {len(new_cols)} interaction feature(s) from top‑{k} "
                f"numeric columns"
            )

        return X

    def _apply_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        for c1, c2 in self._interaction_pairs:
            pair_name = f"{c1}__x__{c2}"
            if c1 in X.columns and c2 in X.columns:
                X[pair_name] = X[c1] * X[c2]
        return X

    def _log(self, message: str) -> None:
        tag = "[FE]"
        self.log.append(message)
        print(f"{tag} {message}")
