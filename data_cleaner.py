"""
data_cleaner.py
Detects and handles missing values and duplicate rows.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import pandas as pd


def clean(
    df: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Clean a DataFrame and optionally keep a target Series in sync.

    When *y* is provided the function returns ``(df, y)``; otherwise
    it returns just ``df`` (backward compatible).
    """
    df = df.copy()
    if y is not None:
        y = y.copy()

    # duplicates 
    dupe_mask = df.duplicated()
    n_dupes = dupe_mask.sum()
    if n_dupes > 0:
        keep_mask = ~dupe_mask
        df = df.loc[keep_mask].reset_index(drop=True)
        if y is not None:
            y = y.loc[keep_mask].reset_index(drop=True)
        if verbose:
            print(f"[Cleaner] Removed {n_dupes} duplicate row(s).")

    # missing values 
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]

    if cols_with_missing.empty:
        if verbose:
            print("[Cleaner] No missing values detected.")
        return (df, y) if y is not None else df

    for col in cols_with_missing.index:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            strategy = "median"
        else:
            fill_value = df[col].mode().iloc[0]
            strategy = "most_frequent"

        df[col] = df[col].fillna(fill_value)
        if verbose:
            print(
                f"[Cleaner] Filled {cols_with_missing[col]} missing value(s) "
                f"in '{col}' with {strategy} ({fill_value})."
            )

    return (df, y) if y is not None else df

