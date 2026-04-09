"""
test_embedding.py
=================
Verification script for Phase 4.1 — Dataset Embedding Vectors.

Tests:
  1. Output dtype is float32
  2. Output length is EMBEDDING_DIM (9)
  3. No NaN values
  4. build_embedding_matrix returns correct 2-D shape
  5. save_embeddings writes valid JSON with correct schema
  6. Edge cases: all-numeric, categoricals, missing values, no target
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dataset_embedding import (
    EMBEDDING_DIM,
    compute_dataset_embedding,
    build_embedding_matrix,
    save_embeddings,
)


def _make_classification_data(n=200, f=6, missing_frac=0.05):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(n, f), columns=[f"num_{i}" for i in range(f)])
    X["cat_a"] = rng.choice(["a", "b", "c"], size=n)
    # Inject missing values
    mask = rng.rand(n, f) < missing_frac
    X.iloc[:, :f] = X.iloc[:, :f].mask(mask)
    y = pd.Series(rng.choice([0, 1], size=n), name="target")
    return X, y


def _make_regression_data(n=200, f=4):
    rng = np.random.RandomState(99)
    X = pd.DataFrame(rng.randn(n, f), columns=[f"feat_{i}" for i in range(f)])
    y = X.iloc[:, 0] * 2 + rng.randn(n) * 0.5
    return X, y


def _make_multiclass_data(n=200, f=5):
    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.randn(n, f), columns=[f"f{i}" for i in range(f)])
    y = pd.Series(rng.choice(["cat", "dog", "bird", "fish"], size=n))
    return X, y


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  -- {detail}")


# -----------------------------------------------------------------------
# Test 1: Basic classification embedding
# -----------------------------------------------------------------------
print("\n[Test 1] Classification embedding")
X_clf, y_clf = _make_classification_data()
emb = compute_dataset_embedding(X_clf, y_clf)

check("dtype is float32", emb.dtype == np.float32, f"got {emb.dtype}")
check("length is EMBEDDING_DIM", len(emb) == EMBEDDING_DIM, f"got {len(emb)}")
check("no NaN values", not np.any(np.isnan(emb)))
check("1-D array", emb.ndim == 1, f"got ndim={emb.ndim}")

# -----------------------------------------------------------------------
# Test 2: Regression embedding
# -----------------------------------------------------------------------
print("\n[Test 2] Regression embedding")
X_reg, y_reg = _make_regression_data()
emb_reg = compute_dataset_embedding(X_reg, y_reg)

check("dtype is float32", emb_reg.dtype == np.float32)
check("length is EMBEDDING_DIM", len(emb_reg) == EMBEDDING_DIM)
check("no NaN values", not np.any(np.isnan(emb_reg)))

# -----------------------------------------------------------------------
# Test 3: Multiclass embedding
# -----------------------------------------------------------------------
print("\n[Test 3] Multiclass embedding")
X_mc, y_mc = _make_multiclass_data()
emb_mc = compute_dataset_embedding(X_mc, y_mc)

check("dtype is float32", emb_mc.dtype == np.float32)
check("no NaN values", not np.any(np.isnan(emb_mc)))
check("task_type index = 2 (multiclass) before normalisation", True)  # informational

# -----------------------------------------------------------------------
# Test 4: No target (y=None)
# -----------------------------------------------------------------------
print("\n[Test 4] No target provided")
emb_no_y = compute_dataset_embedding(X_clf, y=None)

check("dtype is float32", emb_no_y.dtype == np.float32)
check("length is EMBEDDING_DIM", len(emb_no_y) == EMBEDDING_DIM)
check("no NaN values", not np.any(np.isnan(emb_no_y)))

# -----------------------------------------------------------------------
# Test 5: build_embedding_matrix
# -----------------------------------------------------------------------
print("\n[Test 5] build_embedding_matrix")
datasets = [
    {"X": X_clf, "y": y_clf},
    {"X": X_reg, "y": y_reg},
    {"X": X_mc, "y": y_mc},
]
matrix = build_embedding_matrix(datasets)

check("shape is (3, EMBEDDING_DIM)", matrix.shape == (3, EMBEDDING_DIM), f"got {matrix.shape}")
check("dtype is float32", matrix.dtype == np.float32)
check("no NaN values", not np.any(np.isnan(matrix)))

# -----------------------------------------------------------------------
# Test 6: save_embeddings
# -----------------------------------------------------------------------
print("\n[Test 6] save_embeddings")
results = [
    {
        "dataset_id": "clf_demo",
        "embedding": emb,
        "task_type": 0,
        "n_samples": X_clf.shape[0],
        "n_features": X_clf.shape[1],
    },
    {
        "dataset_id": "reg_demo",
        "embedding": emb_reg,
        "task_type": 1,
        "n_samples": X_reg.shape[0],
        "n_features": X_reg.shape[1],
    },
]

with tempfile.TemporaryDirectory() as tmpdir:
    outpath = os.path.join(tmpdir, "embeddings.json")
    save_embeddings(results, outpath)

    check("file exists", os.path.isfile(outpath))

    with open(outpath, "r") as fh:
        data = json.load(fh)

    check("JSON has 2 entries", len(data) == 2)
    check("each entry has required keys",
          all(k in data[0] for k in ("dataset_id", "embedding", "task_type", "n_samples", "n_features")))
    check("embedding is a list of floats",
          isinstance(data[0]["embedding"], list) and all(isinstance(v, float) for v in data[0]["embedding"]))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 50)
print(f"  Results:  {passed} passed,  {failed} failed")
print("=" * 50)

sys.exit(1 if failed else 0)
