"""
test_cold_start.py
==================
Verification script for Phase 4.2 — Adaptive Cold-Start Strategy.

Tests:
  1. ColdStartConfig defaults and validation
  2. Cosine-similarity computation accuracy
  3. Adaptive threshold eps(D) computation
  4. Overall similarity score (max vs top-k mean)
  5. MemoryStore add / index / query workflow
  6. Decision logic: memory path (high similarity)
  7. Decision logic: cold-start path (low similarity)
  8. Edge case: empty memory -> always cold-start
  9. ColdStartLogger captures entries and exports JSON
 10. Full integration: embedding -> FAISS -> cold-start decision
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

from cold_start import (
    ColdStartConfig,
    ColdStartResult,
    ColdStartLogger,
    MemoryStore,
    adaptive_cold_start,
    compute_similarity_scores,
    compute_adaptive_threshold,
    compute_overall_similarity,
    _cosine_similarity,
    get_fallback_models,
)
from dataset_embedding import compute_dataset_embedding, EMBEDDING_DIM

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  -- {detail}")


# -----------------------------------------------------------------------
# Helpers: Create synthetic data & embeddings
# -----------------------------------------------------------------------

def _make_synthetic_dataset(n=200, f=5, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, f), columns=[f"feat_{i}" for i in range(f)])
    y = pd.Series(rng.choice([0, 1], size=n), name="target")
    return X, y


def _make_embedding(seed=42):
    X, y = _make_synthetic_dataset(seed=seed)
    return compute_dataset_embedding(X, y)


# -----------------------------------------------------------------------
# Test 1: ColdStartConfig defaults and validation
# -----------------------------------------------------------------------
print("\n[Test 1] ColdStartConfig")
cfg = ColdStartConfig()
check("default k_neighbors", cfg.k_neighbors == 10)
check("default lambda_sensitivity", cfg.lambda_sensitivity == 0.5)
check("default memory_models_count", cfg.memory_models_count == 3)
check("default fallback_models_count", cfg.fallback_models_count == 5)
check("default use_top_k_mean", cfg.use_top_k_mean is True)
check("default top_k_for_score", cfg.top_k_for_score == 3)

# Validation
try:
    ColdStartConfig(k_neighbors=0)
    check("invalid k_neighbors raises", False, "no error raised")
except ValueError:
    check("invalid k_neighbors raises", True)

cfg_custom = ColdStartConfig(
    k_neighbors=5,
    lambda_sensitivity=1.0,
    memory_models_count=5,
    fallback_models_count=8,
)
check("custom config values", cfg_custom.k_neighbors == 5 and cfg_custom.lambda_sensitivity == 1.0)


# -----------------------------------------------------------------------
# Test 2: Cosine similarity
# -----------------------------------------------------------------------
print("\n[Test 2] Cosine similarity computation")
a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
check("identical vectors -> sim=1.0", abs(_cosine_similarity(a, b) - 1.0) < 1e-6)

c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
check("orthogonal vectors -> sim=0.0", abs(_cosine_similarity(a, c)) < 1e-6)

d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
check("opposite vectors -> sim=-1.0", abs(_cosine_similarity(a, d) + 1.0) < 1e-6)

zero = np.zeros(3, dtype=np.float32)
check("zero vector -> sim=0.0", abs(_cosine_similarity(a, zero)) < 1e-6)


# -----------------------------------------------------------------------
# Test 3: Adaptive threshold epsilon(D)
# -----------------------------------------------------------------------
print("\n[Test 3] Adaptive threshold computation")
sims = np.array([0.9, 0.85, 0.8, 0.7, 0.6])
eps, mu, sigma = compute_adaptive_threshold(sims, lambda_sensitivity=0.5)
expected_mu = float(np.mean(sims))
expected_sigma = float(np.std(sims))
expected_eps = expected_mu - 0.5 * expected_sigma

check("mu_S correct", abs(mu - expected_mu) < 1e-6, f"got {mu}, expected {expected_mu}")
check("sigma_S correct", abs(sigma - expected_sigma) < 1e-6, f"got {sigma}, expected {expected_sigma}")
check("epsilon(D) correct", abs(eps - expected_eps) < 1e-6, f"got {eps}, expected {expected_eps}")

# lambda=0 -> epsilon = mu
eps0, _, _ = compute_adaptive_threshold(sims, lambda_sensitivity=0.0)
check("lambda=0 -> epsilon=mu", abs(eps0 - expected_mu) < 1e-6)


# -----------------------------------------------------------------------
# Test 4: Overall similarity score
# -----------------------------------------------------------------------
print("\n[Test 4] Overall similarity score")
sims_unsorted = np.array([0.6, 0.9, 0.8, 0.85, 0.7])
s_max = compute_overall_similarity(sims_unsorted, use_top_k_mean=False)
check("max similarity", abs(s_max - 0.9) < 1e-6, f"got {s_max}")

s_top3 = compute_overall_similarity(sims_unsorted, use_top_k_mean=True, top_k=3)
expected_top3 = float(np.mean([0.9, 0.85, 0.8]))
check("top-3 mean", abs(s_top3 - expected_top3) < 1e-6, f"got {s_top3}, expected {expected_top3}")

# Single element
s_single = compute_overall_similarity(np.array([0.75]), use_top_k_mean=True, top_k=3)
check("single similarity", abs(s_single - 0.75) < 1e-6)


# -----------------------------------------------------------------------
# Test 5: MemoryStore workflow
# -----------------------------------------------------------------------
print("\n[Test 5] MemoryStore")
store = MemoryStore()
check("empty store", len(store) == 0)

emb1 = _make_embedding(seed=42)
emb2 = _make_embedding(seed=99)
emb3 = _make_embedding(seed=7)

store.add("dataset_1", emb1, ["rf", "gb"], {"task": "clf"})
store.add("dataset_2", emb2, ["logistic", "rf"], {"task": "clf"})
store.add("dataset_3", emb3, ["gb", "lightgbm"], {"task": "clf"})
check("store has 3 records", len(store) == 3)

# Build index
store.build_index()
check("index built", store.index is not None)
check("embeddings shape", store.embeddings.shape == (3, EMBEDDING_DIM))

# Model voting: query with emb1 -> should find dataset_1 first
models = store.get_models_for_indices(np.array([0, 1, 2]), top_n=3)
check("get_models returns list", isinstance(models, list) and len(models) > 0)
check("models contain rf", "rf" in models)  # rf appears in dataset_1 and dataset_2


# -----------------------------------------------------------------------
# Test 6: Decision logic — MEMORY path
# -----------------------------------------------------------------------
print("\n[Test 6] Memory path (high similarity)")
# Query with the same embedding as dataset_1 -> should get high similarity
result_mem = adaptive_cold_start(
    emb1, store,
    config=ColdStartConfig(k_neighbors=3, lambda_sensitivity=0.5),
    problem_type="classification",
)
check("result is ColdStartResult", isinstance(result_mem, ColdStartResult))
check("decision is 'memory'", result_mem["decision"] == "memory",
      f"got '{result_mem['decision']}' (S={result_mem['similarity_score']}, eps={result_mem['epsilon']})")
check("epsilon is float", isinstance(result_mem["epsilon"], float))
check("similarity_score is float", isinstance(result_mem["similarity_score"], float))
check("models_selected is list", isinstance(result_mem["models_selected"], list))
check("S(D,M) >= eps(D)", result_mem["similarity_score"] >= result_mem["epsilon"],
      f"S={result_mem['similarity_score']}, eps={result_mem['epsilon']}")


# -----------------------------------------------------------------------
# Test 7: Decision logic — COLD-START path
# -----------------------------------------------------------------------
print("\n[Test 7] Cold-start path (low similarity)")
# Create a very different embedding (noise)
rng = np.random.RandomState(999)
noise_emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
noise_emb = noise_emb / (np.linalg.norm(noise_emb) + 1e-12)

# Use aggressive lambda to force cold-start (high threshold)
result_cs = adaptive_cold_start(
    noise_emb, store,
    config=ColdStartConfig(k_neighbors=3, lambda_sensitivity=-2.0),  # very high threshold
    problem_type="classification",
)
check("decision is 'cold_start'", result_cs["decision"] == "cold_start",
      f"got '{result_cs['decision']}' (S={result_cs['similarity_score']}, eps={result_cs['epsilon']})")
check("fallback models returned", len(result_cs["models_selected"]) > 0)


# -----------------------------------------------------------------------
# Test 8: Edge case — empty memory
# -----------------------------------------------------------------------
print("\n[Test 8] Empty memory -> cold-start")
empty_store = MemoryStore()
result_empty = adaptive_cold_start(
    emb1, empty_store,
    problem_type="classification",
)
check("decision is 'cold_start'", result_empty["decision"] == "cold_start")
check("epsilon is 0", result_empty["epsilon"] == 0.0)
check("similarity_score is 0", result_empty["similarity_score"] == 0.0)
check("models selected > 0", len(result_empty["models_selected"]) > 0)


# -----------------------------------------------------------------------
# Test 9: ColdStartLogger
# -----------------------------------------------------------------------
print("\n[Test 9] ColdStartLogger")
cs_logger = ColdStartLogger()

# Log a decision through the router
result_logged = adaptive_cold_start(
    emb1, store,
    config=ColdStartConfig(k_neighbors=3),
    cs_logger=cs_logger,
)
check("logger has 1 entry", len(cs_logger.entries) == 1)
check("entry has timestamp", "timestamp" in cs_logger.entries[0])
check("entry has decision", cs_logger.entries[0]["decision"] == result_logged["decision"])
check("entry has mu_s", "mu_s" in cs_logger.entries[0])
check("entry has sigma_s", "sigma_s" in cs_logger.entries[0])
check("entry has epsilon", "epsilon" in cs_logger.entries[0])
check("entry has num_models_evaluated", "num_models_evaluated" in cs_logger.entries[0])

# Export to JSON
with tempfile.TemporaryDirectory() as tmpdir:
    json_path = cs_logger.to_json(os.path.join(tmpdir, "cold_start_log.json"))
    check("JSON file created", os.path.isfile(json_path))

    with open(json_path, "r") as fh:
        data = json.load(fh)
    check("JSON has 1 entry", len(data) == 1)
    check("JSON entry has all keys",
          all(k in data[0] for k in ("timestamp", "mu_s", "sigma_s", "epsilon",
                                      "similarity_score", "decision", "num_models_evaluated")))

# DataFrame export
df = cs_logger.to_dataframe()
check("DataFrame has 1 row", len(df) == 1)
check("DataFrame has expected columns", "decision" in df.columns and "epsilon" in df.columns)


# -----------------------------------------------------------------------
# Test 10: Full integration (embedding -> FAISS -> decision)
# -----------------------------------------------------------------------
print("\n[Test 10] Full integration")
# Build a memory store with real embeddings
int_store = MemoryStore()
for seed in [10, 20, 30, 40, 50]:
    X, y = _make_synthetic_dataset(seed=seed)
    emb = compute_dataset_embedding(X, y)
    int_store.add(f"ds_{seed}", emb, ["rf", "gb"])

# Query with a new dataset
X_new, y_new = _make_synthetic_dataset(seed=42)
query = compute_dataset_embedding(X_new, y_new)

cs_log = ColdStartLogger()
result = adaptive_cold_start(
    query, int_store,
    config=ColdStartConfig(k_neighbors=5, lambda_sensitivity=0.5, use_top_k_mean=True),
    cs_logger=cs_log,
)

check("result has 'decision'", "decision" in result)
check("result has 'epsilon'", "epsilon" in result)
check("result has 'similarity_score'", "similarity_score" in result)
check("result has 'models_selected'", "models_selected" in result)
check("result has 'mu_s'", "mu_s" in result)
check("result has 'sigma_s'", "sigma_s" in result)
check("result has 'similarities'", "similarities" in result)
check("result has 'neighbor_indices'", "neighbor_indices" in result)
check("result has 'num_models_evaluated'", "num_models_evaluated" in result)
check("decision is valid", result["decision"] in ("memory", "cold_start"))
check("similarities length matches k", len(result["similarities"]) == 5)
check("logger captured entry", len(cs_log.entries) == 1)

# Print decision details for visual inspection
print(f"\n    Decision: {result['decision']}")
print(f"    S(D,M)  = {result['similarity_score']:.4f}")
print(f"    eps(D)  = {result['epsilon']:.4f}")
print(f"    mu_S    = {result['mu_s']:.4f}")
print(f"    sigma_S = {result['sigma_s']:.4f}")
print(f"    Models  = {result['models_selected']}")


# -----------------------------------------------------------------------
# Test 11: Configurable parameters test
# -----------------------------------------------------------------------
print("\n[Test 11] Configurable parameters")
# Test with different configurations
cfg_custom = ColdStartConfig(
    k_neighbors=3,
    lambda_sensitivity=1.5,
    memory_models_count=5,
    fallback_models_count=8,
    use_top_k_mean=False,
    top_k_for_score=1,
)
result_custom = adaptive_cold_start(
    query, int_store, config=cfg_custom,
)
check("custom config accepted", "decision" in result_custom)
check("similarities length matches custom k", len(result_custom["similarities"]) == 3)

# Test fallback model count
fallback_clf = get_fallback_models("classification", count=8)
fallback_reg = get_fallback_models("regression", count=8)
check("fallback clf returns list", isinstance(fallback_clf, list))
check("fallback reg returns list", isinstance(fallback_reg, list))
check("clf and reg have different models", fallback_clf != fallback_reg)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"  Phase 4.2 Results:  {passed} passed,  {failed} failed")
print("=" * 60)

sys.exit(1 if failed else 0)
