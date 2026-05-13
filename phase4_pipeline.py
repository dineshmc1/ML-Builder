import time
import numpy as np
import pandas as pd
import faiss
from sklearn.datasets import fetch_openml
import random

# Custom imports from our pipeline
from dataset_embedding import compute_dataset_embedding
from data_loader import detect_problem_type
from data_cleaner import clean
from feature_processing import build_preprocessor
from model_trainer import get_models, baseline_screen
from cold_start import (
    MemoryStore,
    ColdStartConfig,
    adaptive_cold_start,
    compute_adaptive_threshold,
    _cosine_similarity
)

# 50 OpenML dataset IDs for classification/regression
DATASET_IDS = [
    # ---------------- CLASSIFICATION ----------------
    61, 31, 153, 44, 1504, 1494, 1462, 37, 1464, 40945,
    1049, 40983, 54, 181, 1510, 40668, 23, 1489, 1120, 38,
    46, 182, 300, 4534, 1067,

    # ---------------- REGRESSION ----------------
    41021, 507, 531, 422, 41540, 560, 574, 589, 1199, 42092,
    42165, 42705, 42726, 42727, 42728,

    # ---------------- MORE DIVERSE DATASETS ----------------
    1590, 151, 11, 14, 16, 18, 22, 50, 188, 307,

    # ---------------- ADDITIONAL 50 DATASETS ----------------
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 
    13, 15, 17, 19, 20, 21, 24, 25, 26, 27, 
    28, 29, 30, 32, 33, 34, 35, 36, 39, 40, 
    41, 42, 43, 45, 47, 48, 49, 51, 52, 53, 
    55, 56, 57, 58, 59, 60, 62, 63, 64, 65,

    # ---------------- ADDITIONAL 15 REGRESSION DATASETS ----------------
    189, 197, 201, 214, 225, 227, 228, 229, 230, 549, 
    564, 1027, 1028, 1029, 1030
]


def load_and_preprocess_openml(dataset_id):
    """
    Helper function to load dataset from OpenML.
    Returns X (DataFrame), y (Series).
    """
    try:
        print(f"  -> Fetching OpenML ID: {dataset_id}...")
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        X = data.data
        y = data.target
        
        # Subsample if dataset is too large, to keep the test fast
        if len(X) > 2000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), 2000, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            
        # Basic imputation to avoid errors if DataFrame contains NaNs
        # Our internal pipeline handles this, but ensuring safe targets
        if y.isna().any():
            y.fillna(method='ffill', inplace=True)
            
        # Clean X to handle missing values
        X = clean(X, verbose=False)
            
        # Detect problem type early and apply categorical encoding if needed
        problem_type = detect_problem_type(y)
        from sklearn.preprocessing import LabelEncoder
        if y.dtype == object or str(y.dtype) == 'category' or problem_type == 'classification':
            y = pd.Series(
                LabelEncoder().fit_transform(y.astype(str)), 
                name=y.name, 
                index=y.index
            )
            
        return X, y
    except Exception as e:
        print(f"  -> Failed to load dataset {dataset_id}: {e}")
        return None, None

def compute_similarity(query_vec, memory_vectors, k=5):
    """
    5. Similarity Function
    Computes cosine similarity between query and multiple memory vectors.
    Returns top-K similarities and indices.
    """
    sims = []
    for mem_vec in memory_vectors:
        sims.append(_cosine_similarity(query_vec, mem_vec))
        
    sims = np.array(sims)
    # Sort descending
    indices = np.argsort(sims)[::-1][:k]
    return sims[indices], indices

def compute_threshold(similarities, epsilon_lambda=0.5):
    """
    6. Adaptive Threshold ε(D)
    Returns epsilon, mean, std.
    """
    return compute_adaptive_threshold(similarities, lambda_sensitivity=epsilon_lambda)

def query_memory(query_vec, store, k=5):
    """Query FAISS for top-K neighbors."""
    return store.search(query_vec, top_k=k)

def build_memory(train_ids):
    """
    3. Memory Building Phase
    For each dataset:
    - Load data
    - Compute embedding vector
    - Train all models to find the best
    - Store mapping in FAISS
    """
    print("\n" + "="*50)
    print("PHASE 3: BUILDING MEMORY STORE")
    print("="*50)
    
    store = MemoryStore()
    
    for did in train_ids:
        print(f"\n[Memory Builder] Processing Dataset {did}")
        X, y = load_and_preprocess_openml(did)
        if X is None:
            continue
            
        start_time = time.time()
        # Find problem type
        problem_type = detect_problem_type(y)
        
        # Extract meta-features
        vec = compute_dataset_embedding(X, y)
        
        # Train models to get best configuration
        try:
            preprocessor, _, _ = build_preprocessor(X)
            all_models = get_models(problem_type)
            
            # Use baseline screening on full sample to quickly pick best model
            _, scores = baseline_screen(
                all_models, preprocessor, X, y, problem_type,
                sample_frac=1.0, cv=3, random_state=42
            )
            
            if not scores:
                print("  -> Training failed. Skipping.")
                continue
                
            best_model_name = max(scores, key=scores.get)
            best_score = scores[best_model_name]
            
        except Exception as e:
            print(f"  -> Error during training/embedding: {e}")
            continue
            
        elapsed = time.time() - start_time
        
        # Store in FAISS memory mapping using MemoryStore
        metadata = {
            "dataset_id": did,
            "problem_type": problem_type,
            "score": best_score,
            "time": elapsed
        }
        store.add(f"openml_{did}", vec, [best_model_name], metadata)
        print(f"  -> Successfully committed to memory: Best Model='{best_model_name}' (Score: {best_score:.4f})")
    
    print("\n[Memory Builder] Initializing FAISS Index...")
    store.build_index()
    return store

def decision_engine(query_vec, store, problem_type):
    """
    7. Decision Logic
    Evaluates similarity S(D, M) vs ε(D) to select path (MEMORY vs FALLBACK).
    Delegates to the existing `adaptive_cold_start` which implements Phase 4 logic.
    """
    cfg = ColdStartConfig(k_neighbors=5, lambda_sensitivity=0.5)
    
    # Cold start logic returns decision and thresholds
    result = adaptive_cold_start(query_vec, store, config=cfg, problem_type=problem_type)
    
    decision = "USE MEMORY" if result["decision"] == "memory" else "FALLBACK"
    return (
        decision,
        result["similarity_score"],
        result["epsilon"],
        result["models_selected"]
    )


def main():
    print("\nUsing RANDOM SEED = 42")
    # Set random seed for reproducibility
    np.random.seed(42)
    
    all_query_vecs = {}
    
    # =====================================================
    # RANDOMIZED 80/20 SPLIT
    # =====================================================

    random.seed(42)

    all_ids = DATASET_IDS.copy()
    random.shuffle(all_ids)

    train_ids = all_ids[:92]
    test_ids = all_ids[92:115]
    
    print(f"Datasets mapped to Knowledge Base (Memory): {train_ids}")
    print(f"Unseen Datasets for Testing: {test_ids}")
    print(f"\nTotal datasets : {len(DATASET_IDS)}")
    print(f"Training sets  : {len(train_ids)}")
    print(f"Testing sets   : {len(test_ids)}")
    
    # 3. Build Memory
    store = build_memory(train_ids)
    
    print("\n" + "="*50)
    print("PHASE 4: TESTING SYSTEM LOGIC (ADAPTIVE COLD-START)")
    print("="*50)
    
    metrics = {
        "memory_decisions": 0,
        "fallback_decisions": 0,
        "total_similarity": 0.0,
        "total_models": 0,
        "processed_count": 0,
        # ADD THESE THREE:
        "total_score_gap": 0.0,
        "total_models_saved": 0,
        "score_validation_count": 0
    }

    # Store experiment results
    results = []

    successful_tests = 0
    failed_tests = 0
    
    for did in test_ids:
        print(f"\n[Test] Evaluating Dataset {did}...")
        X, y = load_and_preprocess_openml(did)
        if X is None:
            failed_tests += 1
            continue

        successful_tests += 1
            
        problem_type = detect_problem_type(y)
        
        # 1. Extract meta-features
        query_vec = compute_dataset_embedding(X, y)
        all_query_vecs[did] = query_vec

        print(f"Successful test datasets : {successful_tests}")
        print(f"Failed test datasets     : {failed_tests}")

        print(f"  [Embedding Raw Values] Dataset {did}: {query_vec}")

        # ABLATION LOG
        print(f"  [Embedding] Shape: {query_vec.shape}")
        print(f"  [Embedding] Mean: {query_vec.mean():.4f}")
        print(f"  [Embedding] Std:  {query_vec.std():.4f}")
        print(f"  [Embedding] Min:  {query_vec.min():.4f}")
        print(f"  [Embedding] Max:  {query_vec.max():.4f}")

        # 2, 3, 4. Threshold Stress Test (Replaces decision_engine)
        print(f"\n  [Threshold Stress Test] Dataset {did}")
        for lambda_val in [0.3, 0.5, 1.0, 1.5, 2.0]:
            cfg = ColdStartConfig(k_neighbors=5, lambda_sensitivity=lambda_val)
            result = adaptive_cold_start(query_vec, store, config=cfg, problem_type=problem_type)
            print(f"    lambda={lambda_val:.1f} -> decision={result['decision']:<8} | "
                  f"similarity={result['similarity_score']:.4f} | "
                  f"eps={result['epsilon']:.4f}")
            
            # Preserve variables to allow the rest of the loop to run normally with default behavior (lambda=0.5)
            if lambda_val == 0.5:
                decision = "USE MEMORY" if result["decision"] == "memory" else "FALLBACK"
                similarity = result["similarity_score"]
                threshold = result["epsilon"]
                selected_models = result["models_selected"]
        
        # Track metrics
        metrics["processed_count"] += 1
        metrics["total_similarity"] += similarity
        metrics["total_models"] += len(selected_models)
        
        if decision == "USE MEMORY":
            metrics["memory_decisions"] += 1
        else:
            metrics["fallback_decisions"] += 1
            
        # 8. Evaluation Logging
        print("-" * 30)
        print(f"Dataset: {did}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Decision: {decision}")
        print(f"Models tried: {len(selected_models)}")
        
        # ---- STEP 4: SCORE VALIDATION ----
        cs_score = 0.0
        full_score = 0.0
        full_model_count = 0

        # Cold-start: train only selected models
        try:
            preprocessor_cs, _, _ = build_preprocessor(X)
            test_models = get_models(problem_type, model_names=selected_models)
            _, cs_scores = baseline_screen(
                test_models, preprocessor_cs, X, y, problem_type,
                sample_frac=1.0, cv=3, random_state=42
            )
            if cs_scores:
                cs_score = cs_scores[max(cs_scores, key=cs_scores.get)]
        except Exception as e:
            print(f"  [Cold-Start Score] Failed: {e}")

        # Full benchmark: train ALL models
        try:
            preprocessor_full, _, _ = build_preprocessor(X)
            all_models_full = get_models(problem_type)
            full_model_count = len(all_models_full)
            _, all_scores = baseline_screen(
                all_models_full, preprocessor_full, X, y, problem_type,
                sample_frac=1.0, cv=3, random_state=42
            )
            if all_scores:
                full_score = all_scores[max(all_scores, key=all_scores.get)]
        except Exception as e:
            print(f"  [Full Benchmark] Failed: {e}")

        # Print per-dataset comparison
        score_gap = full_score - cs_score
        models_saved = full_model_count - len(selected_models)
        print(f"  Cold-Start Score : {cs_score:.4f} ({len(selected_models)} models tried)")
        print(f"  Full Train Score : {full_score:.4f} ({full_model_count} models tried)")
        print(f"  Score Gap        : {score_gap:+.4f}")
        print(f"  Models Saved     : {models_saved}")

        # Accumulate
        if full_score > 0.0 and cs_score > 0.0:
            metrics["total_score_gap"] += score_gap
            metrics["total_models_saved"] += models_saved
            metrics["score_validation_count"] += 1
            
        if full_score == 0.0 or cs_score == 0.0:
            print(f"  [Skipped from validation] Dataset {did} - "
                  f"cs_score={cs_score:.4f}, full_score={full_score:.4f}")
            
        # Save experiment result
        results.append({
            "dataset_id": did,
            "problem_type": problem_type,
            "similarity": similarity,
            "threshold": threshold,
            "decision": decision,
            "models_tried": len(selected_models),
            "final_score": cs_score,
            "full_score": full_score
        })
        print("-" * 30)

    print("\n[Ablation] Pairwise Cosine Similarities Between Test Embeddings:")
    dids = list(all_query_vecs.keys())
    for i in range(len(dids)):
        for j in range(i+1, len(dids)):
            sim = _cosine_similarity(all_query_vecs[dids[i]], all_query_vecs[dids[j]])
            print(f"  Dataset {dids[i]} vs {dids[j]}: {sim:.4f}")

    # 9. Sanity Checks & Summary
    print("\n" + "="*50)
    print("SUMMARY METRICS")
    print("="*50)
    if metrics["processed_count"] > 0:
        total = metrics["processed_count"]
        pct_mem = (metrics["memory_decisions"] / total) * 100
        pct_fb = (metrics["fallback_decisions"] / total) * 100
        avg_sim = metrics["total_similarity"] / total
        avg_models = metrics["total_models"] / total
        
        print(f"% using Memory : {pct_mem:.1f}%")
        print(f"% Fallback     : {pct_fb:.1f}%")
        print(f"Avg Similarity : {avg_sim:.4f}")
        print(f"Avg Models     : {avg_models:.2f}")

        if metrics["score_validation_count"] > 0:
            n = metrics["score_validation_count"]
            avg_gap = metrics["total_score_gap"] / n
            avg_saved = metrics["total_models_saved"] / n
            print(f"Avg Score Gap   : {avg_gap:+.4f}  (target: < 0.05)")
            print(f"Avg Models Saved: {avg_saved:.1f}   (target: >= 2)")
            print(f"Validated on    : {n} datasets")
    else:
        print("No test datasets were successfully processed.")

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv("phase4_results.csv", index=False)

    print("\nSaved results to phase4_results.csv")
        
    print("\nScript completed successfully.")

if __name__ == "__main__":
    main()
