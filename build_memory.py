import time
import numpy as np
import pandas as pd
import faiss
from sklearn.datasets import fetch_openml
import random
from dataset_embedding import compute_dataset_embedding
from data_loader import detect_problem_type
from data_cleaner import clean
from feature_processing import build_preprocessor
from model_trainer import get_models, baseline_screen
from cold_start import (
from config import USE_LLM, USE_WANDB

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
    564, 1027, 1028, 1029, 1030,

    # ---------------- EXPANDED 300 DATASET POOL ----------------
    # Multi-class classification
    23381, 40691, 1468, 1475, 1478, 1480, 1485, 1486, 1487,
    1488, 4134, 6332, 23517, 40670, 40701,
    # Binary classification  
    179, 184, 554, 772, 917, 1019, 1020, 1021, 1040, 1053,
    1063, 1068, 4538, 6956, 40536,
    # Regression
    41702, 42225, 43071, 43439, 43551, 41278, 42563, 41980, 43928, 44027,
    # High dimensional
    1169, 1170, 1442, 1443, 1444, 1446, 1447, 1448,
    # Various / Leftovers
    34
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
        
        if len(X) < 50:
            print(f"  -> Dataset {dataset_id} too small ({len(X)} samples). Skipping.")
            return None, None
            
        X.columns = [
            str(col).replace('[', '').replace(']', '')
                     .replace('{', '').replace('}', '')
                     .replace('"', '').replace("'", '')
                     .replace(':', '_').replace(',', '_')
                     .replace('<', '').replace('>', '')
            for col in X.columns
        ]
        
        # Sanitize categorical values to prevent OneHotEncoder from creating bad feature names
        import re
        def clean_json_chars(val):
            if pd.isna(val):
                return val
            return re.sub(r'[\[\]\{\}"\':,<>]', '', str(val))
            
        for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
            if str(X[col].dtype) == 'category':
                new_categories = [clean_json_chars(c) for c in X[col].cat.categories]
                X[col] = X[col].cat.rename_categories(new_categories)
            else:
                X[col] = X[col].astype(str).apply(clean_json_chars)
            
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
            
        # Clean X to handle missing values and keep y in sync
        X, y = clean(X, y=y, verbose=False)
            
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

            
        y = data[target_column]
        X = data.drop(columns=[target_column])
        
        if len(X) < 50:
            print(f"  -> Dataset too small ({len(X)} samples). Skipping.")
            return None, None
            
        X.columns = [
            str(col).replace('[', '').replace(']', '')
                     .replace('{', '').replace('}', '')
                     .replace('"', '').replace("'", '')
                     .replace(':', '_').replace(',', '_')
                     .replace('<', '').replace('>', '')
            for col in X.columns
        ]
        
        # Sanitize categorical values to prevent OneHotEncoder from creating bad feature names
        import re
        def clean_json_chars(val):
            if pd.isna(val):
                return val
            return re.sub(r'[\[\]\{\}"\':,<>]', '', str(val))
            
        for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
            if str(X[col].dtype) == 'category':
                new_categories = [clean_json_chars(c) for c in X[col].cat.categories]
                X[col] = X[col].cat.rename_categories(new_categories)
            else:
                X[col] = X[col].astype(str).apply(clean_json_chars)
            
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
            
        # Clean X to handle missing values and keep y in sync
        X, y = clean(X, y=y, verbose=False)
            
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
        print(f"  -> Failed to load local dataset: {e}")
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


def build_memory(train_ids, store=None, config=None):
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
    
    if store is None:
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
                
            from multi_objective import select_best_model_multiobjective
            
            if problem_type == 'regression':
                w1_def, w2_def, w3_def = 0.8, 0.15, 0.05
            else:
                w1_def, w2_def, w3_def = 0.6, 0.3, 0.1
                
            best_model_by_score = max(scores, key=lambda k: scores[k]['score'])
            best_model_by_utility, utility_scores = select_best_model_multiobjective(scores, task_type=problem_type, w1=w1_def, w2=w2_def, w3=w3_def)

            best_model_name = best_model_by_utility
            best_score = scores[best_model_name]['score']
            
            if best_model_by_score != best_model_by_utility:
                print(f"  [Multi-Obj] Score winner: {best_model_by_score} ({scores[best_model_by_score]['score']:.4f}) vs Utility winner: {best_model_by_utility} (utility={utility_scores[best_model_by_utility]:.4f})")
            
        except Exception as e:
            print(f"  -> Error during training/embedding: {e}")
            continue
            
        elapsed = time.time() - start_time
        
        from model_trainer import REGRESSION_MODELS, CLASSIFICATION_MODELS
        cat = REGRESSION_MODELS if problem_type == 'regression' else CLASSIFICATION_MODELS
        hparams = cat[best_model_name].get_params()
        
        # Store in FAISS memory mapping using MemoryStore
        metadata = {
            "dataset_id": did,
            "problem_type": problem_type,
            "score": best_score,
            "time": elapsed,
            "hparams": {best_model_name: hparams}
        }
        store.add(str(did), vec, [best_model_name], metadata)
        print(f"  -> Successfully committed to memory: Best Model='{best_model_name}' (Score: {best_score:.4f})")
    
    print("\n[Memory Builder] Initializing FAISS Index...")
    store.build_index()
    return store

ROUTING_CFG = RoutingConfig(
    lambda_memory=0.6,
    lambda_llm=0.2,
    lambda_heuristic=0.2,
    use_llm=USE_LLM,
    top_k_output=3
)



def main():
    import os
    import json
    from datetime import datetime
    import random
    import numpy as np

    MEMORY_INDEX_PATH = "memory_store.faiss"
    MEMORY_META_PATH  = "memory_store.pkl"
    
    np.random.seed(42)
    random.seed(42)
    
    all_ids = list(dict.fromkeys(DATASET_IDS))
    random.shuffle(all_ids)
    
    train_limit = int(len(all_ids) * 0.8)
    train_ids = all_ids[:train_limit]
    
    print(f"Total datasets : {len(all_ids)}")
    print(f"Training sets  : {len(train_ids)}")
    
    store = MemoryStore()
    
    if os.path.exists(MEMORY_INDEX_PATH):
        loaded = store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"Loaded {loaded} existing records from disk")
        existing_keys = store.get_keys()
        remaining_train = [d for d in train_ids if f"openml_{d}" not in existing_keys]
        print(f"Skipping {len(train_ids)-len(remaining_train)} already-processed datasets")
    else:
        remaining_train = train_ids
        
    if remaining_train:
        store = build_memory(remaining_train, store, config=None)
        store.save_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"Memory saved. Total records: {len(store.records)}")
    else:
        print("Memory fully loaded from disk. No new datasets to process.")
        
    print("\n" + "="*50)
    print("TRAINING TASK ENCODER")
    print("="*50)
    
    from task_encoder import train_encoder, encode_all, TaskEncoderConfig
    
    cfg = TaskEncoderConfig(
        input_dim=10, hidden_dim=64, output_dim=32,
        epochs=100, early_stopping_patience=20
    )
    
    encoder, history = train_encoder(store, config=cfg, force_retrain=False)
    
    print(f"Encoder trained. Best epoch: {history['best_epoch']}")
    print(f"Early stopped: {history['stopped_early']}")
    
    learned_vectors = encode_all(store, encoder)
    store.rebuild_index(learned_vectors)
    store.save_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
    print(f"FAISS rebuilt and saved with {len(learned_vectors)} learned embeddings")

if __name__ == "__main__":
    main()
