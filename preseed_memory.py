"""
preseed_memory.py
=================
Phase 4.3 Pre-seeding Script.

Generates 25-50 realistic dataset embedding signatures and populates the 
UnifiedMemoryStore with mocked best ML and best DL architectures to 
bootstrap the cold-start meta-learning routing algorithm.

Usage:
  python preseed_memory.py --output models/unified_memory.json --num_datasets 30
"""

import argparse
import os
import random
import time
import numpy as np

from unified_memory import UnifiedMemoryStore, MemoryEntry, PerformanceMetrics
from dataset_embedding import EMBEDDING_DIM

def generate_mock_embedding(seed: int) -> np.ndarray:
    """Generate a realistic Phase 4.1 embedding profile using normal distributions."""
    rng = np.random.RandomState(seed)
    
    # 0 n_samples, 1 n_features, 2 missing_ratio, 3 mean_skew, 4 mean_kurtosis, 
    # 5 entropy, 6 corr_density, 7 interaction_score, 8 task_type
    
    emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    # Log scaled row/col counts
    emb[0] = rng.uniform(5.0, 15.0) 
    emb[1] = rng.uniform(1.0, 6.0)
    emb[2] = rng.uniform(0.0, 0.3)
    emb[3] = rng.uniform(-1.0, 1.0)
    emb[4] = rng.uniform(-1.0, 5.0)
    emb[5] = rng.uniform(0.0, 2.0)
    emb[6] = rng.uniform(0.0, 0.8)
    emb[7] = rng.uniform(0.0, 1.0)
    emb[8] = float(rng.choice([0, 1, 2])) # task type (bin/reg/multi)
    
    # Normalize mimicking Phase 4.1 normalize_embedding behavior
    vmin, vmax = emb.min(), emb.max()
    if vmax - vmin > 1e-12:
        emb = (emb - vmin) / (vmax - vmin)
        
    return emb

def preseed(output_path: str, num_datasets: int) -> None:
    print(f"Pre-seeding Unified FAISS Memory with {num_datasets} datasets...")
    
    store = UnifiedMemoryStore()
    ml_models = ["rf", "gb", "xgboost", "logistic", "lightgbm"]
    
    for i in range(num_datasets):
        # 1. Compute realistic dataset embedding 
        emb = [float(x) for x in generate_mock_embedding(seed=42 + i)]
        task_type = "classification" if emb[8] < 0.5 or emb[8] > 0.8 else "regression"
        dataset_id = f"openml_{100 + i}"
        
        # 2. Simulate ML Training (Store Best ML)
        best_ml = random.choice(ml_models)
        perf_ml = random.uniform(0.70, 0.98) if task_type == "classification" else random.uniform(5.0, 30.0) 
        time_ml = random.uniform(5.0, 120.0)
        
        ml_entry = MemoryEntry(
            dataset_embedding=emb,
            paradigm="ML",
            model_name=best_ml,
            hyperparameters={"n_estimators": random.choice([50, 100, 200]), "max_depth": random.choice([3, 5, 10])},
            performance=PerformanceMetrics(f1=perf_ml, accuracy=perf_ml+0.01) if task_type == "classification" else PerformanceMetrics(rmse=perf_ml),
            training_time=time_ml,
            dataset_id=dataset_id,
            task_type=task_type
        )
        store.add(ml_entry)
        
        # 3. Simulate DL Training (Store Best DL)
        layers = random.choice([2, 3, 5])
        units = random.choice([64, 128, 256])
        time_dl = random.uniform(30.0, 600.0)  # generally slower
        # High correlation with ML performance, sometimes slightly better handling complex interactions
        perf_dl = min(0.99, perf_ml + random.uniform(-0.05, 0.08)) if task_type == "classification" else max(1.0, perf_ml * random.uniform(0.8, 1.2))
        
        dl_entry = MemoryEntry(
            dataset_embedding=emb,
            paradigm="DL",
            architecture_config={"layers": layers, "units_per_layer": units, "activation": "relu"},
            hyperparameters={"batch_size": random.choice([32, 64, 128]), "learning_rate": 0.001},
            performance=PerformanceMetrics(f1=perf_dl, accuracy=perf_dl+0.01) if task_type == "classification" else PerformanceMetrics(rmse=perf_dl),
            training_time=time_dl,
            dataset_id=dataset_id,
            task_type=task_type
        )
        store.add(dl_entry)
        
    # Build FAISS shared multi-paradigm index
    store.build_index()
    
    # Save decoupled schema to explicit output
    store.save(output_path)
    print(f"[OK] Seeded {len(store)} unified entries (2x per dataset).")
    print(f"[OK] FAISS index active. Dimensionality: {store.index.d}. Metric: L2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/unified_memory.json", help="Path to save the JSON mapping.")
    parser.add_argument("--num_datasets", type=int, default=30, help="Number of datasets to simulate.")
    args = parser.parse_args()
    
    preseed(args.output, args.num_datasets)
