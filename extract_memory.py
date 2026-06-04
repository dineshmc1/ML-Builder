import sys
import re

with open("phase4_pipeline.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Collect necessary blocks
# 1. Imports
imports = []
for line in lines:
    if line.startswith("import") or line.startswith("from"):
        if "get_heuristic_suggestions" not in line and "get_llm_suggestions" not in line and "routing_engine" not in line and "paradigm_router" not in line and "dataset_profiler" not in line and "load_local_dataset" not in line:
            imports.append(line)
    elif line.startswith("# 50 OpenML"):
        break

# 2. DATASET_IDS, load_and_preprocess_openml, build_memory
blocks = []
in_block = False
current_block = []

def get_block(start_str, stop_str=None):
    b = []
    capture = False
    for line in lines:
        if line.startswith(start_str):
            capture = True
        if capture:
            b.append(line)
            if stop_str and line.startswith(stop_str):
                break
            if not stop_str and line.startswith("def ") and not line.startswith(start_str):
                # Backtrack one line if we hit next def
                b.pop()
                break
    return b

dataset_ids_block = get_block("DATASET_IDS = [", "]\n")
load_openml_block = get_block("def load_and_preprocess_openml(dataset_id):")

# Compute similarity, etc.
similarity_block = get_block("def compute_similarity(query_vec, memory_vectors, k=5):")
threshold_block = get_block("def compute_threshold(similarities, epsilon_lambda=0.5):")
query_block = get_block("def query_memory(query_vec, store, k=5):")
build_memory_block = get_block("def build_memory(train_ids, store=None, config=None):")

# The main block for building memory
main_code = """
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
        
    print("\\n" + "="*50)
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
"""

with open("build_memory.py", "w", encoding="utf-8") as f:
    f.writelines(imports)
    f.write("\n")
    f.writelines(dataset_ids_block)
    f.write("\n")
    f.writelines(load_openml_block)
    f.write("\n")
    f.writelines(similarity_block)
    f.write("\n")
    f.writelines(threshold_block)
    f.write("\n")
    f.writelines(query_block)
    f.write("\n")
    f.writelines(build_memory_block)
    f.write("\n")
    f.write(main_code)
