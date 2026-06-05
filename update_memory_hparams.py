from cold_start import MemoryStore
from data_loader import detect_problem_type
import pickle
import os

MEMORY_INDEX_PATH = "memory_store.faiss"
MEMORY_META_PATH = "memory_store.pkl"

store = MemoryStore()
if os.path.exists(MEMORY_INDEX_PATH):
    store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
else:
    raise FileNotFoundError("Memory store not found.")

updated_count = 0
for idx, record in enumerate(store.records):
    # Skip if already has hparams
    if record.metadata.get("hparams") and len(record.metadata["hparams"]) > 0:
        continue
        
    # Fetch dataset from OpenML to extract best model params
    from sklearn.datasets import fetch_openml
    try:
        did = int(record.metadata["dataset_id"])
        data = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = data.data, data.target
        
        # Quick baseline screen to find best model
        from model_trainer import get_models, baseline_screen
        from feature_processing import build_preprocessor
        preprocessor, _, _ = build_preprocessor(X)
        problem_type = detect_problem_type(y)
        models = get_models(problem_type)
        _, scores = baseline_screen(models, preprocessor, X, y, problem_type, sample_frac=0.05)
        
        if scores:
            best_model_name = max(scores, key=lambda k: scores[k]['score'])
            hparams = models[best_model_name].get_params()
            
            # Update record metadata
            record.metadata["hparams"] = {best_model_name: hparams}
            updated_count += 1
            print(f"[Update] Dataset {did}: Added hparams for {best_model_name}")
            
    except Exception as e:
        print(f"[WARN] Failed to update Dataset {record.metadata['dataset_id']}: {e}. Skipping.")
        continue

# Save updated memory
store.save_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
print(f"\\n✅ Updated {updated_count} records with hyperparameters.")
