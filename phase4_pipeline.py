import time
import numpy as np
import pandas as pd
import faiss
import random
import os
import json
import argparse
import re

# Custom imports from our pipeline
from dataset_embedding import compute_dataset_embedding
from data_loader import detect_problem_type
from data_loader import load_local_dataset
from data_cleaner import clean
from feature_processing import build_preprocessor
from model_trainer import get_models, baseline_screen
from cold_start import MemoryStore
from config import USE_LLM, USE_WANDB
from paradigm_router import route_paradigm
from dataset_profiler import profile_dataset

def extract_meta_features(X, y) -> dict:
    import numpy as np
    numeric_cols = X.select_dtypes(include=[np.number])
    cat_cols = X.select_dtypes(exclude=[np.number])
    n_samples, n_cols = X.shape
    n_num = numeric_cols.shape[1]
    n_cat = cat_cols.shape[1]
    n_classes = y.nunique()

    return {
        "n_samples":         n_samples,
        "n_features":        n_cols,
        "num_ratio":         n_num / max(n_cols, 1),
        "cat_ratio":         n_cat / max(n_cols, 1),
        "missing_rate":      X.isnull().mean().mean(),
        "skewness_mean":     numeric_cols.skew().mean() if n_num > 0 else 0.0,
        "mean_corr":         numeric_cols.corr().abs().values[np.triu_indices(n_num, k=1)].mean() if n_num > 1 else 0.0,
        "n_classes":         int(n_classes),
        "is_binary":         n_classes == 2,
        "target_entropy":    float(-(y.value_counts(normalize=True) * np.log(y.value_counts(normalize=True) + 1e-10)).sum()),
        "majority_class_ratio": float(y.value_counts(normalize=True).iloc[0])
    }

def sanitize_filename(dataset_id_or_path):
    """Converts a full path or ID into a safe filename string."""
    base = os.path.basename(str(dataset_id_or_path))
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', os.path.splitext(base)[0])
    return safe_name

def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False):
    import numpy as np
    from wandb_logger import log
    
    did_safe = sanitize_filename(did)
    did = did_safe
    
    # 1. Extract meta-features
    from task_encoder import encode_dataset
    raw_vec = compute_dataset_embedding(X, y)
    query_vec = encode_dataset(raw_vec, encoder).reshape(1, -1).astype(np.float32)
    
    print(f"\\n[DEBUG] Raw Meta-Features (10D): {np.round(raw_vec, 4)}")
    print(f"[DEBUG] Learned Embedding (32D) [First 5]: {np.round(query_vec[0][:5], 4)} ...")
    
    meta_features = extract_meta_features(X, y)
    
    # 2. Query FAISS memory
    dists, idxs = store._index.search(query_vec, k=5)
    neighbors = [store.records[i] for i in idxs[0] if i != -1]
    
    # Safety Check: If fewer than 3 valid neighbors found, trigger cold-start heuristics
    if len(neighbors) < 3:
        print("\n⚠️ WARNING: Fewer than 3 valid neighbors found in FAISS. Triggering cold-start fallback heuristics.")
        from heuristics import get_heuristic_suggestions
        best_retrieved_models = get_heuristic_suggestions(meta_features, problem_type)[:3]
        warm_params = {}
    else:
        best_retrieved_models = []
        for n in neighbors[:3]:
            if n.models:
                best_retrieved_models.extend(n.models)
        # Deduplicate while preserving order
        best_retrieved_models = list(dict.fromkeys(best_retrieved_models))[:3]
        warm_params = neighbors[0].metadata.get("hparams", {})
        
        print(f"\n📂 DATASET: {did} | TARGET: {y.name if hasattr(y, 'name') else 'Unknown'} | TYPE: {problem_type}\n")
        print("🔍 MEMORY RETRIEVAL (Top 3 Similar Past Experiments):")
        print("┌────────────┬────────────┬──────────────┬──────────────────────────────┐")
        print("│ Dataset    │ Similarity │ Best Model   │ Warm-Start Hyperparameters   │")
        print("├────────────┼────────────┼──────────────┼──────────────────────────────┤")
        for i, n in enumerate(neighbors[:3]):
            did_name = n.metadata.get("dataset_id", "Unknown")
            sim = dists[0][i]
            b_model = n.models[0] if n.models else "Unknown"
            hp = str(n.metadata.get("hparams", {})).replace("\n", "")[:28]
            print(f"│ {did_name:<10} │ {sim:<10.4f} │ {b_model:<12} │ {hp:<28} │")
        print("└────────────┴────────────┴──────────────┴──────────────────────────────┘")

    # 3. LLM Graceful Fallback
    from llm_suggester import get_llm_suggestions
    try:
        llm_models, llm_reasoning, llm_ok = get_llm_suggestions(meta_features, problem_type, str(did))
        llm_suggestion = llm_reasoning
    except Exception as e:
        llm_suggestion = f"LLM unavailable ({e}). Relying solely on memory retrieval."
        llm_models = []
        
    print(f"\n🤖 LLM SUGGESTION: {llm_suggestion}\n")
    print("⚙️ CONFIGURING PIPELINE...")
    
    # 4. Paradigm Routing
    profile = profile_dataset(did, X, y, problem_type)
    paradigm_decision, r_d_score, llm_score, memory_score, heuristics_score = route_paradigm(
        dataset_profile=profile,
        faiss_store=store,
        query_embedding=query_vec
    )
    
    w1_def, w2_def, w3_def = (0.8, 0.15, 0.05) if problem_type == 'regression' else (0.6, 0.3, 0.1)
    
    print(f"→ Routing Decision: {paradigm_decision} (R(D)={r_d_score:.2f})")
    print(f"→ Multi-Objective Weights: Accuracy={w1_def}, Speed={w2_def}, Complexity={w3_def}")
    
    if paradigm_decision == "AutoML":
        print("\n🚀 Executing Classical ML Pipeline...")
        preprocessor_cs, _, _ = build_preprocessor(X)
        
        full_search_best_model_name = "NONE"
        best_params = {}
        top_3_shap_features = []
        
        if validate:
            print(f"  [VALIDATION] Running full benchmark across all models...")
            all_models_full = get_models(problem_type)
            _, all_scores = baseline_screen(
                all_models_full, preprocessor_cs, X, y, problem_type,
                sample_frac=1.0, cv=3, random_state=42
            )
            if all_scores:
                from multi_objective import select_best_model_multiobjective
                best_model_by_utility, _ = select_best_model_multiobjective(all_scores, task_type=problem_type, w1=w1_def, w2=w2_def, w3=w3_def)
                full_score = all_scores[best_model_by_utility]['score']
                full_search_best_model_name = best_model_by_utility
                print(f"  [VALIDATION] Full Search Winner: {full_search_best_model_name} (Score: {full_score:.4f})")
                
        # HPO on top models
        top_models_hpo = best_retrieved_models
        print(f"  [HPO] Running HPO on FAISS Top Models: {top_models_hpo}")
        from hpo_optuna import run_hpo
        best_hpo_model, best_params = run_hpo(
            X, y, preprocessor_cs, top_models_hpo, warm_params, problem_type, str(did)
        )
        
        if best_hpo_model:
            print(f"  [HPO] Winner: {best_hpo_model} with params {best_params}")
            full_search_best_model_name = best_hpo_model
        
        # XAI SHAP Explanations
        if full_search_best_model_name not in ["NONE", "FAILED"]:
            from shap_explainer import generate_shap_explanations
            print(f"  [SHAP] Generating explanations for {full_search_best_model_name}...")
            try:
                final_model_instance = get_models(problem_type, [full_search_best_model_name])[full_search_best_model_name]
                X_prep_shap = preprocessor_cs.fit_transform(X, y)
                final_model_instance.fit(X_prep_shap, y)
                try:
                    feature_names = preprocessor_cs.get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(X_prep_shap.shape[1])]
                
                dense_X = X_prep_shap.toarray() if hasattr(X_prep_shap, 'toarray') else X_prep_shap
                X_train_df = pd.DataFrame(dense_X, columns=feature_names)
                
                success, top_3_shap_features = generate_shap_explanations(
                    model=final_model_instance, X_train=X_train_df, X_test=X_train_df, 
                    model_name=full_search_best_model_name, dataset_id=str(did)
                )
            except Exception as e:
                print(f"  [SHAP] Failed: {e}")
                
        # Phase 5.6: LLM Explainability Report
        try:
            from llm_explainer import generate_comprehensive_report
            master_context = {
                "paradigm_routing": {
                    "decision": paradigm_decision,
                    "R_D_score": round(float(r_d_score), 4),
                    "llm_signal": round(float(llm_score), 4),
                    "memory_signal": round(float(memory_score), 4),
                    "heuristic_signal": round(float(heuristics_score), 4)
                },
                "dataset_profile": profile,
                "training_and_hpo": {
                    "final_model": full_search_best_model_name,
                    "best_hpo_params": best_params,
                },
                "shap_interpretability": {
                    "top_3_features": top_3_shap_features,
                    "model_type": "Tree-based" if full_search_best_model_name in ['rf', 'gb', 'xgb_clf', 'xgb_reg', 'lgbm_clf', 'lgbm_reg', 'et_clf', 'et_reg'] else "Linear/Black-box"
                }
            }
            generate_comprehensive_report(master_context, str(did))
        except Exception as e:
            print(f"  [Phase 5.6 Report] Failed: {e}")
            
        # Store in FAISS memory mapping using MemoryStore
        if full_search_best_model_name not in ["NONE", "FAILED"]:
            print(f"\\n[Memory] Saving execution results to FAISS...")
            metadata = {
                "dataset_id": str(did),
                "problem_type": problem_type,
                "hparams": {full_search_best_model_name: best_params}
            }
            store.add(str(did), raw_vec, [full_search_best_model_name], metadata)
            
            # Rebuild index with 32D embeddings to keep it consistent
            from task_encoder import encode_all
            learned_vectors = encode_all(store, encoder)
            store.rebuild_index(learned_vectors)
            
            # Save to disk
            MEMORY_INDEX_PATH = "memory_store.faiss"
            MEMORY_META_PATH  = "memory_store.pkl"
            store.save_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
            print("✅ Run successfully added to FAISS Memory Store!")
            
    elif paradigm_decision == "AutoDL":
        print("\n🧠 Executing AutoDL NAS Pipeline...")
        try:
            from auto_dl_nas import objective_nas
            import torch
            import optuna
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  [AutoDL] Running NAS on {device}...")
            
            nas_prep, _, _ = build_preprocessor(X)
            X_nas_prep = nas_prep.fit_transform(X, y)
            X_train_numpy = X_nas_prep.toarray() if hasattr(X_nas_prep, 'toarray') else np.array(X_nas_prep)
            y_train_numpy = np.array(y)
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            nas_study = optuna.create_study(direction='maximize', study_name=f"nas_mlp_{did}")
            nas_study.optimize(
                lambda trial: objective_nas(trial, X_train_numpy, y_train_numpy, problem_type, device, w1_def, w2_def, w3_def), 
                n_trials=10 
            )
            print(f"  [AutoDL] Best DL Utility: {nas_study.best_value:.4f}")
            print(f"  [AutoDL] Best DL Architecture: {nas_study.best_params}")
            
        except Exception as e:
            print(f"  [AutoDL NAS] Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run Phase 4 Pipeline')
    parser.add_argument('--validate', action='store_true', help='Run full baseline screening for validation')
    args = parser.parse_args()

    if not os.path.exists("config.json"):
        from onboarding_agent import run_onboarding_cli
        csv_path = input("Enter path to dataset CSV: ")
        config = run_onboarding_cli(csv_path)
        if config and "csv_path" not in config:
            config["csv_path"] = csv_path
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
    else:
        config = json.load(open("config.json"))
        print(f"📂 Loaded existing config: {config.get('target_column', 'Unknown')} ({config.get('problem_type', 'Unknown')})")
        if "csv_path" not in config:
            csv_path = input("Enter path to dataset CSV (missing in config.json): ")
            config["csv_path"] = csv_path
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
    
    MEMORY_INDEX_PATH = "memory_store.faiss"
    MEMORY_META_PATH  = "memory_store.pkl"
    
    store = MemoryStore()
    if os.path.exists(MEMORY_INDEX_PATH):
        loaded = store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"✅ Loaded {loaded} existing records from disk")
    else:
        raise FileNotFoundError("Memory store not found. Run build_memory.py first.")
        
    print("🤖 Loading pre-trained Task Encoder...")
    from task_encoder import SiameseEncoder, TaskEncoderConfig, encode_all
    import torch
    cfg = TaskEncoderConfig(input_dim=10, hidden_dim=64, output_dim=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SiameseEncoder(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim).to(device)
    encoder_path = cfg.encoder_save_path
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Task encoder model not found at {encoder_path}. Run build_memory.py first.")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()
    
    # Rebuild index in memory using the 32D task encoder to match query dimensions
    learned_vectors = encode_all(store, encoder)
    store.rebuild_index(learned_vectors)
    
    # Check if this is a LOCAL dataset run
    if "csv_path" in config and config["csv_path"]:
        csv_path = config["csv_path"]
        target = config["target_column"]
        X, y, problem_type = load_local_dataset(csv_path, target)
        if X is not None:
            run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=csv_path, validate=args.validate)
            print("\n✅ Local dataset processing complete!")
        else:
            print("\n❌ Local dataset failed to load.")
    else:
        print("No valid 'csv_path' found in config.json. Pipeline handles local datasets only.")


if __name__ == "__main__":
    main()
