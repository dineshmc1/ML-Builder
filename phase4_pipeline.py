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

def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False, modality="tabular"):
    import numpy as np
    from wandb_logger import log
    
    did_safe = sanitize_filename(did)
    did = did_safe
    
    # 🚨 MULTI-MODAL OVERRIDE LOGIC 🚨
    if modality in ['vision', 'audio', 'text', 'video']:
        print("\n" + "="*50)
        print("🚨 MULTI-MODAL OVERRIDE DETECTED")
        print("="*50)
        print("🛑 Bypassing Tabular FAISS Memory (Incompatible Vector Space)")
        print("🛑 Bypassing Tabular R(D) Router")
        print("🧠 Forcing Paradigm Decision: AutoDL NAS")
        print("="*50 + "\n")
        
        paradigm_decision = "AutoDL"
        r_d_score = 1.0
        llm_score, memory_score, heuristics_score = 1.0, 0.0, 1.0
        warm_params = {}
        best_retrieved_models = []
        w1_def, w2_def, w3_def = (0.6, 0.3, 0.1)
        
    else:
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

            # ─── Warm-start NAS from SQLite DL history ───────────────────────
            try:
                from dl_memory import warm_start_from_memory
                prior_params = warm_start_from_memory(modality)
            except Exception:
                prior_params = None

            nas_study = optuna.create_study(direction='maximize', study_name=f"nas_mlp_{did}")
            if prior_params:
                nas_study.enqueue_trial(prior_params)
            nas_study.optimize(
                lambda trial: objective_nas(trial, X_train_numpy, y_train_numpy, problem_type, device, w1_def, w2_def, w3_def), 
                n_trials=10 
            )
            print(f"  [AutoDL] Best DL Utility: {nas_study.best_value:.4f}")
            best_dl_params = nas_study.best_params
            print(f"  [AutoDL] Best DL Architecture: {best_dl_params}")
            
            # ─────────────────────────────────────────────────────────────────
            # 🏆 FINAL PRODUCTION MODEL TRAINING
            # ─────────────────────────────────────────────────────────────────
            from auto_dl_nas import DynamicMLP
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from torch.utils.data import DataLoader, TensorDataset

            print("\n" + "="*50)
            print("🏆 TRAINING FINAL PRODUCTION MODEL")
            print("="*50)

            # Encode labels to contiguous integers
            le_final = LabelEncoder()
            y_encoded = le_final.fit_transform(np.array(y_train_numpy))
            class_names = [str(c) for c in le_final.classes_]
            is_clf = (problem_type == 'classification')
            out_dim = len(le_final.classes_) if is_clf else 1

            # Scale features
            scaler_final = StandardScaler()
            X_pca = scaler_final.fit_transform(X_train_numpy)

            # 80/20 stratified split
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_pca, y_encoded, test_size=0.2, random_state=42,
                stratify=y_encoded if is_clf else None
            )

            X_train_t = torch.tensor(X_train_f, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train_f, dtype=torch.long if is_clf else torch.float32).to(device)
            X_test_t  = torch.tensor(X_test_f,  dtype=torch.float32).to(device)
            y_test_t  = torch.tensor(y_test_f,  dtype=torch.long if is_clf else torch.float32).to(device)

            # Build final model with best NAS architecture
            final_model = DynamicMLP(
                input_dim=X_pca.shape[1],
                output_dim=out_dim,
                num_layers=best_dl_params['dl_num_layers'],
                hidden_dim=best_dl_params['dl_hidden_dim'],
                dropout=best_dl_params['dl_dropout'],
                is_classification=is_clf
            ).to(device)

            optimizer_f = torch.optim.Adam(final_model.parameters(), lr=best_dl_params['dl_lr'])
            criterion_f = torch.nn.CrossEntropyLoss() if is_clf else torch.nn.MSELoss()

            # Split out a validation set from the training fold for early stopping
            from sklearn.model_selection import train_test_split as _tts
            X_tr, X_val_f, y_tr, y_val_f = _tts(
                X_train_f, y_train_f, test_size=0.2, random_state=42,
                stratify=y_train_f if is_clf else None
            )
            X_tr_t  = torch.tensor(X_tr,    dtype=torch.float32).to(device)
            y_tr_t  = torch.tensor(y_tr,    dtype=torch.long if is_clf else torch.float32).to(device)
            X_val_t = torch.tensor(X_val_f, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val_f, dtype=torch.long if is_clf else torch.float32).to(device)

            train_loader_f = DataLoader(
                TensorDataset(X_tr_t, y_tr_t),
                batch_size=best_dl_params['dl_batch_size'],
                shuffle=True
            )

            # ── Early-Stopping Training (max 50 epochs, patience=5) ──────────
            final_epochs  = 50
            patience      = 5
            best_val_loss = float('inf')
            patience_ctr  = 0
            best_state    = None

            for epoch in range(final_epochs):
                # --- Train ---
                final_model.train()
                epoch_loss = 0.0
                for batch_x, batch_y in train_loader_f:
                    optimizer_f.zero_grad()
                    loss = criterion_f(final_model(batch_x), batch_y)
                    loss.backward()
                    optimizer_f.step()
                    epoch_loss += loss.item()

                # --- Validate ---
                final_model.eval()
                with torch.no_grad():
                    val_out  = final_model(X_val_t)
                    val_loss = criterion_f(val_out, y_val_t).item()

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{final_epochs} | "
                          f"Train Loss: {epoch_loss/len(train_loader_f):.4f} | "
                          f"Val Loss: {val_loss:.4f}")

                # --- Early stopping check ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
                    patience_ctr  = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        print(f"  ⏹ Early stopping triggered at epoch {epoch+1} "
                              f"(val loss hasn't improved for {patience} epochs).")
                        break

            # Restore best weights
            if best_state:
                final_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

            # Evaluate on held-out test set
            final_model.eval()
            with torch.no_grad():
                test_preds = final_model(X_test_t)
                if is_clf:
                    _, predicted_classes = torch.max(test_preds, 1)
                    y_pred_np = predicted_classes.cpu().numpy()
                else:
                    y_pred_np = test_preds.cpu().numpy().flatten()
                y_true_np = y_test_t.cpu().numpy()

            final_acc = accuracy_score(y_true_np, y_pred_np) if is_clf else None

            print(f"\n✅ FINAL TEST ACCURACY: {final_acc * 100:.2f}%" if is_clf else "")
            print("\n📊 CLASSIFICATION REPORT:")
            clf_report_str = classification_report(y_true_np, y_pred_np, target_names=class_names)
            print(clf_report_str)
            conf_matrix = confusion_matrix(y_true_np, y_pred_np)
            print("\n🔥 CONFUSION MATRIX:")
            print(conf_matrix)

            # ─── Save to SQLite DL Memory ────────────────────────────────────
            try:
                from dl_memory import save_dl_result
                save_dl_result(
                    dataset_name=str(did),
                    modality=modality,
                    best_params=best_dl_params,
                    final_accuracy=float(final_acc) if final_acc is not None else 0.0
                )
            except Exception as mem_err:
                print(f"  [DL Memory] Failed to save: {mem_err}")

            # ─── LLM Consultant Report ───────────────────────────────────────
            dl_context = {
                "paradigm_routing": {
                    "decision": "AutoDL",
                    "R_D_score": r_d_score,
                    "modality": modality,
                    "extractor_used": "CLIP" if modality in ["vision", "video"] else
                                      "SentenceTransformer" if modality == "text" else
                                      "Librosa MFCC"
                },
                "dataset": {
                    "id": did,
                    "n_samples": int(X_pca.shape[0]),
                    "n_features_after_pca": int(X_pca.shape[1]),
                    "n_classes": int(out_dim),
                    "class_names": class_names
                },
                "nas_results": {
                    "best_architecture": best_dl_params,
                    "best_dl_utility": round(float(nas_study.best_value), 4)
                },
                "final_performance": {
                    "test_accuracy": round(float(final_acc), 4) if final_acc is not None else None,
                    "classification_report": clf_report_str,
                    "confusion_matrix": conf_matrix.tolist()
                }
            }

            try:
                from llm_explainer import generate_comprehensive_report
                generate_comprehensive_report(dl_context, did)
            except Exception as report_err:
                print(f"  [LLM Report] Failed to generate AutoDL report: {report_err}")
            
        except Exception as e:
            print(f"  [AutoDL NAS] Failed: {e}")
            import traceback; traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Run Phase 4 Pipeline')
    parser.add_argument('--validate', action='store_true', help='Run full baseline screening for validation')
    args = parser.parse_args()

    MEMORY_INDEX_PATH = "memory_store.faiss"
    MEMORY_META_PATH  = "memory_store.pkl"
    
    store = MemoryStore()
    if os.path.exists(MEMORY_INDEX_PATH):
        loaded = store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"✅ Loaded {loaded} existing records from disk")
    else:
        print("⚠️ Memory store not found. Creating a new empty store.")
        
    print("🤖 Loading pre-trained Task Encoder...")
    from task_encoder import SiameseEncoder, TaskEncoderConfig
    import torch
    cfg = TaskEncoderConfig(input_dim=10, hidden_dim=64, output_dim=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SiameseEncoder(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim).to(device)
    encoder_path = cfg.encoder_save_path
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        encoder.eval()
    else:
        print(f"⚠️ Task encoder model not found at {encoder_path}. Using un-trained encoder.")
        encoder.eval()

    print("\n" + "="*50)
    print("META-AUTOML UNIVERSAL INGESTION")
    print("="*50)
    
    user_input = input("Enter path to dataset (CSV file OR Image/Audio/Text Folder): ").strip()
    
    # ==========================================
    # PATH A: MULTI-MODAL (FOLDERS)
    # ==========================================
    if os.path.isdir(user_input):
        print(f"\n📂 Detected Directory: {user_input}")
        print("🚀 Bypassing Agentic DataAgent. Routing to Multi-Modal Embedder...")
        
        from modality_router import ModalityRouter
        from multimodal_extractor import UniversalEmbedder
        
        router = ModalityRouter(user_input)
        modality = router.get_modality()
        print(f"🔍 Modality Detected: {modality.upper()}")
        
        if modality in ['vision', 'text', 'audio', 'video']:
            embedder = UniversalEmbedder(device=device, batch_size=32)
            X, y = embedder.embed_directory(user_input, modality)
            
            print(f"✅ Embedding Extraction Complete! Shape: {X.shape}")
            
            problem_type = 'classification'
            run_single_dataset_pipeline(
                X, y, problem_type, store, encoder, 
                did=os.path.basename(user_input), 
                validate=args.validate,
                modality=modality
            )
            return
        else:
            print("❌ Unknown modality. Please check folder contents.")
            return

    # ==========================================
    # PATH B: TABULAR (CSV FILES)
    # ==========================================
    elif os.path.isfile(user_input) and user_input.endswith('.csv'):
        print(f"\n📄 Detected CSV File: {user_input}")
        
        try:
            use_agentic = input("Use Agentic AutoML pipeline? (y/n): ").strip().lower() == 'y'
        except EOFError:
            use_agentic = False
            
        if use_agentic:
            print("\n" + "="*80)
            print("PHASE 6.3: AGENTIC AUTOML PIPELINE")
            print("="*80)
            
            from agents.agent_orchestrator import AgenticAutoMLOrchestrator
            orchestrator = AgenticAutoMLOrchestrator()
            result = orchestrator.run_pipeline(user_input, force_run=True)
            
            if result:
                print(f"\n✅ Agentic pipeline complete!")
                print(f"📓 Notebook saved to: {result.get('notebook')}")
                print("\n🚀 Proceeding to Execute Auto ML/DL Pipeline based on Agentic Plan...")
                
                target_column = result['profile'].get('target_column')
                X, y, problem_type = load_local_dataset(user_input, target_column)
                if X is not None:
                    run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=user_input, validate=args.validate)
            else:
                print("🛑 Agentic pipeline halted. Exiting.")
                return
        else:
            from onboarding_agent import run_onboarding_cli
            config = run_onboarding_cli(user_input)
            if config:
                target_column = config.get("target_column")
                X, y, problem_type = load_local_dataset(user_input, target_column)
                if X is not None:
                    run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=user_input, validate=args.validate)
            
    else:
        print("❌ Error: Invalid path. Please provide a valid CSV file or a dataset folder.")
        return

if __name__ == "__main__":
    main()
