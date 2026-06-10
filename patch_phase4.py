import re

with open('phase4_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update run_single_dataset_pipeline signature
old_sig = 'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False, modality="tabular"):'
new_sig = 'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False, modality="tabular", config=None):'
code = code.replace(old_sig, new_sig)

# 2. Add Notebook Generator to AutoDL NAS end
autodl_end_marker = """            except Exception as report_err:
                print(f"  [LLM Report] Failed to generate AutoDL report: {report_err}")"""

autodl_notebook_addition = """            except Exception as report_err:
                print(f"  [LLM Report] Failed to generate AutoDL report: {report_err}")

            # ─── Advanced Notebook Generator ─────────────────────────────────
            try:
                from notebook_generator import generate_advanced_notebook
                results_dict = {
                    "X": X_test_f, "y": y_test_f, "y_test": y_true_np, "y_pred": y_pred_np,
                    "final_accuracy": final_acc,
                    "paradigm": "AutoDL",
                    "modality": modality
                }
                nb_path = f"reports/{did}_advanced_analysis.ipynb"
                if config:
                    generate_advanced_notebook(config, results_dict, nb_path)
            except Exception as nb_err:
                print(f"  [Notebook Generator] Failed: {nb_err}")
"""
code = code.replace(autodl_end_marker, autodl_notebook_addition)

# 3. Add Notebook Generator to AutoML end
automl_end_marker = """            print("✅ Run successfully added to FAISS Memory Store!")"""

automl_notebook_addition = """            print("✅ Run successfully added to FAISS Memory Store!")

            # ─── Advanced Notebook Generator ─────────────────────────────────
            try:
                from notebook_generator import generate_advanced_notebook
                if full_search_best_model_name not in ["NONE", "FAILED"]:
                    results_dict = {
                        "X": X, "y": y, "y_test": None, "y_pred": None, # Test metrics aren't neatly localized here without re-evaluating
                        "final_accuracy": best_params.get("utility_score", 0),
                        "paradigm": "AutoML",
                        "modality": "tabular"
                    }
                    nb_path = f"reports/{did}_advanced_analysis.ipynb"
                    if config:
                        generate_advanced_notebook(config, results_dict, nb_path)
            except Exception as nb_err:
                print(f"  [Notebook Generator] Failed: {nb_err}")
"""
code = code.replace(automl_end_marker, automl_notebook_addition)

# 4. Rewrite main() data ingestion
main_start = code.find('    user_input = input("Enter path to dataset')
main_end = code.find('def __init__(self', main_start)
if main_end == -1:
    main_end = len(code)

# We want to replace from user_input up to the end of main()
new_main_block = """    user_input = input("Enter path to dataset (CSV file OR Image/Audio/Text Folder): ").strip()
    
    from onboarding_agent import OnboardingAgent
    agent = OnboardingAgent()
    config = agent.run(user_input)
    
    if not config:
        print("❌ Failed to process input. Exiting.")
        return
        
    modality = config['modality']
    
    # ==========================================
    # PATH A: MULTI-MODAL (FOLDERS)
    # ==========================================
    if modality in ['vision', 'text', 'audio', 'video']:
        print("🚀 Bypassing Agentic DataAgent. Routing to Multi-Modal Embedder...")
        
        from multimodal_extractor import UniversalEmbedder
        
        embedder = UniversalEmbedder(device=device, batch_size=32, domain=config['domain'])
        X, y = embedder.embed_directory(user_input, modality)
        
        print(f"✅ Embedding Extraction Complete! Shape: {X.shape}")
        
        problem_type = 'classification'
        run_single_dataset_pipeline(
            X, y, problem_type, store, encoder, 
            did=os.path.basename(user_input), 
            validate=args.validate,
            modality=modality,
            config=config
        )
        return

    # ==========================================
    # PATH B: TABULAR (CSV FILES)
    # ==========================================
    elif modality == 'tabular':
        try:
            use_agentic = input("Use Agentic AutoML pipeline? (y/n): ").strip().lower() == 'y'
        except EOFError:
            use_agentic = False
            
        if use_agentic:
            print("\\n" + "="*80)
            print("PHASE 6.3: AGENTIC AUTOML PIPELINE")
            print("="*80)
            
            from agents.agent_orchestrator import AgenticAutoMLOrchestrator
            orchestrator = AgenticAutoMLOrchestrator()
            result = orchestrator.run_pipeline(user_input, force_run=True)
            
            if result:
                print(f"\\n✅ Agentic pipeline complete!")
                print(f"📓 Notebook saved to: {result.get('notebook')}")
                print("\\n🚀 Proceeding to Execute Auto ML/DL Pipeline based on Agentic Plan...")
                
                target_column = result['profile'].get('target_column')
                X, y, problem_type = load_local_dataset(user_input, target_column)
                if X is not None:
                    run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=os.path.basename(user_input), validate=args.validate, config=config)
            else:
                print("🛑 Agentic pipeline halted. Exiting.")
                return
        else:
            target_column = config.get("target_column")
            X, y, problem_type = load_local_dataset(user_input, target_column)
            if X is not None:
                run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=os.path.basename(user_input), validate=args.validate, config=config)

if __name__ == "__main__":
    main()
"""

# Find the start of `    user_input = input("Enter path to dataset` and replace everything down to `if __name__ == "__main__":`
main_start_idx = code.find('    user_input = input("Enter path to dataset (CSV file OR Image/Audio/Text Folder): ").strip()')
if main_start_idx != -1:
    code = code[:main_start_idx] + new_main_block

with open('phase4_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Patch applied to phase4_pipeline.py")
