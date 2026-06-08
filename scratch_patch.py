import os
import re

with open('phase4_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Replace FAISS save
old_faiss = """        # Store in FAISS memory mapping using MemoryStore
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
            print("✅ Run successfully added to FAISS Memory Store!")"""

new_faiss = """        # Store in FAISS memory mapping using MemoryStore
        # RESEARCH DECISION: We DO NOT save local user datasets into the FAISS index
        # to avoid polluting the pre-trained OpenML meta-learning embeddings.
        if full_search_best_model_name not in ["NONE", "FAILED"]:
            print(f"\\n[Memory] Run successful. Skipping FAISS save to protect OpenML memory store.")"""

code = code.replace(old_faiss, new_faiss)

# 2. Replace main()
# Find where main starts
main_match = re.search(r'def main\(\):.*', code, re.DOTALL)
if main_match:
    main_start_idx = main_match.start()
    
    new_main = """def main():
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

    print("\\n" + "="*50)
    print("META-AUTOML UNIVERSAL INGESTION")
    print("="*50)
    
    user_input = input("Enter path to dataset (CSV file OR Image/Audio/Text Folder): ").strip()
    
    # ==========================================
    # PATH A: MULTI-MODAL (FOLDERS)
    # ==========================================
    if os.path.isdir(user_input):
        print(f"\\n📂 Detected Directory: {user_input}")
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
                validate=args.validate
            )
            return
        else:
            print("❌ Unknown modality. Please check folder contents.")
            return

    # ==========================================
    # PATH B: TABULAR (CSV FILES)
    # ==========================================
    elif os.path.isfile(user_input) and user_input.endswith('.csv'):
        print(f"\\n📄 Detected CSV File: {user_input}")
        
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
"""
    code = code[:main_start_idx] + new_main
    
    with open('phase4_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched successfully!")
else:
    print("Could not find main()")
