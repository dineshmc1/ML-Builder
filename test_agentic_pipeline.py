import sys
import os
import pandas as pd
import numpy as np

# Adjust imports to match the actual implementation from Phase 6.3
from agents.agent_orchestrator import AgenticAutoMLOrchestrator
from cold_start import MemoryStore
from task_encoder import encode_dataset
from dataset_embedding import compute_dataset_embedding

# Fallback dummy load_encoder if not explicitly defined in task_encoder.py
try:
    from task_encoder import load_encoder
except ImportError:
    def load_encoder(path):
        import torch
        from task_encoder import SiameseEncoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SiameseEncoder(input_dim=10, hidden_dim=64, output_dim=32).to(device)
        encoder.load_state_dict(torch.load(path, map_location=device))
        encoder.eval()
        return encoder

def test_agentic_pipeline(csv_path):
    """
    Test the complete agentic pipeline with a new dataset and verify memory store similarity.
    """
    print("=" * 80)
    print("TESTING PHASE 6.3: AGENTIC AUTOML PIPELINE")
    print("=" * 80)
    
    # 1. Load the dataset
    print(f"\\n[1] Loading dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns[:5])}...")
    except Exception as e:
        print(f"    ❌ Failed to load dataset: {e}")
        return False
    
    # We will assume 'income' as target for adult_clean.csv if the user doesn't pass one, 
    # or rely on the agent to detect it.
    
    # 2. Load memory store and encoder
    print("\\n[2] Loading FAISS memory store and task encoder...")
    memory_store = MemoryStore()
    if os.path.exists("memory_store.faiss"):
        memory_store.load_index("memory_store.faiss", "memory_store.pkl")
        print(f"    Loaded {len(memory_store.records)} records from memory")
    else:
        print("    ⚠️  Memory store not found. FAISS search will return empty.")
    
    try:
        encoder = load_encoder("task_encoder.pt")
        print("    Task encoder loaded")
    except Exception as e:
        print(f"    ⚠️  Failed to load task encoder: {e}")
        encoder = None
    
    # 3. Compute embedding and find similarity (if possible)
    avg_similarity = 0.0
    if encoder and len(memory_store.records) > 0:
        print("\\n[3] Computing dataset embedding and similarity scores...")
        # A rough heuristic to find the target column for the test script
        target_col = df.columns[-1]
        if 'income' in df.columns:
            target_col = 'income'
        elif 'target' in df.columns:
            target_col = 'target'
            
        print(f"    Using '{target_col}' as target for preliminary similarity check...")
        
        try:
            raw_embedding = compute_dataset_embedding(df.drop(columns=[target_col]), df[target_col])
            # task_encoder expects raw_vec to be passed into encode_dataset directly
            query_embedding = encode_dataset(raw_embedding, encoder).reshape(1, -1).astype(np.float32)
            
            # Search for similar datasets
            similarities, indices = memory_store._index.search(query_embedding, k=5)
            
            print(f"    Top 5 Similar Datasets:")
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:
                    record = memory_store.records[idx]
                    print(f"      {i+1}. Dataset {record.dataset_id}: similarity = {sim:.4f}")
                    print(f"         Best model: {record.models}")
                    print(f"         HParams: {list(record.metadata.get('hparams', {}).keys())}")
            
            avg_similarity = np.mean([s for s in similarities[0] if s > 0])
            print(f"\\n    Average similarity: {avg_similarity:.4f}")
            
            if avg_similarity < 0.5:
                print("    ⚠️  WARNING: Low similarity scores detected!")
                print("    This might indicate the memory store needs more diverse datasets.")
            else:
                print("    ✅ Good similarity scores!")
        except Exception as e:
            print(f"    ⚠️ Similarity check failed: {e}")
    else:
        print("\\n[3] Skipping similarity check (missing memory store or encoder).")
    
    # 4. Initialize agents
    print("\\n[4] Initializing agentic pipeline...")
    # Our implementation initializes the agents internally
    orchestrator = AgenticAutoMLOrchestrator()
    
    # 5. Run the agentic pipeline
    print("\\n[5] Running agentic pipeline...")
    try:
        # run_pipeline automatically handles data agent, business agent, feature agent, model agent, and critic
        result = orchestrator.run_pipeline(csv_path)
        
        if result:
            print("\\n    ✅ Pipeline completed successfully!")
            print(f"\\n    Results Summary:")
            print(f"      - Data Profile: target='{result['profile'].get('target_column')}', type='{result['profile'].get('problem_type')}'")
            print(f"      - Business Context: {result['requirements'].get('business_objective')}")
            print(f"      - Recommended Models: {result['models'].get('recommended_models')}")
            print(f"      - Critic Feedback: {result['critic_feedback'].get('feedback')}")
        else:
            print("\\n    ❌ Pipeline halted during execution.")
            return False
            
    except Exception as e:
        print(f"\\n    ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Generate notebook
    # The orchestrator already generated it during step 1 (data_agent)
    notebook_path = result.get('notebook', 'dataset_exploration.ipynb')
    
    # 7. Summary
    print("\\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Dataset: {csv_path}")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(f"Pipeline status: {'✅ SUCCESS' if result else '❌ FAILED'}")
    print(f"Notebook generated: {'✅ YES' if os.path.exists(notebook_path) else '❌ NO'} ({notebook_path})")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_agentic_pipeline.py <path_to_csv>")
        print("Example: python test_agentic_pipeline.py C:\\Dinesh\\Datasets\\adult_clean.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    test_agentic_pipeline(csv_path)
