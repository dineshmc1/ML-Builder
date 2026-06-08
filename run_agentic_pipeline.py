import argparse
from agents.agent_orchestrator import AgenticAutoMLOrchestrator
import json

def main():
    parser = argparse.ArgumentParser(description="Run the Phase 6.3 Agentic AutoML Pipeline")
    parser.add_argument("csv_path", type=str, help="Path to the dataset CSV file")
    args = parser.parse_args()

    orchestrator = AgenticAutoMLOrchestrator()
    results = orchestrator.run_pipeline(args.csv_path)

    if results:
        print("\\n=== FINAL AGENTIC REPORT ===")
        print(json.dumps(results['report'], indent=2))
        
        final_plan_approved = results.get('critic_feedback', {}).get('approved', False)
        if final_plan_approved:
            print("\\n🚀 PLAN APPROVED. STARTING EXECUTION (Phase 4/5 Pipeline)...")
            
            # Load memory and encoder
            from cold_start import MemoryStore
            from task_encoder import load_encoder
            from data_loader import load_local_dataset
            from phase4_pipeline import run_single_dataset_pipeline
            
            csv_path = results['profile'].get('csv_path')
            target_column = results['profile'].get('target_column')
            
            X, y, problem_type = load_local_dataset(csv_path, target_column)
            if X is not None:
                store = MemoryStore()
                store.load_index("memory_store.faiss", "memory_store.pkl")
                try:
                    encoder = load_encoder("task_encoder.pt")
                except Exception as e:
                    print(f"Warning: Could not load encoder for execution: {e}")
                    encoder = None
                    
                run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=csv_path)
        else:
            print("\\n❌ PLAN REJECTED. Please refine requirements.")
    else:
        print("Pipeline failed to complete.")

if __name__ == "__main__":
    main()
