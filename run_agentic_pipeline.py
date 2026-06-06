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
        print("\\nPipeline is ready for execution. Next step: Pass the 'features' and 'models' dictionary to phase4_pipeline.py!")
    else:
        print("Pipeline failed to complete.")

if __name__ == "__main__":
    main()
