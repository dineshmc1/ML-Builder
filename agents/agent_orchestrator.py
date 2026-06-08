import json
from agents.data_agent import DataUnderstandingAgent
from agents.business_agent import BusinessContextAgent
from agents.feature_agent import FeatureEngineeringAgent
from agents.model_agent import ModelSelectionAgent
from agents.critic_agent import CriticAgent
from cold_start import MemoryStore
import os

class AgenticAutoMLOrchestrator:
    def __init__(self):
        print("Initializing Agentic AutoML Orchestrator...")
        self.data_agent = DataUnderstandingAgent()
        self.business_agent = BusinessContextAgent()
        self.feature_agent = FeatureEngineeringAgent()
        self.model_agent = ModelSelectionAgent()
        self.critic_agent = CriticAgent()
        
        self.memory = MemoryStore()
        # Initialize memory only if it exists, otherwise it will be empty
        if os.path.exists("memory_store.faiss"):
            self.memory.load_index("memory_store.faiss", "memory_store.pkl")

    def generate_consultant_report(self, profile, requirements, features, models):
        # A simple aggregation for now. Real implementation could use LLM to format this beautifully.
        report = {
            "Executive Summary": f"AutoML Plan for {profile.get('csv_path')}",
            "Business Goals": requirements,
            "Data Profile": profile,
            "Feature Engineering Strategy": features,
            "Recommended Models": models
        }
        return report

    def run_pipeline(self, csv_path: str, force_run: bool = False):
        print(f"\\n{'='*50}\\n🚀 STARTING AGENTIC AUTOML PIPELINE\\n{'='*50}")
        
        # Step 1: Data Understanding
        profile = self.data_agent.analyze_dataset(csv_path)
        if not profile:
            print("❌ Pipeline halted: Failed to understand dataset.")
            return None
            
        notebook_path = self.data_agent.generate_notebook(profile)
        
        # Step 2: Business Context
        requirements = self.business_agent.gather_requirements()
        ml_objectives = self.business_agent.translate_to_ml_objectives(requirements)
        
        # Step 3: Critic Review - Phase 1
        critique1 = self.critic_agent.validate_data_and_requirements(profile, requirements)
        if not critique1.get("approved"):
            print(f"❌ Pipeline halted by Critic Agent: {critique1.get('feedback')}")
            return None
            
        # Step 4: Feature Engineering
        features = self.feature_agent.create_features(profile)
        
        # Step 5: Model Selection
        # Search FAISS memory for similar datasets
        try:
            from dataset_embedding import compute_dataset_embedding
            import pandas as pd
            df = pd.read_csv(csv_path)
            y = df[profile.get('target_column')]
            X = df.drop(columns=[profile.get('target_column')])
            raw_vec = compute_dataset_embedding(X, y)
            
            # We need to encode the 10D raw vector into 32D using the Task Encoder
            from task_encoder import encode_dataset, load_encoder
            import numpy as np
            try:
                encoder = load_encoder("task_encoder.pt")
                query_embedding = encode_dataset(raw_vec, encoder).reshape(1, -1).astype(np.float32)
            except Exception as e:
                print(f"[Orchestrator] Failed to load encoder, falling back to empty memory: {e}")
                query_embedding = None
            
            similar_datasets = []
            if query_embedding is not None and self.memory._index is not None:
                dists, idxs = self.memory._index.search(query_embedding, 3)
                for idx in idxs[0]:
                    if idx != -1:
                        r = self.memory.records[idx]
                        similar_datasets.append({
                            "id": r.dataset_id,
                            "best_model": r.models[0] if r.models else "unknown",
                            "score": 0.0 # Placeholder
                        })
        except Exception as e:
            print(f"[Orchestrator] Memory search failed: {e}")
            similar_datasets = []

        model_recommendations = self.model_agent.recommend_models(profile, ml_objectives, similar_datasets)
        
        # Step 6: Critic Review - Phase 2
        final_critique = self.critic_agent.validate_full_pipeline(profile, features, model_recommendations)
        if not final_critique.get("approved"):
            print(f"⚠️ Warning from Critic Agent: {final_critique.get('feedback')}")
            if not force_run:
                print("🛑 Pipeline halted by Critic Agent. Fix the issues or use force_run=True.")
                return None
            else:
                print(" 🚀 FORCE RUN enabled. Bypassing Critic rejection and starting training...")
                final_critique['approved'] = True  # Override so the downstream trigger runs
        # Step 7: Generate Report
        report = self.generate_consultant_report(profile, requirements, features, model_recommendations)
        
        print(f"\\n{'='*50}\\n✅ AGENTIC PIPELINE COMPLETE\\n{'='*50}")
        print(f"- EDA Notebook generated: {notebook_path}")
        print(f"- Recommended Models: {model_recommendations.get('recommended_models')}")
        
        return {
            'notebook': notebook_path,
            'profile': profile,
            'requirements': requirements,
            'features': features,
            'models': model_recommendations,
            'report': report,
            'critic_feedback': final_critique
        }
