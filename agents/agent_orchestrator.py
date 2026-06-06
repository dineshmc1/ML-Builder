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

    def run_pipeline(self, csv_path: str):
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
            
            # Note: We should ideally encode this with Task Encoder to get 32D, but for this step we will
            # simulate passing it to the model agent.
            similar_datasets = []
            
            # Simple retrieval strategy for agent context
            distances, indices = self.memory.search(raw_vec, top_k=3)
            for idx in indices[0]:
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
