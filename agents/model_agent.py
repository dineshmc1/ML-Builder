import json
import os
import pandas as pd
from litellm import completion

class ModelSelectionAgent:
    def __init__(self, llm_model="openrouter/openai/gpt-4o-mini"):
        self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def recommend_models(self, profile: dict, ml_objectives: dict, similar_datasets: list) -> list:
        print("[ModelAgent] Recommending models based on memory and constraints...")
        
        # Similar datasets are retrieved from FAISS
        memory_context = ""
        for sd in similar_datasets:
            memory_context += f"- Dataset {sd['id']} used {sd['best_model']} (Score: {sd['score']:.4f})\\n"

        system_prompt = """You are the Model Selection Agent.
Given the dataset profile, ML objectives, and historical similar datasets from FAISS memory, recommend the top 3 best models.
Ensure your recommendations respect the constraints (e.g. if interpretability is high, favor tree-based/linear).

Output MUST be a valid JSON object:
{
  "recommended_models": ["model1", "model2", "model3"],
  "reasoning": "..."
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile: {json.dumps(profile)}\\nObjectives: {json.dumps(ml_objectives)}\\nMemory Context:\\n{memory_context}"}
        ]

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            recommendations = json.loads(raw_content)
            print(f"[ModelAgent] Top recommendations: {recommendations.get('recommended_models')}")
            return recommendations
            
        except Exception as e:
            print(f"[ModelAgent] LLM parsing failed: {e}")
            return {
                "recommended_models": ["rf", "gb", "logistic" if profile.get('problem_type') == 'classification' else "ridge"],
                "reasoning": "Fallback to robust baselines."
            }
