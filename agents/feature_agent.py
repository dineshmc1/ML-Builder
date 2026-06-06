import json
import os
from litellm import completion

class FeatureEngineeringAgent:
    def __init__(self, llm_model="openrouter/openai/gpt-4o-mini"):
        self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def analyze_features(self, profile: dict) -> dict:
        print("[FeatureAgent] Analyzing dataset profile to recommend feature engineering strategies...")
        
        system_prompt = """You are the Feature Engineering Agent.
Review the dataset profile. Based on the target column, problem type, and shape, suggest a feature engineering plan.
Output MUST be a valid JSON object:
{
  "missing_values_strategy": "...",
  "categorical_encoding": "...",
  "suggested_transformations": ["...", "..."],
  "feature_selection": "..."
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Dataset Profile:\\n{json.dumps(profile, indent=2)}"}
        ]

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            plan = json.loads(raw_content)
            print(f"[FeatureAgent] Recommended Missing Value Strategy: {plan.get('missing_values_strategy')}")
            return plan
            
        except Exception as e:
            print(f"[FeatureAgent] LLM parsing failed: {e}")
            return {
                "missing_values_strategy": "mean_imputation",
                "categorical_encoding": "one_hot",
                "suggested_transformations": ["standard_scaling"],
                "feature_selection": "none"
            }

    def create_features(self, profile: dict) -> dict:
        # In a real implementation, this would execute pandas/sklearn code.
        # For now, it delegates the plan generation.
        plan = self.analyze_features(profile)
        return plan
