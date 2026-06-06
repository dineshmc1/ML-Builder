import json
import os
from litellm import completion

class BusinessContextAgent:
    def __init__(self, llm_model="openrouter/openai/gpt-4o-mini"):
        self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def gather_requirements(self) -> dict:
        print("\\n[BusinessAgent] Let's align on your business objectives.")
        print("Please answer the following questions (or press Enter to skip/auto-fill):")
        
        objective = input("1. What is the primary business objective? (e.g., 'Predict customer churn to offer discounts'): ").strip()
        metrics = input("2. What is the primary success metric? (e.g., 'F1-score', 'RMSE', 'Accuracy'): ").strip()
        constraints = input("3. Are there constraints? (e.g., 'Low latency required', 'Must be highly interpretable'): ").strip()
        
        if not objective: objective = "Build an accurate predictive model for the target variable."
        if not metrics: metrics = "Auto-select based on problem type."
        if not constraints: constraints = "None specified. Balance performance and speed."

        reqs = {
            "business_objective": objective,
            "success_metrics": metrics,
            "constraints": constraints
        }
        return reqs

    def translate_to_ml_objectives(self, requirements: dict) -> dict:
        print("[BusinessAgent] Translating requirements into ML Objectives...")
        
        system_prompt = """You are the Business Context Agent.
Translate the following business requirements into technical ML objectives.
Focus on:
1. 'optimization_priority': e.g., 'recall', 'precision', 'rmse', 'speed'
2. 'model_constraints': e.g., ['linear_only', 'tree_based_ok', 'no_neural_nets']
3. 'risk_tolerance': 'low', 'medium', 'high'

Output MUST be a valid JSON object:
{"optimization_priority": "...", "model_constraints": ["..."], "risk_tolerance": "..."}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Business Requirements:\\n{json.dumps(requirements, indent=2)}"}
        ]

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            ml_objectives = json.loads(raw_content)
            print(f"[BusinessAgent] ML Priorities established: Optimize for {ml_objectives.get('optimization_priority')}")
            return ml_objectives
            
        except Exception as e:
            print(f"[BusinessAgent] LLM parsing failed: {e}")
            return {
                "optimization_priority": "accuracy",
                "model_constraints": [],
                "risk_tolerance": "medium"
            }
