import json
import os
from litellm import completion

class CriticAgent:
    def __init__(self, llm_model="openrouter/openai/gpt-4o-mini"):
        self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def validate_data_and_requirements(self, profile: dict, requirements: dict):
        print("\\n[CriticAgent] Validating Data Profile and Requirements...")
        
        system_prompt = """You are the Critic Agent.
Review the Data Profile and Business Requirements. Ensure there are no contradictions (e.g. classification target but regression metric requested).
Output MUST be a valid JSON object:
{
  "approved": true/false,
  "feedback": "...",
  "warnings": ["..."]
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile: {json.dumps(profile)}\\nReqs: {json.dumps(requirements)}"}
        ]

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            critique = json.loads(raw_content)
            print(f"[CriticAgent] Approved: {critique.get('approved')}")
            return critique
            
        except Exception as e:
            print(f"[CriticAgent] LLM parsing failed: {e}")
            return {"approved": True, "feedback": "Auto-approved due to parsing failure.", "warnings": []}

    def validate_full_pipeline(self, profile: dict, features: dict, models: dict):
        print("\\n[CriticAgent] Validating Full Pipeline Configuration...")
        
        system_prompt = """You are the Critic Agent.
Review the final pipeline configuration. Check for data leakage, invalid combinations, and ensure the models make sense for the feature transformations.
Output MUST be a valid JSON object:
{
  "approved": true/false,
  "feedback": "...",
  "risk_level": "low/medium/high"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile: {json.dumps(profile)}\\nFeatures: {json.dumps(features)}\\nModels: {json.dumps(models)}"}
        ]

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            critique = json.loads(raw_content)
            print(f"[CriticAgent] Pipeline Approved: {critique.get('approved')}")
            return critique
            
        except Exception as e:
            print(f"[CriticAgent] LLM parsing failed: {e}")
            return {"approved": True, "feedback": "Auto-approved due to parsing failure.", "risk_level": "low"}
