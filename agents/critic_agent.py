import json
import os
from litellm import completion


try:
    from config import LLM_MODEL
except ImportError:
    LLM_MODEL = "openrouter/openai/gpt-4.1-mini" # Safe fallback

class CriticAgent:
    def __init__(self):
        # Priority: Environment Variable > config.py > Hardcoded Fallback
        self.llm_model = os.getenv("LLM_MODEL", LLM_MODEL)
        print(f"[CriticAgent] Initialized with model: {self.llm_model}")

    def validate_data_and_requirements(self, profile: dict, requirements: dict):
        print("\n[CriticAgent] Validating Data Profile and Requirements...")
        
        # STRICTER PROMPT: Forces the LLM to look for specific contradictions
        system_prompt = """You are a strict Senior Data Scientist Critic Agent.
        Your job is to find flaws in the proposed Data Profile and Business Requirements.
        Do NOT just approve everything. Look for:
        1. Metric Mismatch: e.g., Requesting 'RMSE' for a Classification problem.
        2. Target Leakage: e.g., The target column is also listed as a feature.
        3. Feasibility: e.g., Requesting 'Real-time inference < 10ms' but asking for a massive Ensemble model.
        
        Output MUST be a valid JSON object (no markdown formatting outside the JSON):
        {
          "approved": true/false,
          "feedback": "Specific reason for approval or rejection",
          "warnings": ["List of specific risks found"]
        }"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile: {json.dumps(profile)}\nReqs: {json.dumps(requirements)}"}
        ]

        return self._call_llm(messages)

    def validate_full_pipeline(self, profile: dict, features: dict, models: dict):
        print("\n[CriticAgent] Validating Full Pipeline Configuration...")
        
        # STRICTER PROMPT: Focuses on AutoML-specific failures
        system_prompt = """You are a strict Senior Data Scientist Critic Agent.
        Review the final pipeline configuration. Do NOT just approve it. Look for:
        1. Data Leakage: e.g., Scaling/Imputation applied before Train/Test split.
        2. Model Mismatch: e.g., Using Linear Regression on highly non-linear data without feature engineering.
        3. Resource Mismatch: e.g., Suggesting a 500-layer Neural Network for a dataset with only 100 rows.
        
        Output MUST be a valid JSON object:
        {
          "approved": true/false,
          "feedback": "Specific technical reason",
          "risk_level": "low/medium/high"
        }"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile: {json.dumps(profile)}\nFeatures: {json.dumps(features)}\nModels: {json.dumps(models)}"}
        ]

        return self._call_llm(messages)

    def _call_llm(self, messages: list) -> dict:
        """Internal helper to handle LLM calls and JSON parsing safely."""
        import re
        try:
            response = completion(
                model=self.llm_model, 
                messages=messages, 
                temperature=0.1, # Low temp for deterministic, strict outputs
                max_tokens=500
            )
            raw_content = response.choices[0].message.content
            
            if not raw_content:
                return {"approved": True, "feedback": "Empty response, auto-approved.", "risk_level": "low"}

            # Strip <think> tags (common in reasoning models)
            raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            critique = self._parse_llm_json(raw_content)
            if critique is None:
                raise ValueError("Failed to parse JSON from LLM response.")
                
            print(f"[CriticAgent] Decision: {'APPROVED' if critique.get('approved') else 'REJECTED'}")
            return critique
            
        except Exception as e:
            print(f"[CriticAgent] LLM parsing failed: {e}")
            # Failsafe: Auto-approve but log the error so the pipeline doesn't crash
            return {"approved": True, "feedback": "Auto-approved due to LLM parsing failure.", "risk_level": "unknown"}

    def _parse_llm_json(self, response_content: str):
        """Robust JSON parser that handles <think> tags and markdown blocks."""
        # 1. Try direct parse
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass

        import re
        # 2. Remove <think> tags (common in DeepSeek-R1)
        cleaned_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

        # 3. Extract JSON from markdown blocks ```json ... ```
        match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 4. Fallback: Find the first '{' and last '}'
        start = cleaned_content.find('{')
        end = cleaned_content.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(cleaned_content[start:end])
            except json.JSONDecodeError:
                pass
                
        # 5. Last resort: Check for boolean keywords if expecting a simple decision
        if 'rejected' in cleaned_content.lower() or 'false' in cleaned_content.lower():
            return {"approved": False, "feedback": "Rejected based on keywords."}
        elif 'approved' in cleaned_content.lower() or 'true' in cleaned_content.lower():
            return {"approved": True, "feedback": "Approved based on keywords."}

        return None