import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# We need to import litellm explicitly or use the one from llm_explainer
try:
    from litellm import completion
except ImportError:
    from llm_explainer import litellm
    completion = litellm.completion

class OnboardingAgent:
    def __init__(self, llm_model="openai/gpt-4o-mini"):
        try:
            from config import LLM_MODEL
            self.llm_model = os.getenv("LLM_MODEL", LLM_MODEL)
        except ImportError:
            self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def run(self, data_path: str) -> dict:
        print("🤖 Onboarding Agent: Initializing Data Intake Protocol...")
        
        # 1. Quick structural scan
        is_folder = os.path.isdir(data_path)
        modality = "tabular"
        if is_folder:
            exts = set()
            for root, _, files in os.walk(data_path):
                for f in files:
                    exts.add(os.path.splitext(f)[1].lower())
            if exts.intersection({'.jpg','.png','.jpeg'}): modality = "vision"
            elif exts.intersection({'.wav','.mp3'}): modality = "audio"
            elif exts.intersection({'.txt','.json'}): modality = "text"
            elif exts.intersection({'.mp4','.avi'}): modality = "video"
            
        print(f"🔍 Detected Modality: {modality.upper()} | Source: {'Folder' if is_folder else 'File'}")

        # 2. LLM Prompt for Business & Domain Context
        system_prompt = f"""You are a Senior Data Analyst & MLOps Engineer. 
        A user is uploading a {modality} dataset. Ask 3 strategic questions to understand:
        1. Business Objective & Success Metric
        2. Domain Context (If {modality} != 'tabular', ask: "Which domain? general, biology, remote_sensing, documents")
        3. Constraints (Latency, Interpretability, Fairness)
        
        Output ONLY valid JSON:
        {{"business_objective": "...", "success_metric": "...", "domain": "...", "constraints": "..."}}"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Auto-fill tabular defaults, ask for multi-modal specifics
        if modality == "tabular":
            print(f"👤 Please provide business context for tabular dataset '{data_path}'.")
            user_input = input("👤 You (Business Goal/Metric/Constraints): ").strip()
            messages.append({"role": "user", "content": user_input})
        else:
            print(f"👤 Please provide business context for {modality} dataset '{data_path}'.")
            user_input = input("👤 You (Business Goal, Domain, Constraints): ").strip()
            messages.append({"role": "user", "content": f"Dataset type: {modality}. User input: {user_input}"})

        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.2)
            
            # Simple JSON extraction in case there's markdown
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            config = json.loads(content.strip())
        except Exception as e:
            print(f"⚠️ LLM parsing failed. Using safe defaults. ({e})")
            config = {"business_objective": "Classification/Regression", "success_metric": "Accuracy/F1", "domain": "general", "constraints": "None"}

        # If it's tabular, we also need to detect the target column if not provided
        if modality == "tabular":
            try:
                head_df = pd.read_csv(data_path, nrows=5)
                cols = list(head_df.columns)
                # Simple guess if target column not provided
                if "target_column" not in config:
                    target_guess = cols[-1] if len(cols) > 0 else "target"
                    config["target_column"] = input(f"🎯 Target column [Default: {target_guess}]: ").strip() or target_guess
            except Exception:
                pass

        # 3. Merge with structural data
        final_config = {
            "data_path": data_path,
            "modality": modality,
            "domain": config.get("domain", "general").lower(),
            "business_context": config
        }
        
        print(f"✅ Configuration Locked: {json.dumps(final_config, indent=2)}")
        return final_config
