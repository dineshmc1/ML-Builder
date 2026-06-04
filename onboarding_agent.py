import json
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from llm_explainer import litellm
try:
    from config import LLM_MODEL
except ImportError:
    LLM_MODEL = "openrouter/openai/gpt-4o-mini"


SYSTEM_PROMPT = """You are MetaAutoML's Data Onboarding Agent. 
A user has uploaded a dataset. Inspect the first 5 rows provided below.
Your goal is to determine:
1. The exact target column name.
2. The problem type (classification, regression, clustering).
3. Any obvious data quality issues (e.g., ID columns, timestamps).

Ask ONE question at a time. Do NOT hallucinate features. 
If the user says 'auto-detect', make your best guess and confirm.
Once confirmed, output ONLY a valid JSON object:
{"target_column": "...", "problem_type": "...", "notes": "..."}"""

def run_onboarding_cli(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return None

    try:
        # Load first 5 rows as string
        head_df = pd.read_csv(csv_path, nrows=5)
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        return None

    context = f"Dataset Preview:\n{head_df.to_string()}\n\nColumns: {list(head_df.columns)}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"I've uploaded '{csv_path}'. Here is the preview:\n{context}"}
    ]
    
    print("🤖 Onboarding Agent: Hello! I see your dataset. Let's configure the pipeline.")
    
    while True:
        try:
            response = litellm.completion(model=os.getenv("LLM_MODEL", LLM_MODEL), messages=messages, temperature=0.2)
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"🤖 Agent Error: LLM call failed - {e}")
            return None
        
        # Check if LLM returned JSON (parsing logic here)
        if reply.startswith("{") and reply.endswith("}"):
            try:
                config = json.loads(reply)
                config["csv_path"] = csv_path
                print(f"\n✅ Configuration Locked: {json.dumps(config, indent=2)}")
                with open("config.json", "w") as f: 
                    json.dump(config, f, indent=4)
                return config
            except json.JSONDecodeError:
                pass
            
        print(f"🤖 Agent: {reply}")
        user_input = input("👤 You: ")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_input})
