import pandas as pd
import json
import os
from litellm import completion
from agents.notebook_generator import create_eda_notebook

class DataUnderstandingAgent:
    def __init__(self, llm_model="openrouter/openai/gpt-4o-mini"):
        self.llm_model = os.getenv("LLM_MODEL", llm_model)

    def analyze_dataset(self, csv_path: str) -> dict:
        print(f"\\n[DataAgent] Analyzing dataset: {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            head_df = df.head()
        except Exception as e:
            print(f"[DataAgent] Error reading '{csv_path}': {e}")
            return {}

        context = f"Dataset Shape: {df.shape}\\nColumns: {list(df.columns)}\\nPreview:\\n{head_df.to_string()}"
        
        system_prompt = """You are the Data Understanding Agent. 
Analyze the dataset preview and determine:
1. 'target_column': The exact name of the target column.
2. 'problem_type': Either 'classification' or 'regression'.
3. 'notes': Any data quality issues (missing values, high cardinality, ID columns).

Output MUST be a valid JSON object with ONLY these keys:
{"target_column": "...", "problem_type": "...", "notes": "..."}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the dataset profile:\\n{context}"}
        ]

        print("[DataAgent] Querying LLM for initial profile...")
        try:
            response = completion(model=self.llm_model, messages=messages, temperature=0.1)
            raw_content = response.choices[0].message.content
            
            # Clean up JSON if wrapped in markdown
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].strip()
                
            profile = json.loads(raw_content)
            profile["csv_path"] = csv_path
            
            # Basic stats
            profile["num_rows"] = df.shape[0]
            profile["num_cols"] = df.shape[1]
            profile["missing_values"] = df.isnull().sum().sum()
            
            print(f"[DataAgent] Detected Target: '{profile.get('target_column')}' ({profile.get('problem_type')})")
            return profile
            
        except Exception as e:
            print(f"[DataAgent] LLM parsing failed: {e}")
            return {
                "target_column": "",
                "problem_type": "unknown",
                "notes": "Failed to generate profile.",
                "csv_path": csv_path
            }

    def generate_notebook(self, profile: dict) -> str:
        csv_path = profile.get("csv_path")
        if not csv_path:
            return ""
        
        print(f"[DataAgent] Generating EDA Notebook...")
        output_path = create_eda_notebook(csv_path, profile)
        print(f"[DataAgent] Notebook saved to '{output_path}'")
        return output_path
