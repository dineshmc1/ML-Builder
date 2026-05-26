import litellm
import os
import json
import wandb

def generate_comprehensive_report(master_context, dataset_id):
    
    system_prompt = """You are an Expert AutoML System acting as a Data Science Consultant. 
    Write a structured, professional Markdown report based EXACTLY on the provided JSON context. 
    Do not invent features or metrics that are not in the context. Use plain English for business concepts, but retain technical accuracy for the engineering team.

    CRITICAL RULES:
    1. You MUST output exactly 7 sections, numbered 1 to 7. Do not stop early.
    2. DO NOT hallucinate or contradict the context (e.g., if a model is the winner, it was NOT dropped).
    3. Base your SHAP hypotheses strictly on the feature names provided.
    
    Structure your report exactly with these headings:
    # 1. Executive Summary & Dataset Context
    (Explain what the dataset is likely about based on columns/description, its size, and any data health issues like missing values or class imbalance).
    
    # 2. Intelligent Routing & Memory Retrieval
    (Explain how the system decided to approach the problem. Mention the similarity to past experience, consistency, and whether memory or LLM intuition drove the choice. Highlight any inconsistency where Memory and LLM disagreed based on the agreement score (c_agree). Do not show math formulas, just explain the meaning of the scores).
    
    # 3. Model Selection & Hyperparameter Optimization (HPO)
    (Explain which models were tested, which were dropped and why, and explicitly state why the final winning model was selected. Detail the exact HPO parameters found and what they mean for this specific model).
    
    # 4. Multi-Objective Trade-offs
    (Explain the balance between Accuracy, Speed, and Complexity. Mention the weights used based on the problem type and the final Utility Score. Explain why this model was the "best overall" rather than just the "most accurate").
    
    # 5. Model Interpretability (SHAP)
    (Explain the top 3 driving features. Give a business-friendly hypothesis on WHY these features impact the prediction based on their names. Provide a deeper explanation of how these features might interact in the context of the domain, avoiding surface-level generic statements).
    
    # 6. Confidence & System Calibration
    (Explain the Confidence Score C(D). Explain what the ECE (Expected Calibration Error) means for trusting this system's future predictions).
    
    # 7. Search Efficiency & Transfer Learning
    (Report and explain the Search Compression Ratio (SCR), showcasing how much faster the system is than brute-force. Report the Performance Retention (PR) to show how much accuracy was retained despite the reduced search space. Finally, present the Transfer Utility Score (TUS) as the ultimate composite metric that mathematically balances accuracy retention with compute savings.)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate the comprehensive report based on this run metadata:\n{json.dumps(master_context, indent=2)}"}
    ]
    
    # Use LLM configuration from config.py if available, otherwise default
    try:
        from config import LLM_MODEL
    except ImportError:
        LLM_MODEL = "openrouter/deepseek/deepseek-r1-distill-llama-70b"

    print(f"  [LLM] Generating Consultant Report for Dataset {dataset_id}...")
    try:
        response = litellm.completion(
            model=os.getenv("LLM_MODEL", LLM_MODEL), 
            messages=messages,
            temperature=0.4,
            max_tokens=3000
        )
        report_md = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM] Report Generation Failed: {str(e)}")
        report_md = f"LLM Report Generation Failed: {str(e)}"
        
    # Save to local disk
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{dataset_id}_consultant_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Log to W&B
    try:
        wandb.log({
            f"consultant_report/{dataset_id}": wandb.Html(f"<pre style='white-space: pre-wrap; font-family: sans-serif;'>{report_md}</pre>")
        })
    except Exception as e:
        print(f"  [W&B] Failed to log consultant report: {e}")

    return report_md
