import litellm
import os
import json
import wandb

def generate_comprehensive_report(master_context, dataset_id):
    """
    Generates a Markdown report.
    Checks if the run was Tabular (AutoML) or Deep Learning (AutoDL)
    and selects the appropriate LLM prompt accordingly.
    """
    paradigm = master_context.get("paradigm_routing", {}).get("decision", "AutoML")

    # ── AutoDL (Deep Learning) Prompt ──────────────────────────────────────────
    if paradigm == "AutoDL":
        system_prompt = """You are an Expert AI Consultant specialising in Deep Learning pipelines.
    The system routed this dataset to the AutoDL (Deep Learning) path via the R(D) Paradigm Router.
    Write a structured, professional Markdown report based EXACTLY on the provided JSON context.
    Do not invent numbers or metrics that are not in the context.

    CRITICAL RULES:
    1. You MUST output exactly 5 sections, numbered 1 to 5. Do not stop early.
    2. DO NOT hallucinate. Use the exact numbers from the context.
    3. Reference the modality (vision/audio/text/video) and the specific extractor used.

    Structure your report exactly with these headings:

    # 1. Executive Summary & Dataset Context
    (Describe the dataset modality, the extractor used to convert raw media into tabular embeddings,
    the number of samples and classes discovered, and any observations about class balance).

    # 2. Why Deep Learning Was Chosen
    (Explain the R(D) Router score and why the Classical ML path was bypassed.
    Unstructured data produces dense, high-dimensional embedding vectors that standard
    tabular AutoML pipelines are not designed to exploit — explain this in business-friendly terms).

    # 3. Neural Architecture Search (NAS) Results
    (Detail the best architecture found: number of layers, hidden dimension, dropout rate,
    learning rate, and batch size. Explain what each hyperparameter means for model quality
    and generalisation. Quote the best NAS utility score).

    # 4. Final Production Model Performance
    (Present the final test accuracy, the full classification report, and interpret the
    confusion matrix. Call out any class the model struggles with and hypothesise why).

    # 5. Compute Efficiency & Recommendations
    (Explain how the PCA + lightweight MLP approach achieves competitive accuracy far faster
    than training a massive CNN from scratch. Suggest one concrete next step to push accuracy
    further, e.g. fine-tuning the backbone, data augmentation, or ensembling).
    """

    # ── Tabular (AutoML) Prompt ─────────────────────────────────────────────────
    else:
        system_prompt = """You are an Expert AI Consultant. Write a comprehensive Markdown report structured EXACTLY into these 4 pillars:

# 1. DESCRIPTIVE ANALYTICS (What happened?)
- Dataset context, size, class distribution, and data health.

# 2. DIAGNOSTIC ANALYTICS (Why did it happen?)
- Why did the model make certain mistakes? Analyze the Confusion Matrix and SHAP values. What features drove the predictions?

# 3. PREDICTIVE ANALYTICS (What will happen?)
- Present the final evaluation metrics (Accuracy, F1, ROC-AUC, etc.). How will this model perform in the real world?

# 4. PRESCRIPTIVE ANALYTICS (What should we do next?)
- Concrete business recommendations. Should we deploy? Do we need more data? What are the ROI implications?

Use the provided JSON context to fill these sections. DO NOT hallucinate."""

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

    # Log to W&B (only if an active run exists)
    if wandb.run is not None:
        try:
            wandb.log({
                f"consultant_report/{dataset_id}": wandb.Html(f"<pre style='white-space: pre-wrap; font-family: sans-serif;'>{report_md}</pre>")
            })
        except Exception as e:
            print(f"  [W&B] Failed to log consultant report: {e}")
    else:
        print("  [W&B] Skipped logging report (No active W&B run).")

    return report_md
