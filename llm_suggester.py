# llm_suggester.py
import json
import os
import time
from wandb_logger import log
from config import LLM_MODEL

try:
    import litellm
except ImportError:
    litellm = None

LLM_SYSTEM_PROMPT = """You are an expert AutoML system. Given dataset 
meta-features, suggest the best ML models to try. Always respond with 
valid JSON only — no explanation, no markdown, no preamble.

Response format:
{
  "suggestions": ["model1", "model2", "model3"],
  "reasoning": "one sentence explanation"
}

CRITICAL RULE: For regression problems you MUST only use these exact names:
ridge, lasso, elastic, sgd_reg, knn_reg, dt_reg, svr, mlp_reg,
rf_reg, et_reg, ada_reg, bag_reg, gb_reg, lgbm_reg, xgb_reg

For classification problems you MUST only use these exact names:
logistic, sgd_clf, knn_clf, naive_bayes, dt_clf, svc, mlp_clf,
rf, et_clf, ada_clf, bag_clf, gb, lgbm_clf, xgb_clf

NEVER use 'linear', 'random_forest', 'gradient_boosting' or any 
other generic names. Use ONLY the exact names above.

Return exactly 3 model names, most promising first."""

def validate_model_names(suggestions: list, problem_type: str) -> list:
    valid_clf = {"logistic","sgd_clf","knn_clf","naive_bayes",
                 "dt_clf","svc","mlp_clf","rf","et_clf","ada_clf",
                 "bag_clf","gb","lgbm_clf","xgb_clf"}
    valid_reg = {"ridge","lasso","elastic","sgd_reg","knn_reg",
                 "dt_reg","svr","mlp_reg","rf_reg","et_reg",
                 "ada_reg","bag_reg","gb_reg","lgbm_reg","xgb_reg"}
    
    valid = valid_clf if problem_type == "classification" else valid_reg
    cleaned = [m for m in suggestions if m in valid]
    
    if not cleaned:
        print(f"  [LLM] All returned names invalid. Returning empty.")
    return cleaned


def get_llm_suggestions(meta_features: dict, 
                        problem_type: str,
                        dataset_id: str = "unknown") -> tuple:
    """
    Returns (suggestions: list, reasoning: str, success: bool)
    Falls back to empty list on failure.
    """
    prompt = f"""Dataset meta-features:
- Problem type: {problem_type}
- Samples: {meta_features.get('n_samples', 'unknown')}
- Features: {meta_features.get('n_features', 'unknown')}
- Numeric ratio: {meta_features.get('num_ratio', 0):.2f}
- Categorical ratio: {meta_features.get('cat_ratio', 0):.2f}
- Missing rate: {meta_features.get('missing_rate', 0):.2f}
- Mean skewness: {meta_features.get('skewness_mean', 0):.2f}
- Mean correlation: {meta_features.get('mean_corr', 0):.2f}
- Target entropy: {meta_features.get('target_entropy', 0):.2f}
- Is binary: {meta_features.get('is_binary', False)}
- N classes: {meta_features.get('n_classes', 2)}

Suggest the 3 best models for this dataset."""

    start = time.time()

    try:
        if litellm is None:
            raise ValueError("litellm is not installed. Please run `pip install litellm` to use LLM routing.")

        response = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        
        latency_ms = (time.time() - start) * 1000
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        
        # Use regex to pluck the JSON object directly from the response,
        # completely bypassing <think> blocks or conversational padding.
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            raw = json_match.group(0)
        else:
            raw = content.strip()

        parsed = json.loads(raw)
        suggestions = validate_model_names(parsed.get("suggestions", []), problem_type)
        reasoning = parsed.get("reasoning", "")

        log({
            "llm/dataset_id":      dataset_id,
            "llm/suggestions":     str(suggestions),
            "llm/reasoning":       reasoning,
            "llm/latency_ms":      latency_ms,
            "llm/input_tokens":    response.usage.prompt_tokens,
            "llm/output_tokens":   response.usage.completion_tokens,
            "llm/success":         True,
        })

        return suggestions, reasoning, True

    except Exception as e:
        print(f"  [LLM] Failed: {e}. Using empty suggestions.")
        log({
            "llm/dataset_id": dataset_id,
            "llm/success":    False,
            "llm/error":      str(e),
        })
        return [], "", False
