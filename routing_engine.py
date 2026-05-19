# routing_engine.py
import numpy as np
from dataclasses import dataclass
from wandb_logger import log

@dataclass
class RoutingConfig:
    lambda_memory:     float = 0.6
    lambda_llm:        float = 0.2
    lambda_heuristic:  float = 0.2
    use_llm:           bool  = True    # can disable to save API costs
    top_k_output:      int   = 3       # how many models to return


def compute_routing_score(
    memory_models:     list,
    memory_score:      float,
    llm_models:        list,
    heuristic_models:  list,
    config:            RoutingConfig,
    problem_type:      str,
    dataset_id:        str = "unknown"
) -> tuple:
    """
    Combines three signals into a ranked model shortlist.
    Returns (ranked_models: list, signal_scores: dict)
    """
    # Collect all candidate models
    all_models = list(set(
        memory_models + llm_models + heuristic_models
    ))

    if not all_models:
        fallback = ["lgbm_clf", "rf", "logistic"] \
                   if problem_type == "classification" \
                   else ["lgbm_reg", "rf_reg", "ridge"]
        return fallback, {}

    model_scores = {}

    for model in all_models:
        # Memory signal: position-based score
        if model in memory_models:
            pos = memory_models.index(model)
            mem_score = memory_score * (1.0 - pos * 0.1)
        else:
            mem_score = 0.0

        # LLM signal: position-based score
        if model in llm_models:
            pos = llm_models.index(model)
            llm_score = 1.0 - pos * 0.2
        else:
            llm_score = 0.0

        # Heuristic signal: position-based score
        if model in heuristic_models:
            pos = heuristic_models.index(model)
            heu_score = 1.0 - pos * 0.2
        else:
            heu_score = 0.0

        # Weighted combination
        combined = (
            config.lambda_memory    * mem_score +
            config.lambda_llm       * llm_score +
            config.lambda_heuristic * heu_score
        )
        model_scores[model] = {
            "memory":    mem_score,
            "llm":       llm_score,
            "heuristic": heu_score,
            "combined":  combined
        }

    # Rank by combined score
    ranked = sorted(
        model_scores.keys(),
        key=lambda m: model_scores[m]["combined"],
        reverse=True
    )[:config.top_k_output]

    # Log to W&B
    log({
        "routing/dataset_id":       dataset_id,
        "routing/top_model":        ranked[0] if ranked else "none",
        "routing/shortlist":        str(ranked),
        "routing/memory_models":    str(memory_models),
        "routing/llm_models":       str(llm_models),
        "routing/heuristic_models": str(heuristic_models),
        "routing/lambda_memory":    config.lambda_memory,
        "routing/lambda_llm":       config.lambda_llm,
        "routing/lambda_heuristic": config.lambda_heuristic,
        "routing/n_candidates":     len(all_models),
        "routing/top_combined_score": model_scores[ranked[0]]["combined"]
                                      if ranked else 0.0,
    })

    return ranked, model_scores
