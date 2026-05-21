# multi_objective.py

MODEL_COMPLEXITY = {
    # Simplest (1)
    'logistic': 1, 'ridge': 1, 'lasso': 1, 'elastic': 1,
    'naive_bayes': 1, 'sgd_clf': 1, 'sgd_reg': 1,
    # Moderate (2)
    'knn_clf': 2, 'knn_reg': 2,
    'dt_clf': 2, 'dt_reg': 2,
    'svr': 2, 'svc': 2,
    # Complex (3)
    'mlp_clf': 3, 'mlp_reg': 3,
    'rf': 3, 'rf_reg': 3,
    'et_clf': 3, 'et_reg': 3,
    'ada_clf': 3, 'ada_reg': 3,
    'bag_clf': 3, 'bag_reg': 3,
    # Most complex (4)
    'gb': 4, 'gb_reg': 4,
    'lgbm_clf': 4, 'lgbm_reg': 4,
    'xgb_clf': 4, 'xgb_reg': 4
}


def compute_utility(model_names, scores, times, complexities,
                    w1=0.6, w2=0.3, w3=0.1):
    """
    Compute multi-objective utility score for each model.
    
    w1 = weight for accuracy (higher is better)
    w2 = weight for speed (lower time is better)
    w3 = weight for simplicity (lower complexity is better)
    """
    # Normalize scores (higher = better)
    s_min, s_max = min(scores), max(scores)
    norm_scores = [(s - s_min) / (s_max - s_min + 1e-9) 
                   for s in scores]

    # Normalize speed (lower time = better, so invert)
    t_min, t_max = min(times), max(times)
    norm_speed = [1 - (t - t_min) / (t_max - t_min + 1e-9) 
                  for t in times]
    # Cap speed penalty to prevent extreme cases
    norm_speed = [min(s, 0.5) for s in norm_speed]
    max_s = max(norm_speed)
    if max_s > 0:
        norm_speed = [s / max_s for s in norm_speed]

    # Normalize simplicity (lower complexity = better, so invert)
    c_min, c_max = min(complexities), max(complexities)
    norm_simplicity = [1 - (c - c_min) / (c_max - c_min + 1e-9)
                       for c in complexities]

    # Combine
    utility = {}
    for i, name in enumerate(model_names):
        utility[name] = (w1 * norm_scores[i] +
                         w2 * norm_speed[i] +
                         w3 * norm_simplicity[i])
    return utility


def select_best_model_multiobjective(model_results, task_type='classification',
                                      w1=None, w2=None, w3=None,
                                      max_score_drop=0.05):
    """
    model_results: dict of {model_name: {'score': float, 'time': float}}
    Returns: (best_model_name, utility_scores_dict)
    """
    # Set task-aware defaults
    if w1 is None:
        w1, w2, w3 = (0.6, 0.3, 0.1) if task_type == 'classification' \
                     else (0.8, 0.15, 0.05)
                     
    names = list(model_results.keys())
    scores = [model_results[n]['score'] for n in names]
    
    # Find best score
    best_score = max(scores)
    
    # Filter candidates: only models within max_score_drop of best
    if task_type == 'regression':
        # scores are negative, best is closest to 0
        threshold = best_score * (1 + max_score_drop)  
        eligible = {n: model_results[n] for n, s in zip(names, scores)
                    if s >= threshold}
    else:
        threshold = best_score * (1 - max_score_drop)
        eligible = {n: model_results[n] for n, s in zip(names, scores)
                    if s >= threshold}
                    
    if not eligible:
        eligible = model_results
        
    e_names = list(eligible.keys())
    e_scores = [eligible[n]['score'] for n in e_names]
    e_times = [eligible[n]['time'] for n in e_names]
    e_complexities = [MODEL_COMPLEXITY.get(n, 3) for n in e_names]

    utility = compute_utility(e_names, e_scores, e_times, 
                               e_complexities, w1, w2, w3)
    best = max(utility, key=utility.get)
    return best, utility
