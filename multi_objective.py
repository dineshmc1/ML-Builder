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

def calculate_utility_absolute(score, time_s, complexity, w1, w2, w3, task_type='classification'):
    """
    Absolute utility calculation for a single HPO trial, without batch normalization.
    """
    # Speed penalty: normalize time 0 to 10s
    speed_penalty = min(time_s / 10.0, 1.0)
    # Simplicity penalty: normalize complexity 1-4 to 0-1
    comp_penalty = (complexity - 1) / 3.0
    
    if task_type == 'classification':
        return w1 * score - w2 * speed_penalty - w3 * comp_penalty
    else:
        # Regression score is negative MSE. We want to maximize it.
        # Add a minor relative penalty for speed/complexity to prevent scale mismatch.
        penalty_factor = 1.0 + (w2 * speed_penalty) + (w3 * comp_penalty)
        if score < 0:
            return score * penalty_factor
        else:
            return score / penalty_factor


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
