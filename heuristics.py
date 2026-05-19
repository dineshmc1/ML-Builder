# heuristics.py

def get_heuristic_suggestions(meta_features: dict, 
                               problem_type: str) -> list:
    """
    Returns ordered list of model names based on dataset properties.
    meta_features dict keys match your embedding features:
        n_samples, n_features, num_ratio, cat_ratio,
        missing_rate, skewness_mean, high_skew_frac,
        mean_corr, cv_mean, avg_cardinality, high_card_frac,
        cat_missing_rate, cat_entropy, n_classes,
        target_entropy, majority_class_ratio, is_binary
    """
    suggestions = []
    n = meta_features.get("n_samples", 1000)
    f = meta_features.get("n_features", 10)
    cat_ratio = meta_features.get("cat_ratio", 0.0)
    num_ratio = meta_features.get("num_ratio", 1.0)
    missing = meta_features.get("missing_rate", 0.0)
    is_binary = meta_features.get("is_binary", False)
    n_classes = meta_features.get("n_classes", 2)
    skew = meta_features.get("skewness_mean", 0.0)
    corr = meta_features.get("mean_corr", 0.0)

    if problem_type == "classification":

        # Rule 1: Large dataset → tree-based wins
        if n > 5000:
            suggestions.extend(["lgbm_clf", "xgb_clf", "rf"])

        # Rule 2: High categorical ratio → tree-based
        if cat_ratio > 0.6:
            suggestions.extend(["lgbm_clf", "xgb_clf", "gb"])

        # Rule 3: High dimensional numeric → linear first
        if f > 100 and num_ratio > 0.8:
            suggestions.extend(["logistic", "sgd_clf", "svc"])

        # Rule 4: Binary + low features → simple models work
        if is_binary and f < 20:
            suggestions.extend(["logistic", "svc", "lgbm_clf"])

        # Rule 5: High skewness → tree-based handles it better
        if abs(skew) > 1.5:
            suggestions.extend(["rf", "et_clf", "lgbm_clf"])

        # Rule 6: High multicollinearity → regularized linear
        if corr > 0.7:
            suggestions.extend(["logistic", "sgd_clf"])

        # Rule 7: Many missing values → tree-based tolerates it
        if missing > 0.2:
            suggestions.extend(["lgbm_clf", "xgb_clf"])

        # Rule 8: Small dataset → simpler models
        if n < 200:
            suggestions.extend(["naive_bayes", "knn_clf", "dt_clf"])

        # Rule 9: Many classes → gradient boosting
        if n_classes > 10:
            suggestions.extend(["lgbm_clf", "rf", "xgb_clf"])

    else:  # regression

        # Rule 1: Large dataset → boosting
        if n > 5000:
            suggestions.extend(["lgbm_reg", "xgb_reg", "rf_reg"])

        # Rule 2: High dimensional → linear regularized
        if f > 100:
            suggestions.extend(["ridge", "lasso", "elastic"])

        # Rule 3: High skewness → tree handles nonlinearity
        if abs(skew) > 1.5:
            suggestions.extend(["rf_reg", "et_reg", "lgbm_reg"])

        # Rule 4: All numeric, low features → knn works
        if num_ratio > 0.9 and f < 15:
            suggestions.extend(["knn_reg", "svr", "mlp_reg"])

        # Rule 5: Small dataset → simple models
        if n < 200:
            suggestions.extend(["ridge", "knn_reg", "svr"])

        # Rule 6: High collinearity → regularization needed
        if corr > 0.7:
            suggestions.extend(["ridge", "lasso", "elastic"])

        # Rule 7: Missing values → tree-based
        if missing > 0.2:
            suggestions.extend(["lgbm_reg", "xgb_reg"])

    # Deduplicate while preserving order
    seen = set()
    ranked = []
    for m in suggestions:
        if m not in seen:
            seen.add(m)
            ranked.append(m)

    # Fallback if no rules fired
    if not ranked:
        if problem_type == "classification":
            ranked = ["lgbm_clf", "rf", "logistic"]
        else:
            ranked = ["lgbm_reg", "rf_reg", "ridge"]

    return ranked[:5]  # top 5 suggestions
