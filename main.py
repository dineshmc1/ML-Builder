"""
main.py
=======
CLI entry point that orchestrates the full MLBuilder pipeline:

    load → clean → (optional) feature engineering → feature process
    → baseline screen → full train → evaluate → (optional) tune
    → save best model → (optional) EDA + explainability + HTML report.

Usage
-----
::

    python main.py --dataset data.csv --target label
    python main.py --dataset data.csv --target label --enable_fe --report
    python main.py --dataset data.csv --target price --enable_fe --outlier_strategy cap --report --no-shap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import pandas as pd

from data_loader import load_dataset
from data_cleaner import clean
from feature_processing import build_preprocessor, select_features
from model_trainer import get_models, baseline_screen, full_train
from model_selector import (
    evaluate_models,
    save_metrics,
    save_model,
    select_best,
    tune_top_models,
)
from resource_manager import ResourceManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(value: Optional[str]) -> Optional[float]:
    """Convert a human-friendly time string (e.g. ``'10m'``) to seconds."""
    if value is None:
        return None
    value = value.strip().lower()
    if value.endswith("m"):
        return float(value[:-1]) * 60
    if value.endswith("s"):
        return float(value[:-1])
    if value.endswith("h"):
        return float(value[:-1]) * 3600
    return float(value) * 60


def _load_config(path: str = "config.json") -> dict:
    """Load configuration from a JSON file if it exists."""
    if os.path.isfile(path):
        with open(path, "r") as fh:
            cfg = json.load(fh)
        print(f"[Config] Loaded settings from {path}")
        return cfg
    return {}


def _print_results(results: pd.DataFrame, problem_type: str) -> None:
    """Pretty-print the evaluation results table."""
    print("\n" + "=" * 72)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 72)

    if problem_type == "classification":
        cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    else:
        cols = ["model", "rmse", "mae", "r2"]

    display = results[[c for c in cols if c in results.columns]].copy()

    for col in display.columns:
        if col != "model" and display[col].dtype in ("float64", "float32"):
            display[col] = display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )

    print(display.to_string(index=False))
    print("=" * 72 + "\n")


# CLI argument parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLBuilder — Automated Machine Learning Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Phase 1
    p.add_argument("--dataset", type=str, required=True, help="Path to CSV or Excel dataset.")
    p.add_argument("--target", type=str, required=True, help="Name of the target column.")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated model keys (e.g. logistic,rf,gb). Default: all.")
    p.add_argument("--sample", type=float, default=None, help="Subsample fraction for baseline screening (0-1).")
    p.add_argument("--cv", type=int, default=None, help="Number of cross-validation folds.")
    p.add_argument("--max_time", type=str, default=None, help="Max training wall-clock time (e.g. 10m, 30s).")
    p.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning for top models.")
    p.add_argument("--tune_method", type=str, default=None, choices=["grid", "randomized"],
                   help="Tuning strategy (default: randomized).")
    p.add_argument("--feature_select", type=str, default=None, help="Feature selection method (mutual_info).")
    p.add_argument("--feature_k", type=int, default=None, help="Number of features to select.")
    p.add_argument("--test_size", type=float, default=None, help="Test split fraction.")
    p.add_argument("--output_dir", type=str, default=None, help="Directory for saved model & metrics.")
    p.add_argument("--config", type=str, default="config.json", help="Path to config JSON file.")
    p.add_argument("--report", action="store_true", help="Generate EDA + explainability HTML report.")
    p.add_argument("--no-shap", action="store_true", dest="no_shap", help="Skip SHAP plots (faster).")
    p.add_argument("--enable_fe", action="store_true", help="Enable smart feature engineering.")
    p.add_argument("--cardinality_threshold", type=float, default=None,
                   help="Unique-ratio threshold for high-cardinality encoding (default: 0.05).")
    p.add_argument("--skew_threshold", type=float, default=None,
                   help="Skewness threshold for log transform (default: 1.0).")
    p.add_argument("--interaction_features", type=int, default=None,
                   help="Top-k numeric features for pairwise interactions (0 = off).")
    p.add_argument("--outlier_strategy", type=str, default=None,
                   choices=["cap", "none"], help="Outlier handling: cap or none.")
    p.add_argument("--encoding_strategy", type=str, default=None,
                   choices=["target", "frequency"], help="High-cardinality encoding: target or frequency.")
    return p


# Main pipeline

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the end-to-end MLBuilder pipeline."""
    wall_start = time.time()

    # merge CLI args over config defaults 
    cfg = _load_config(args.config)
    dataset_path = args.dataset or cfg.get("dataset")
    target_col = args.target or cfg.get("target")
    model_names = (
        args.models.split(",") if args.models else cfg.get("models", ["all"])
    )
    sample_frac = args.sample or cfg.get("sample_fraction", 0.3)
    cv_folds = args.cv or cfg.get("cv_folds", 5)
    max_time = _parse_time(args.max_time) or (
        cfg.get("max_time_minutes", None)
        and cfg["max_time_minutes"] * 60
    )
    do_tune = args.tune or cfg.get("tune", False)
    tune_method = args.tune_method or cfg.get("tune_method", "randomized")
    tune_iter = cfg.get("tune_iter", 20)
    feat_select = args.feature_select or cfg.get("feature_selection", None)
    feat_k = args.feature_k or cfg.get("feature_selection_k", 10)
    test_size = args.test_size or cfg.get("test_size", 0.2)
    output_dir = args.output_dir or cfg.get("output_dir", "models")
    random_state = cfg.get("random_state", 42)
    save_csv = cfg.get("save_metrics_csv", True)
    do_report = True
    skip_shap = getattr(args, "no_shap", False) or cfg.get("no_shap", False)
    reports_dir = cfg.get("reports_dir", "reports")
    do_fe = getattr(args, "enable_fe", False) or cfg.get("enable_fe", False)
    cardinality_thr = args.cardinality_threshold or cfg.get("cardinality_threshold", 0.05)
    skew_thr = args.skew_threshold or cfg.get("skew_threshold", 1.0)
    interaction_k = (args.interaction_features if args.interaction_features is not None
                     else cfg.get("interaction_features", 0))
    outlier_strat = args.outlier_strategy or cfg.get("outlier_strategy", "cap")
    encoding_strat = args.encoding_strategy or cfg.get("encoding_strategy", "frequency")

    # Step count
    base_steps = 7 if do_fe else 6
    total_steps = base_steps + (2 if do_report else 0)

    if not dataset_path or not target_col:
        print("ERROR: --dataset and --target are required.")
        sys.exit(1)

    step = 0

    # 1. Load dataset 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Loading dataset")
    print("─" * 72)
    bundle = load_dataset(
        dataset_path, target_col, test_size=test_size,
        random_state=random_state,
    )

    # Keep raw data for EDA (before cleaning)
    raw_df = pd.concat([bundle.X_train, bundle.X_test], ignore_index=True)
    raw_df[target_col] = pd.concat(
        [bundle.y_train, bundle.y_test], ignore_index=True,
    )

    # 2. Clean data 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Cleaning data")
    print("─" * 72)
    X_train_clean, y_train_clean = clean(bundle.X_train, bundle.y_train)
    X_test_clean, y_test_clean = clean(bundle.X_test, bundle.y_test, verbose=False)

    # 2.5 Resource-Aware Engine Analysis
    rm = ResourceManager(
        max_onehot_features=cfg.get("max_onehot_features", 5000),
        low_cardinality_threshold=cfg.get("low_cardinality_threshold", 0.01),
        high_cardinality_threshold=cfg.get("high_cardinality_threshold", 0.1),
        high_cardinality_strategy=cfg.get("high_cardinality_strategy", "target"),
    )
    resource_config = rm.analyze(X_train_clean, problem_type=bundle.problem_type)

    # Override pipeline settings based on resource constraints
    do_fe = do_fe and resource_config["enable_fe"]
    interaction_k = min(interaction_k, resource_config["interaction_k"])
    encoding_map = resource_config["encoding_strategies"]
    
    # Restrict models based on dataset size constraints
    if "all" not in model_names:
        model_names = [m for m in model_names if m in resource_config["models_to_run"]]
        if not model_names:
            model_names = resource_config["models_to_run"]
    else:
        model_names = resource_config["models_to_run"]

    # 3. Feature Engineering (Phase 3, optional) 
    fe_engine = None
    fe_log: list[str] = []
    scaler_map = None

    if do_fe:
        from feature_engineering import FeatureEngineer

        step += 1
        print("\n" + "─" * 72)
        print(f"  STEP {step} / {total_steps} — Smart Feature Engineering")
        print("─" * 72)

        fe_engine = FeatureEngineer(
            cardinality_threshold=cardinality_thr,
            skew_threshold=skew_thr,
            outlier_strategy=outlier_strat,
            encoding_strategy=encoding_strat,
            interaction_features=interaction_k,
            random_state=random_state,
            encoding_map=encoding_map,
        )

        X_train_clean = fe_engine.fit_transform(
            X_train_clean, y_train_clean, bundle.problem_type,
        )
        X_test_clean = fe_engine.transform(X_test_clean)
        fe_log = fe_engine.log.copy()
        scaler_map = fe_engine.get_scalers()

        if not fe_log:
            print("[FE] No conditional transforms were applied.")

    # N. Feature processing 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Feature processing")
    print("─" * 72)
    preprocessor, num_cols, cat_cols = build_preprocessor(
        X_train_clean, scaler_map=scaler_map, encoding_map=encoding_map
    )

    # Optional feature selection (applied after preprocessing)
    feature_selector = None
    if feat_select:
        preprocessor.fit(X_train_clean)
        X_train_transformed = preprocessor.transform(X_train_clean)
        X_train_transformed, feature_selector = select_features(
            X_train_transformed, y_train_clean, bundle.problem_type,
            method=feat_select, k=feat_k,
        )

    # N+1. Baseline screening 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Baseline screening")
    print("─" * 72)
    models = get_models(bundle.problem_type, model_names)
    promising, baseline_scores = baseline_screen(
        models, preprocessor, X_train_clean, y_train_clean,
        bundle.problem_type, sample_frac=sample_frac, cv=cv_folds,
        random_state=random_state, max_time_seconds=max_time,
    )

    # N+2. Full training on promising models 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Full training")
    print("─" * 72)
    trained, full_scores = full_train(
        promising, preprocessor, X_train_clean, y_train_clean,
        bundle.problem_type, cv=cv_folds, max_time_seconds=max_time,
    )

    if not trained:
        print("ERROR: No models were trained (possible time-budget issue).")
        sys.exit(1)

    # N+3. Evaluation & selection 
    step += 1
    print("\n" + "─" * 72)
    print(f"  STEP {step} / {total_steps} — Evaluation & selection")
    print("─" * 72)
    results = evaluate_models(
        trained, X_test_clean, y_test_clean, bundle.problem_type,
    )
    _print_results(results, bundle.problem_type)

    best_name = select_best(results, bundle.problem_type)
    print(f"  ★ Best model: {best_name}")

    # Optional: hyperparameter tuning 
    if do_tune:
        print("\n" + "─" * 72)
        print("  BONUS — Hyperparameter tuning")
        print("─" * 72)
        tuned = tune_top_models(
            trained, X_train_clean, y_train_clean,
            bundle.problem_type, results,
            top_n=2, method=tune_method, n_iter=tune_iter, cv=cv_folds,
        )
        tuned_results = evaluate_models(
            tuned, X_test_clean, y_test_clean, bundle.problem_type,
        )
        _print_results(tuned_results, bundle.problem_type)
        best_name = select_best(tuned_results, bundle.problem_type)
        print(f"  ★ Best model (after tuning): {best_name}")
        trained.update(tuned)
        results = pd.concat([results, tuned_results]).drop_duplicates(
            subset="model", keep="last",
        ).reset_index(drop=True)

    # Save outputs 
    best_model = trained[best_name]
    model_path = os.path.join(output_dir, "best_model.pkl")
    save_model(best_model, model_path)

    if save_csv:
        metrics_path = os.path.join(output_dir, "metrics.csv")
        save_metrics(results, metrics_path)

    # Phase 2: EDA + Explainability + Report 
    if do_report:
        from eda import run_eda
        from explainer import run_explanations
        from report_generator import generate_report

        step += 1
        print("\n" + "─" * 72)
        print(f"  STEP {step} / {total_steps} — Exploratory Data Analysis")
        print("─" * 72)
        eda_result = run_eda(
            raw_df, target_col, bundle.problem_type,
            output_dir=os.path.join(reports_dir, "eda"),
        )

        step += 1
        print("\n" + "─" * 72)
        print(f"  STEP {step} / {total_steps} — Explainability & Report")
        print("─" * 72)
        explanation_paths = run_explanations(
            best_model, X_test_clean, y_test_clean,
            output_dir=os.path.join(reports_dir, "explanations"),
            use_shap=not skip_shap,
        )

        report_path = generate_report(
            summary=eda_result["summary"],
            results=results,
            best_name=best_name,
            problem_type=bundle.problem_type,
            eda_paths=eda_result,
            explanation_paths=explanation_paths,
            fe_log=fe_log,
            output_path=os.path.join(reports_dir, "report.html"),
        )

    elapsed = time.time() - wall_start
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s.")


# Entry point

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
