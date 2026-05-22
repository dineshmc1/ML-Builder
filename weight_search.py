import os
import time
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

from phase4_pipeline import DATASET_IDS, load_and_preprocess_openml
from model_trainer import get_models, baseline_screen
from feature_processing import build_preprocessor
from data_loader import detect_problem_type
from multi_objective import select_best_model_multiobjective, MODEL_COMPLEXITY
from config import WANDB_PROJECT, WANDB_ENTITY

def get_test_datasets(n=30):
    all_ids = list(dict.fromkeys(DATASET_IDS))
    # train_limit = 100 in phase4_pipeline.py
    test_ids = all_ids[100:100+n]
    return test_ids

def main():
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="weight-search-pareto")
    
    test_ids = get_test_datasets(30)
    print(f"Testing on {len(test_ids)} datasets: {test_ids}")
    
    weight_configs = {
        'classification': [
            (1.0, 0.0, 0.0),   # pure accuracy (baseline)
            (0.6, 0.3, 0.1),   # default multi-objective
            (0.5, 0.4, 0.1),   # speed-heavy
            (0.7, 0.2, 0.1),   # accuracy-heavy
            (0.33, 0.33, 0.33),# equal
        ],
        'regression': [
            (1.0, 0.0, 0.0),   # pure accuracy
            (0.8, 0.15, 0.05), # regression-default
            (0.7, 0.2, 0.1),   # speed-aware
            (0.9, 0.05, 0.05), # heavily accuracy
            (0.33, 0.33, 0.33),# equal
        ]
    }
    
    pdf_path = "multi_objective_report.pdf"
    
    with PdfPages(pdf_path) as pdf:
        for did in test_ids:
            print(f"\n[Test] Evaluating Dataset {did}...")
            X, y = load_and_preprocess_openml(did)
            if X is None:
                continue
                
            problem_type = detect_problem_type(y)
            print(f"  -> Problem Type: {problem_type}")
            
            try:
                preprocessor, _, _ = build_preprocessor(X)
                all_models = get_models(problem_type)
                
                _, all_scores = baseline_screen(
                    all_models, preprocessor, X, y, problem_type,
                    sample_frac=1.0, cv=3, random_state=42
                )
            except Exception as e:
                print(f"  -> Failed benchmark for dataset {did}: {e}")
                continue
                
            if not all_scores:
                continue
                
            # Log all models to W&B for the Pareto front as requested
            if problem_type == 'regression':
                w1_def, w2_def, w3_def = 0.8, 0.15, 0.05
            else:
                w1_def, w2_def, w3_def = 0.6, 0.3, 0.1
                
            _, def_u_scores = select_best_model_multiobjective(all_scores, task_type=problem_type, w1=w1_def, w2=w2_def, w3=w3_def)
            
            for name, metrics in all_scores.items():
                wandb.log({
                    "dataset_id": str(did),
                    "problem_type": problem_type,
                    "model": name,
                    "accuracy": metrics['score'],
                    "train_time_seconds": metrics['time'],
                    "model_complexity": MODEL_COMPLEXITY.get(name, 3),
                    "multi_objective_score": def_u_scores.get(name, 0.0)
                })
                
            # Generate PDF plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Dataset {did} ({problem_type.capitalize()}) - Model Trade-offs")
            ax.set_xlabel("Accuracy / Score")
            ax.set_ylabel("Training Time (seconds)")
            
            names = list(all_scores.keys())
            scores = [all_scores[n]['score'] for n in names]
            times = [all_scores[n]['time'] for n in names]
            complexities = [MODEL_COMPLEXITY.get(n, 3) * 50 for n in names]
            
            scatter = ax.scatter(scores, times, s=complexities, alpha=0.6, edgecolors='k')
            for i, name in enumerate(names):
                ax.annotate(name, (scores[i], times[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            table_text = "Best Models per Weight Configuration:\n\n"
            table_text += f"{'W1':>5} {'W2':>5} {'W3':>5} | {'Best Model':<15} {'Score':>8} {'Time(s)':>8} {'Utility':>8}\n"
            table_text += "-" * 65 + "\n"
            
            for w1, w2, w3 in weight_configs[problem_type]:
                best_model, u_scores = select_best_model_multiobjective(all_scores, task_type=problem_type, w1=w1, w2=w2, w3=w3)
                best_score = all_scores[best_model]['score']
                best_time = all_scores[best_model]['time']
                best_util = u_scores.get(best_model, 0.0)
                
                table_text += f"{w1:>5.2f} {w2:>5.2f} {w3:>5.2f} | {best_model:<15} {best_score:>8.4f} {best_time:>8.3f} {best_util:>8.4f}\n"
            
            plt.figtext(0.1, -0.2, table_text, family='monospace', fontsize=10, verticalalignment='top')
            fig.subplots_adjust(bottom=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            print(table_text)
            
    wandb.finish()
    print(f"\n[Done] Successfully evaluated and saved multi_objective_report.pdf")

if __name__ == '__main__':
    main()
