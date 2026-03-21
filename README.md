# MLBuilder — Automated Machine Learning Pipeline

A modular, efficient AutoML pipeline for tabular datasets built with Scikit-learn. Supports both **classification** and **regression** problems with automatic detection, data cleaning, feature engineering, efficient model training, hyperparameter tuning, explainability (SHAP), and HTML report generation.

---

## Features

| Phase | Capability |
|-------|-----------|
| **Data Loading** | CSV and Excel input, auto problem-type detection, stratified train/test split |
| **Data Cleaning** | Duplicate removal, median imputation (numeric), mode imputation (categorical), ID and leakage column auto-detection |
| **Feature Processing** | Auto column-type detection, `StandardScaler` + `OneHotEncoder`, optional mutual-information feature selection |
| **Model Training** | Baseline screening on subsample with early-stopping, full training on promising models, parallel execution (`n_jobs=-1`), wall-clock time budget |
| **Model Selection** | Evaluation on held-out test set, best model selection (F1 for classification, RMSE for regression), optional hyperparameter tuning (top-2 models) |
| **EDA** | Dataset summary, target/feature distribution plots, correlation heatmap |
| **Explainability** | Built-in/permutation feature importance, SHAP summary and bar plots |
| **Reporting** | Self-contained HTML report with embedded images |

---

## Project Structure

```
MLBuilder/
├── main.py                  # CLI entry point — orchestrates the full pipeline
├── data_loader.py           # Load CSV/Excel, detect problem type, split data
├── data_cleaner.py          # Remove duplicates, impute missing values
├── feature_processing.py    # ColumnTransformer + optional feature selection
├── model_trainer.py         # Model catalogue, baseline screening, full training
├── model_selector.py        # Evaluate, select best, tune, persist
├── eda.py                   # Exploratory data analysis plots
├── explainer.py             # Feature importance + SHAP explanations
├── report_generator.py      # Self-contained HTML report builder
├── config.json              # Default configuration
├── requirements.txt         # Python dependencies
├── models/                  # Saved best model (.pkl) and metrics (.csv)
└── reports/                 # EDA images, explanation plots, report.html
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone <repository-url>
cd MLBuilder

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Minimal Run (Training Only)

```bash
python main.py --dataset data.csv --target label
```

### With Report Generation

```bash
python main.py --dataset data.csv --target label --report
```

### Skip SHAP for Faster Reports

```bash
python main.py --dataset data.csv --target label --report --no-shap
```

### Select Specific Models

```bash
python main.py --dataset data.csv --target label --models rf,gb
```

### Full Example with All Options

```bash
python main.py \
  --dataset properties.csv \
  --target price \
  --models rf,gb \
  --sample 0.4 \
  --cv 3 \
  --max_time 10m \
  --tune \
  --tune_method randomized \
  --report
```

---

## CLI Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | *required* | Path to CSV or Excel file |
| `--target` | str | *required* | Name of the target column |
| `--models` | str | `all` | Comma-separated model keys (see below) |
| `--sample` | float | `0.3` | Subsample fraction for baseline screening |
| `--cv` | int | `5` | Cross-validation folds |
| `--max_time` | str | None | Training time budget (e.g. `10m`, `30s`, `1h`) |
| `--tune` | flag | off | Enable hyperparameter tuning |
| `--tune_method` | str | `randomized` | `grid` or `randomized` search |
| `--feature_select` | str | None | Feature selection method (`mutual_info`) |
| `--feature_k` | int | `10` | Number of features to retain |
| `--test_size` | float | `0.2` | Test split fraction |
| `--output_dir` | str | `models` | Directory for saved model and metrics |
| `--config` | str | `config.json` | Path to config JSON file |
| `--report` | flag | off | Generate EDA + explainability HTML report |
| `--no-shap` | flag | off | Skip SHAP plots (faster) |

### Available Model Keys

| Problem Type | Key | Algorithm |
|-------------|-----|-----------|
| Classification | `logistic` | Logistic Regression |
| Classification | `rf` | Random Forest Classifier |
| Classification | `gb` | Gradient Boosting Classifier |
| Regression | `linear` | Linear Regression |
| Regression | `rf` | Random Forest Regressor |
| Regression | `gb` | Gradient Boosting Regressor |

---

## Configuration File

All CLI arguments can also be set in `config.json`. CLI arguments take precedence over config values.

```json
{
    "dataset": null,
    "target": null,
    "test_size": 0.2,
    "models": ["all"],
    "sample_fraction": 0.3,
    "cv_folds": 5,
    "max_time_minutes": null,
    "feature_selection": null,
    "feature_selection_k": 10,
    "tune": false,
    "tune_method": "randomized",
    "tune_iter": 20,
    "save_metrics_csv": true,
    "output_dir": "models",
    "random_state": 42
}
```

---

## Pipeline Steps

The pipeline executes these steps in order:

```
STEP 1  Load dataset         → CSV/Excel, auto-detect problem type, split 80/20
STEP 2  Clean data            → Remove duplicates, impute missing values
STEP 3  Feature processing    → StandardScaler (numeric) + OneHotEncoder (categorical)
STEP 4  Baseline screening    → Train all models on subsample, drop underperformers
STEP 5  Full training         → Train promising models on full training set
STEP 6  Evaluation            → Test-set metrics, select best model, save outputs
STEP 7  EDA (--report)        → Summary stats, distribution plots, correlation heatmap
STEP 8  Explainability        → Feature importance + SHAP plots, generate HTML report
        (--report)
```

---

## Evaluation Metrics

### Classification

| Metric | Description |
|--------|-------------|
| Accuracy | Correct predictions / total predictions |
| Precision | True positives / predicted positives (weighted) |
| Recall | True positives / actual positives (weighted) |
| F1 Score | Harmonic mean of precision and recall (weighted) |
| ROC-AUC | Area under the ROC curve |

**Selection criterion:** highest F1 score.

### Regression

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |

**Selection criterion:** lowest RMSE.

---

## Output

After a successful run, the pipeline produces:

```
models/
├── best_model.pkl      # Serialized best model (joblib)
└── metrics.csv         # All models with their test-set metrics

reports/                # Generated when --report is used
├── eda/
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   └── correlation_heatmap.png
├── explanations/
│   ├── feature_importance.png
│   ├── shap_summary.png        # Omitted if --no-shap
│   └── shap_importance.png     # Omitted if --no-shap
└── report.html                 # Self-contained HTML report
```

---

## Module API

Each module is importable and reusable independently:

```python
from data_loader import load_dataset, DataBundle
from data_cleaner import clean
from feature_processing import build_preprocessor, select_features
from model_trainer import get_models, baseline_screen, full_train
from model_selector import evaluate_models, select_best, tune_top_models, save_model
from eda import run_eda
from explainer import run_explanations
from report_generator import generate_report
```

---

## Built-in Safeguards

- **Leakage Detection** — Automatically drops numeric features with |correlation| > 0.95 to target, and any feature that alone achieves ROC-AUC > 0.98 via a single-feature decision tree test.
- **ID Column Removal** — Drops columns named `id` or ending with `_id` that have high cardinality.
- **Early Stopping** — Models scoring in the bottom 30% of baseline range are dropped before full training.
- **Time Budget** — Optional `--max_time` prevents runaway training sessions.

---

## Requirements

- Python ≥ 3.9
- pandas ≥ 1.5.0
- numpy ≥ 1.23.0
- scikit-learn ≥ 1.2.0
- joblib ≥ 1.2.0
- openpyxl ≥ 3.0.0
- matplotlib ≥ 3.6.0
- seaborn ≥ 0.12.0
- shap ≥ 0.42.0

---

## License

This project is provided as-is for educational and personal use.
