# MLBuilder — Automated Machine Learning Pipeline

A modular, efficient AutoML pipeline for tabular datasets built with Scikit-learn. Supports both **classification** and **regression** problems with automatic detection, data cleaning, scalable feature engineering, resource-aware adaptive training, hyperparameter tuning, explainability (SHAP), and HTML report generation.

---

## Features

| Phase | Capability |
|-------|-----------|
| **Data Loading** | CSV and Excel input, auto problem-type detection, stratified train/test split |
| **Data Cleaning** | Duplicate removal, median imputation (numeric), mode imputation (categorical), ID and leakage column auto-detection |
| **Resource-Aware Engine** | Dynamically adapts pipeline to dataset size to prevent OOM errors. Restricts models on large datasets, prevents feature explosions by capping one-hot encoding limits, and falls back to frequency encoding. |
| **Smart Feature Engineering**| Automatic skewness handling (log transforms), outlier capping, cardinality-based encoding (one-hot/frequency/target), and top-k numeric interactions. |
| **Feature Processing** | Auto column-type detection, `StandardScaler` + encoders, optional mutual-information feature selection |
| **Model Training** | Epoch-level validation/early-stopping (for GB/LightGBM/XGBoost), baseline screening on subsample, full training on promising models, parallel execution (`n_jobs=-1`), wall-clock time budget |
| **Model Selection** | Evaluation on held-out test set, best model selection (F1 for classification, RMSE for regression), optional hyperparameter tuning (top-2 models) |
| **EDA** | Dataset summary, target/feature distribution plots, correlation heatmap |
| **Explainability** | Built-in/permutation feature importance, SHAP summary and bar plots |
| **Reporting** | Self-contained HTML report with embedded images |
| **Meta-Learning Memory** | **(Phase 4 New)** Dataset embeddings (meta-features), FAISS-based memory store, adaptive cold-start strategy for model selection. |

---

## Project Structure

```
MLBuilder/
├── main.py                  # CLI entry point — orchestrates the full pipeline
├── data_loader.py           # Load CSV/Excel, detect problem type, split data
├── data_cleaner.py          # Remove duplicates, impute missing values
├── resource_manager.py      # Adaptive engine for data constraints & OOM prevention
├── feature_engineering.py   # Smart outlier handling, transforms, and interactions
├── feature_processing.py    # ColumnTransformer + optional feature selection
├── model_trainer.py         # Model catalogue, baseline screening, full training
├── model_selector.py        # Evaluate, select best, tune, persist
├── eda.py                   # Exploratory data analysis plots
├── explainer.py             # Feature importance + SHAP explanations
├── report_generator.py      # Self-contained HTML report builder
├── dataset_embedding.py     # Phase 4: Computes dataset meta-feature embeddings
├── cold_start.py            # Phase 4: Adaptive memory retrieval & FAISS index
├── phase4_pipeline.py       # Phase 4: Testing pipeline for memory logic
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

### With Resource-Aware Feature Engineering & Report Generation

```bash
python main.py --dataset data.csv --target label --enable_fe --report
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
  --enable_fe \
  --outlier_strategy cap \
  --interaction_features 5 \
  --report
```

---

## Phase 4: Meta-Learning Memory (Current Active Phase)

The project is currently in **Phase 4: Meta-Learning Memory**, specifically having completed sub-phases for **Learned Task Embeddings (Phase 4.1 / Item 4)** and **Adaptive Memory Retrieval Policy (Phase 4.2 / Item 1)**.

### Features in this Phase:
- **Rich Dataset Embedding Vectors (`dataset_embedding.py`)**: Computes a 10-dimensional `float32` vector that summarizes a dataset's statistical fingerprint (samples, features, missing ratio, skewness, kurtosis, entropy, etc.). This replaces handcrafted meta-features with learned embeddings to capture task similarity.
- **Adaptive Cold-Start Strategy (`cold_start.py`)**: Given a new dataset embedding, an adaptive decision engine determines whether to use **memory-based retrieval** (leveraging past experiments on similar datasets) or fallback to a **cold-start** broad search.
- **FAISS Memory Store**: A lightweight in-memory vector database (`MemoryStore`) that indexes dataset embeddings and maps them to the best-performing models.
- **Dynamic Thresholding $\epsilon(D)$**: Computes similarity scores between the query dataset and memory to adaptively decide the routing threshold.

### Running Phase 4 Pipeline
A dedicated testing pipeline has been created to evaluate the adaptive cold-start logic against 50 OpenML datasets.

```bash
python phase4_pipeline.py
```

This script will:
1. Build a memory store from a subset of datasets.
2. Extract meta-features and embeddings for unseen test datasets.
3. Query the FAISS memory and evaluate the decision engine's routing (Memory vs Fallback).
4. Save the experiment results to `phase4_results.csv`.

---

## CLI Reference

### Core Pipeline Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | *required* | Path to CSV or Excel file |
| `--target` | str | *required* | Name of the target column |
| `--models` | str | `all` | Comma-separated model keys (e.g., logistic,rf,gb) |
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
| `--skip_cv_large` | flag | off | Skip cross validation for large datasets (uses 1 validation split) |

### Feature Engineering & Resource Management Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_fe` | flag | off | Enable smart feature engineering |
| `--cardinality_threshold` | float | `0.05` | Unique-ratio threshold for high-carinality encoding |
| `--skew_threshold` | float | `1.0` | Skewness threshold for log transform |
| `--interaction_features`| int | `0` | Top-k numeric features for pairwise interactions |
| `--outlier_strategy` | str | `cap` | Outlier handling: `cap` or `none` |
| `--encoding_strategy`| str | `frequency` | High-cardinality encoding (`target` or `frequency`) |

### Reporting Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--report` | flag | off | Generate EDA + explainability HTML report |
| `--no-shap` | flag | off | Skip SHAP plots (faster) |

### Available Model Keys

| Problem Type | Key | Algorithm |
|-------------|-----|-----------|
| Classification | `logistic` | Logistic Regression |
| Classification | `rf` | Random Forest Classifier |
| Classification | `gb` | Gradient Boosting Classifier |
| Classification | `lightgbm` | LightGBM Classifier |
| Classification | `xgboost` | XGBoost Classifier |
| Regression | `linear` | Linear Regression |
| Regression | `rf` | Random Forest Regressor |
| Regression | `gb` | Gradient Boosting Regressor |
| Regression | `lightgbm` | LightGBM Regressor |
| Regression | `xgboost` | XGBoost Regressor |

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
    "enable_fe": false,
    "outlier_strategy": "cap",
    "encoding_strategy": "frequency",
    "interaction_features": 0,
    "save_metrics_csv": true,
    "output_dir": "models",
    "random_state": 42
}
```

---

## System Architecture

ML-Builder is designed around a highly modular, multimodal, and adaptive architecture.

1. **Orchestrator & Modality Router**: The pipeline entry point automatically detects the data type (Tabular, Vision, Audio, Text) and routes it to the correct subsystem.
2. **Adaptive Resource Manager**: A dynamic control unit that analyzes data footprint and system constraints, proactively capping memory-intensive operations (like excessive One-Hot Encoding or deep Cross-Validation) to prevent Out-Of-Memory (OOM) errors.
3. **Smart Feature Engine**: An intelligent processing layer for tabular data that performs conditional data cleaning, outlier capping, skewness correction (log-transforms), and cardinality-aware encoding.
4. **Meta-Learning Memory (Phase 4)**: A hybrid FAISS/SQLite memory database that stores dataset embeddings and their best-performing hyperparameter configurations.
5. **Domain Registry (Deep Learning)**: For non-tabular modalities (e.g., Vision), it selects domain-optimized foundation models (e.g., BioCLIP for biology, TrOCR for documents) to extract robust embeddings.
6. **Training & Auto-Tuning Engine**: Screens baseline models, drops underperformers, and fully trains the top candidates. Capable of warm-starting Neural Architecture Search (NAS) using the Meta-Learning Memory.
7. **Explainability & Reporting**: Generates interactive Exploratory Data Analysis (EDA), SHAP feature importance plots, and bundles everything into a shareable HTML report.

---

## How It Works (Step-by-Step)

The pipeline executes a highly adaptive workflow, conditionally altering its path based on the dataset's characteristics and available system memory.

### 1. Data Ingestion & Routing
- Loads the dataset and automatically detects the modality (Tabular, Vision, Audio, Text).
- Identifies the problem type (Classification vs. Regression).
- Generates a stratified train/test split.

### 2. Resource Analysis & Constraint Mapping
- The **Resource Manager** analyzes the dataset's footprint.
- It sets hard limits on feature engineering depth and cross-validation folds based on available system memory, categorizing the dataset as *Small*, *Medium*, or *Large*.

### 3. Meta-Learning Memory Lookup (Adaptive Cold-Start)
- Computes a mathematical "fingerprint" (meta-feature embedding vector) of the dataset.
- Queries the **FAISS Memory Store** for similar historical datasets.
- If a strong match is found, the system "warm-starts" by retrieving the best past models and hyperparameters, bypassing the broad baseline search.

### 4. Intelligent Feature Engineering
- **Data Cleaning**: Removes duplicates and imputes missing values (median for numeric, mode for categorical).
- **Transformations**: Conditionally applies log-transforms for skewed data, caps outliers, and limits high-cardinality categorical variables.
- **Feature Processing**: Standardizes numeric features and applies Target, Frequency, or One-Hot encoding based on the Resource Manager's budget constraints.

### 5. Baseline Model Screening
- Trains a suite of models (Linear, Random Forest, XGBoost, LightGBM, etc.) on a subsample of the data.
- Drops models scoring in the bottom percentile to save computational time.

### 6. Full Training & Hyperparameter Tuning
- Fully trains the most promising models on the complete training set.
- If enabled, performs hyperparameter tuning (Randomized or Grid Search). 

### 7. Evaluation & Selection
- Evaluates all trained models against the hold-out test set using robust metrics (e.g., F1 Score for Classification, RMSE for Regression).
- Selects the global best model and serializes it to disk for production use.

### 8. Explainability & Report Generation
- Runs comprehensive Exploratory Data Analysis (EDA) on the raw inputs.
- Computes built-in feature importance and SHAP values for the winning model.
- Compiles all metrics, charts, and configuration logs into an interactive, self-contained HTML report.

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
from resource_manager import ResourceManager
from feature_engineering import FeatureEngineer
from feature_processing import build_preprocessor, select_features
from model_trainer import get_models, baseline_screen, full_train
from model_selector import evaluate_models, select_best, tune_top_models, save_model
from eda import run_eda
from explainer import run_explanations
from report_generator import generate_report
from dataset_embedding import compute_dataset_embedding, build_embedding_matrix
from cold_start import MemoryStore, adaptive_cold_start, ColdStartConfig
```

---

## Built-in Safeguards

- **Resource-Aware Dynamic Tiering** — Automatically switches to scalable tree engines (LightGBM/XGBoost) and disables highly complex feature engineering routines for large datasets to prevent OOM errors.
- **Leakage Detection** — Automatically drops numeric features with |correlation| > 0.95 to target, and any feature that alone achieves ROC-AUC > 0.98 via a single-feature decision tree test.
- **ID Column Removal** — Drops columns named `id` or ending with `_id` that have high cardinality.
- **Early Stopping & Logs** — Models scoring in the bottom 30% of baseline range are dropped before full training. Also features live epoch evaluation logging and early stopping (10 continuous rounds with no gain) mapped automatically for Gradient Boosting structures.
- **Large Dataset CV Skipping** — With the `--skip_cv_large` parameter, large data runs clamp Cross-Validation bounds to a single 85/15 validation split, reducing wait times exponentially.
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

*Note: The Resource-Aware engine may attempt to use `xgboost` and `lightgbm` via the model catalogue if they are installed, otherwise it falls back to scikit-learn implementations.*

---

## License

This project is provided as-is for educational and personal use.
