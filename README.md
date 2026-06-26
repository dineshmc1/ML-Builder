# ML-Builder: Intelligent Meta-Learning & Agentic AutoML Framework

> A production-grade Automated Machine Learning and Deep Learning framework that **learns from past experiments**, uses **LLM agents** for intelligent decision-making, and delivers **SHAP-explainable models** with comprehensive HTML reports.

---

## 📖 Table of Contents
1. [What is ML-Builder?](#what-is-ml-builder)
2. [Problems in Current AutoML Systems](#problems-in-current-automl-systems)
3. [Key Features](#key-features)
4. [Pipeline Workflows](#pipeline-workflows)
5. [Design Choices: Why These Components?](#design-choices-why-these-components)
6. [Folder Structure & File Explanations](#folder-structure--file-explanations)
7. [Limitations](#limitations)
8. [Installation & Setup](#installation--setup)
9. [Quick Start](#quick-start)

---

## 🚀 What is ML-Builder?

**ML-Builder** is an advanced end-to-end framework for automating machine learning (AutoML) and deep learning (AutoDL) pipelines. 
It ingests raw datasets (Tabular, Vision, Audio, Text, Video) and outputs fully trained, optimized, and mathematically explainable models. 

**What it solves:** It automates the tedious, trial-and-error process of data cleaning, feature engineering, model selection, and hyperparameter tuning, bridging the gap between raw data and production-ready models.

**Why it solves it:** By leveraging **Meta-Learning** (a memory of past experiments) and **Agentic AI** (Large Language Models acting as data scientists), ML-Builder makes decisions intelligently rather than relying on exhaustive brute-force search.

---

## ⚠️ Problems in Current AutoML Systems

Traditional AutoML tools (like TPOT, Auto-sklearn, or H2O) suffer from fundamental flaws that ML-Builder directly addresses:

1. **Amnesia (Starting Blind):** Every time a traditional AutoML system sees a new dataset, it starts from scratch. It doesn't remember that a dataset from yesterday was nearly identical and that XGBoost with depth 4 worked best.
2. **Brute-Force Inefficiency:** They blindly loop through hundreds of algorithms and hyperparameter combinations, wasting massive amounts of compute and time.
3. **Lack of Business Context:** They optimize purely for mathematical metrics (e.g., RMSE or Accuracy) without understanding real-world constraints (e.g., latency, interpretability, or risk tolerance).
4. **Single Modality:** Most are strictly limited to tabular CSV data and cannot natively handle images, audio, or text without manual pre-processing.

---

## ✨ Key Features

- **Meta-Learning Memory:** Remembers past datasets using FAISS vector search to instantly warm-start hyperparameter optimization.
- **LLM Agentic Consultant:** A team of 5 AI Agents (Data, Business, Feature, Model, Critic) that analyze your data and business goals to dynamically generate a pipeline plan.
- **Universal Multi-Modality:** Natively supports Tabular, Vision, Audio, Text, and Video datasets using domain-specific embedders (CLIP, SentenceTransformers, AST).
- **AutoDL & Neural Architecture Search (NAS):** Automatically routes complex or unstructured data to Deep Learning pipelines, optimizing MLP/CNN architectures via Optuna.
- **Adaptive Resource Management:** Intelligently scales feature engineering and cross-validation down for massive datasets to prevent Out-Of-Memory (OOM) crashes.
- **Self-Contained Reporting:** Generates beautiful, portable HTML reports and Jupyter Notebooks with SHAP (SHapley Additive exPlanations) values for deep interpretability.

---

## 🔄 Pipeline Workflows

ML-Builder operates through several intelligent workflows depending on your data and preferences.

### 1. Meta-Learning Similarity Retrieval
* **Fingerprinting:** Converts the dataset into a 10D statistical fingerprint (rows, cols, skewness, missing rate, etc.).
* **Siamese Encoding:** Passes the 10D vector through a trained contrastive neural network to project it into a 32D semantic space.
* **Retrieval:** Queries the **FAISS MemoryStore** to find the top 5 historically similar datasets.
* **Warm-Start:** Extracts the best hyperparameters from those past datasets to seed the Optuna search, skipping hours of blind exploration.

### 2. Agentic Workflow
* **Data Agent:** Profiles the dataset and detects the target variable.
* **Business Agent:** Interacts with the user to understand goals, constraints, and success metrics.
* **Feature & Model Agents:** Propose custom feature engineering strategies and model architectures.
* **Critic Agent:** A strict dual-gate validation system that checks for data leakage, metric mismatch, and resource constraints before allowing execution.

### 3. AutoML Pipeline (Classical Machine Learning)
* **Pre-processing:** Imputation (median/mode) and automatic scaling (Standard/Robust based on normality tests).
* **Feature Engineering:** Up to 7 adaptive stages, including target encoding, skewness correction (log1p), outlier capping (IQR), and polynomial/interaction generation.
* **Screening:** Evaluates 29 different algorithms (XGBoost, LightGBM, RF, SVM, etc.) on a data subsample.
* **Full Train & Selection:** Trains the top contenders on full data, using a Multi-Objective Utility function (Accuracy + Speed + Simplicity) to pick the winner.

### 4. AutoDL Pipeline (Deep Learning)
* **Routing:** Unstructured data (Images, Audio, Text) or highly complex tabular data is routed to AutoDL.
* **Feature Extraction:** Uses Universal Embedders (e.g., CLIP) to convert media to dense vectors.
* **NAS (Neural Architecture Search):** Uses Optuna to dynamically search for the best PyTorch MLP architecture (layers, hidden dims, dropout, LR).
* **Early Stopping:** Trains the optimal architecture with patience-based early stopping to prevent overfitting.

---

## 🧠 Design Choices: Why These Components?

- **Why Meta-Learning?** Because human data scientists rely on intuition from past projects. Meta-learning mathematically replicates this intuition, cutting search times by up to 90%.
- **Why FAISS?** Facebook AI Similarity Search is the industry standard for dense vector retrieval. It allows us to search thousands of past ML experiments in milliseconds.
- **Why a Siamese Task Encoder?** Raw statistical features (like row count) don't map linearly to model performance. Contrastive learning forces datasets that require similar model families (e.g., tree-based vs. linear) closer together in the vector space.
- **Why Optuna?** Compared to GridSearch or Hyperopt, Optuna's define-by-run API and efficient Bayesian optimization (TPE) algorithm find better hyperparameter spaces faster, and it supports our memory-based "warm-starting".
- **Why SHAP?** Feature importance from trees is often biased. SHAP provides mathematically provable, game-theoretic attributions for both global (dataset-wide) and local (single prediction) interpretability.

---

## 📁 Folder Structure & File Explanations

```text
ML-Builder/
├── agents/                       # LLM Agentic System
│   ├── agent_orchestrator.py     # Coordinates all agents sequentially
│   ├── business_agent.py         # Translates human goals to ML objectives
│   ├── critic_agent.py           # 2-stage quality gate checking for leakage/mismatch
│   ├── data_agent.py             # Auto-detects targets and problem types
│   ├── feature_agent.py          # Plans categorical encoding and FE strategies
│   ├── model_agent.py            # Recommends baseline algorithms
│   └── notebook_generator.py     # Compiles EDA and analysis to Jupyter Notebooks
├── reports/                      # Output directory for HTML reports and plots
├── shap_plots/                   # Output directory for SHAP visualizations
├── wandb/                        # Weights & Biases local logs
├── auto_dl_nas.py                # Neural Architecture Search for PyTorch models
├── build_memory.py               # Script to pre-train memory on 250+ OpenML datasets
├── cold_start.py                 # Core FAISS MemoryStore and threshold logic
├── confidence_calibration.py     # Calibrates reliability of memory decisions
├── config.py                     # Global constants (LLM models, WandB config)
├── data_cleaner.py               # Imputation and duplicate removal
├── data_loader.py                # Ingestion, leakage detection, train/test splitting
├── dataset_embedding.py          # Computes the 10D statistical dataset fingerprint
├── dataset_profiler.py           # Extracts schema and metadata for LLM context
├── dl_faiss_memory.py            # Specialized FAISS memory for multimodal DL
├── domain_registry.py            # Registry of domain-specific embedders (e.g., BioClip)
├── eda.py                        # Generates basic Exploratory Data Analysis plots
├── explainer.py                  # Standard feature importance extraction
├── feature_engineering.py        # 7-stage adaptive FE engine (skew, interactions, etc.)
├── feature_processing.py         # Builds scikit-learn ColumnTransformers
├── heuristics.py                 # Rule-based fallback system for model selection
├── hpo_optuna.py                 # Hyperparameter tuning logic with warm-start injection
├── llm_explainer.py              # Generates natural language consultant reports
├── llm_suggester.py              # Queries LLM for direct model suggestions
├── main.py                       # CLI entry point for standard AutoML
├── modality_router.py            # Determines data type (tabular/vision/audio/etc.)
├── model_selector.py             # Multi-metric evaluation and model persistence
├── model_trainer.py              # 29-model catalogue and baseline screening loops
├── multi_objective.py            # Utility scoring (Accuracy + Speed + Simplicity)
├── multimodal_extractor.py       # Converts images/audio/text to dense vectors via HF
├── onboarding_agent.py           # First-touch interactive prompt for business context
├── paradigm_router.py            # Mathematically decides between AutoML and AutoDL
├── phase4_pipeline.py            # The core Meta-Learning research pipeline execution
├── report_generator.py           # Compiles plots and logs into self-contained HTML
├── routing_engine.py             # Combines Memory, LLM, and Heuristics into 1 signal
├── run_agentic_pipeline.py       # Entry point for the Agent-driven workflow
├── shap_explainer.py             # Standalone SHAP generator for Phase 4
├── task_encoder.py               # The PyTorch Siamese neural network definition
└── wandb_logger.py               # Wrapper for experiment tracking
```

---

## 🛑 Limitations

While ML-Builder is highly advanced, it currently has the following limitations:
1. **Unsupervised Learning:** The system requires labeled data. It currently only supports Supervised Learning (Classification and Regression). It cannot perform native clustering or anomaly detection without labels.
2. **Reinforcement Learning:** Not supported.
3. **Time-Series Forecasting:** While it can handle date columns via feature extraction, it lacks native sliding-window cross-validation and ARIMA/Prophet models for strict sequential forecasting.
4. **Extreme Big Data:** It relies on Pandas/Scikit-learn. It does not use distributed processing frameworks like Apache Spark or Dask. Datasets larger than your machine's RAM will cause OOM errors (though the `ResourceManager` tries to prevent this via downsampling).
5. **Generative AI:** It uses LLMs to *build* models and *explain* data, but it does not train Generative AI models (like LLMs or Diffusion models) on your data.

---

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone <repository-url>
cd ML-Builder

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install litellm torch lightgbm xgboost wandb openml python-dotenv transformers
```

**Environment Variables:**
Copy `.env.example` to `.env` and add your API keys. The system uses `litellm` and defaults to OpenRouter to access modern reasoning models.
```env
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=openrouter/deepseek/deepseek-r1-distill-llama-70b
```

---

## 🏃 Quick Start

### 1. Populate the FAISS Memory (Run Once)
To enable Meta-Learning, build the memory bank by training on OpenML datasets:
```bash
python build_memory.py
```

### 2. Standard AutoML Run
Run a classic pipeline on tabular data:
```bash
python main.py --dataset data.csv --target your_label --enable_fe --tune --report
```

### 3. Agentic & Meta-Learning Pipeline
Run the full intelligent pipeline (LLM agents + FAISS routing + Multi-modal support):
```bash
python run_agentic_pipeline.py data.csv
# or for an image directory:
python phase4_pipeline.py
# (When prompted, paste the path to your image directory)
```

### Output Files
After a run, check the generated directories:
* `models/` - Contains the saved `.pkl` pipeline and metrics.
* `reports/` - Contains the interactive HTML report and Jupyter Notebooks.
* `shap_plots/` - Contains the global and local interpretability visuals.

---
*Built for the next generation of intelligent, automated data science.*
