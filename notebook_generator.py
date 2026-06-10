import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

def generate_advanced_notebook(config: dict, results: dict, output_path: str):
    nb = new_notebook()
    modality = config.get("modality", "tabular")
    paradigm = results.get("paradigm", "AutoML")
    business_context = config.get("business_context", {})
    
    # 1. Executive Header
    nb.cells.append(new_markdown_cell(f"# 📊 MetaAutoML Advanced Analysis Report\n**Dataset:** `{config.get('data_path', 'Unknown')}`\n**Modality:** {modality.upper()} | **Paradigm:** {paradigm}\n**Business Objective:** {business_context.get('business_objective', 'N/A')}"))
    nb.cells.append(new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay\ntry:\n    import shap\nexcept ImportError:\n    pass\nsns.set_theme(style='whitegrid')"))
    
    # 2. Data Profiling & Quality
    nb.cells.append(new_markdown_cell("## 1. Data Profiling & Quality Assessment"))
    if modality == "tabular":
        nb.cells.append(new_code_cell("# Load & Profile\nX, y = results['X'], results['y']\ndf = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()\ndf['target'] = y\nprint('Shape:', df.shape)\nprint('Missing Values:\\n', df.isnull().sum())\nprint('Class Distribution:\\n', df['target'].value_counts(normalize=True) * 100)"))
    else:
        code_str = (
            f"# Multi-Modal Summary\n"
            f"print('Modality: {modality}')\n"
            f"print('Samples: {{len(results.get('y', []))}}')\n"
            f"print('Classes: {{len(set(results.get('y', [])))}}')\n"
            f"if 'X' in results:\n"
            f"    print('Embedding Dim: {{results[\"X\"].shape[1]}}')"
        ).format(results=results)
        # Actually wait, we don't need f-string to inject the len of results into the CODE of the notebook.
        # We want the python code in the notebook to just print those things.
        code_str = f"# Multi-Modal Summary\nprint('Modality: {modality}')\nprint('Samples: ', len(results.get('y', [])))\nprint('Classes: ', len(set(results.get('y', []))))\nif 'X' in results:\n    print('Embedding Dim: ', results['X'].shape[1])"
        nb.cells.append(new_code_cell(code_str))
        
    # 3. Advanced EDA (Paradigm Specific)
    nb.cells.append(new_markdown_cell("## 📈 2. Advanced Exploratory Analysis"))
    if paradigm == "AutoML":
        nb.cells.append(new_code_cell("# Correlation Heatmap & Feature Distributions\nif isinstance(X, pd.DataFrame):\n    plt.figure(figsize=(10,8))\n    sns.heatmap(X.corr(), cmap='coolwarm', annot=False)\n    plt.title('Feature Correlation Matrix')\n    plt.show()\n\n    # Target vs Key Features\n    for col in X.columns[:3]:\n        plt.figure()\n        sns.boxplot(x=y, y=X[col])\n        plt.title(f'{col} vs Target')\n        plt.show()"))
    else:
        nb.cells.append(new_code_cell("# Embedding Space Visualization (t-SNE)\nfrom sklearn.manifold import TSNE\nX_embed = results.get('X')\nif X_embed is not None and len(X_embed) > 0:\n    tsne = TSNE(n_components=2, random_state=42)\n    X_2d = tsne.fit_transform(X_embed)\n\n    plt.figure(figsize=(10,8))\n    for cls in set(results['y']):\n        mask = np.array(results['y']) == cls\n        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Class {cls}', alpha=0.7)\n    plt.title('Multi-Modal Embedding Clustering (t-SNE)')\n    plt.legend()\n    plt.show()"))
        
    # 4. Model Diagnostics & Error Analysis
    nb.cells.append(new_markdown_cell("## 🤖 3. Model Performance & Error Analysis"))
    nb.cells.append(new_code_cell(f"# Confusion Matrix & Classification Report\ny_true, y_pred = results.get('y_test', results.get('y')), results.get('y_pred')\nif y_true is not None and y_pred is not None:\n    print(classification_report(y_true, y_pred))\n\n    fig, ax = plt.subplots(figsize=(8,6))\n    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues')\n    plt.title('Confusion Matrix')\n    plt.show()"))
    
    # 5. Business Impact Simulation
    nb.cells.append(new_markdown_cell("## 💼 4. Business Impact & Recommendations"))
    nb.cells.append(new_code_cell("# ROI Simulation based on Success Metric\nmetric = config.get('business_context', {}).get('success_metric', 'Accuracy')\nacc = results.get('final_accuracy', 0.0)\nif acc:\n    print(f'Estimated {metric}: {acc:.2%}')\n    print('Recommendation: ' + ('Deploy to production.' if acc > 0.90 else 'Collect more data or adjust domain extractor.' if acc > 0.75 else 'Re-evaluate feature extraction pipeline.'))"))
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"📓 Advanced notebook saved to: {output_path}")
