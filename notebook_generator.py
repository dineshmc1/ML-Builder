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
    nb.cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
try:
    import shap
except ImportError:
    pass
sns.set_theme(style='whitegrid')"""))
    
    # 2. Data Profiling & Quality
    nb.cells.append(new_markdown_cell("## 1. Data Profiling & Quality Assessment"))
    if modality == "tabular":
        nb.cells.append(new_code_cell("""# Load & Profile
X, y = results['X'], results['y']
df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
df['target'] = y
print('Shape:', df.shape)
print('Missing Values:\\n', df.isnull().sum())
print('Class Distribution:\\n', df['target'].value_counts(normalize=True) * 100)"""))
    else:
        # FIX: Used triple quotes to avoid the 'len(results' crash
        nb.cells.append(new_code_cell("""print('Modality:', '""" + modality + """')
print('Total Samples:', len(results['y']))
print('Classes:', len(set(results['y'])))
print('Embedding Dimensions:', results['X'].shape[1])"""))
        
    # 3. Advanced EDA (Paradigm Specific)
    nb.cells.append(new_markdown_cell("## 📈 2. Advanced Exploratory Analysis"))
    if paradigm == "AutoML":
        nb.cells.append(new_code_cell("""# Correlation Heatmap & Feature Distributions
if isinstance(X, pd.DataFrame):
    plt.figure(figsize=(10,8))
    sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Target vs Key Features
    for col in X.columns[:3]:
        plt.figure()
        sns.boxplot(x=y, y=X[col])
        plt.title(f'{col} vs Target')
        plt.show()"""))
    else:
        nb.cells.append(new_code_cell("""# Embedding Space Visualization (t-SNE)
from sklearn.manifold import TSNE
X_embed = results.get('X')
if X_embed is not None and len(X_embed) > 0:
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X_embed)

    plt.figure(figsize=(10,8))
    for cls in set(results['y']):
        mask = np.array(results['y']) == cls
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Class {cls}', alpha=0.7)
    plt.title('Multi-Modal Embedding Clustering (t-SNE)')
    plt.legend()
    plt.show()"""))
        
    # 4. Model Diagnostics & Error Analysis
    nb.cells.append(new_markdown_cell("## 🤖 3. Model Performance & Error Analysis"))
    nb.cells.append(new_code_cell("""# Confusion Matrix & Classification Report
y_true, y_pred = results.get('y_test', results.get('y')), results.get('y_pred')
if y_true is not None and y_pred is not None:
    print(classification_report(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()"""))
    
    # 5. Business Impact Simulation
    nb.cells.append(new_markdown_cell("## 💼 4. Business Impact & Recommendations"))
    nb.cells.append(new_code_cell("""# ROI Simulation based on Success Metric
metric = config.get('business_context', {}).get('success_metric', 'Accuracy')
acc = results.get('final_accuracy', 0.0)
if acc:
    print(f'Estimated {metric}: {acc:.2%}')
    print('Recommendation: ' + ('Deploy to production.' if acc > 0.90 else 'Collect more data or adjust domain extractor.' if acc > 0.75 else 'Re-evaluate feature extraction pipeline.'))"""))
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"📓 Advanced notebook saved to: {output_path}")
