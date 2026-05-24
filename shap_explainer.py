import shap
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os

def generate_shap_explanations(model, X_train, X_test, model_name, dataset_id):
    """
    Generates SHAP plots and logs them to W&B.
    """
    print(f"[SHAP] Generating explanations for {model_name}...")
    
    # 1. Initialize the correct Explainer
    try:
        if model_name in ['rf', 'et_clf', 'et_reg', 'gb', 'gb_reg', 'xgb_clf', 'xgb_reg', 'lgbm_clf', 'lgbm_reg', 'ada_clf', 'ada_reg', 'bag_clf', 'bag_reg']:
            explainer = shap.TreeExplainer(model)
            # Sample test data if it's too large to prevent OOM
            X_explain = shap.sample(X_test, 100) if len(X_test) > 100 else X_test
            shap_values = explainer.shap_values(X_explain)
            
            # Handle binary vs multiclass output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Take the positive class for binary
                
        elif model_name in ['logistic', 'ridge', 'lasso', 'elastic', 'sgd_reg', 'sgd_clf']:
            # For linear models, we can use the linear explainer or treat as black box
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(model.predict, background)
            X_explain = shap.sample(X_test, 50) if len(X_test) > 50 else X_test
            shap_values = explainer.shap_values(X_explain)
            
        else:
            # Fallback for MLP, SVM, KNN (KernelExplainer is slow, keep sample small)
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(model.predict, background)
            X_explain = shap.sample(X_test, 30) if len(X_test) > 30 else X_test
            shap_values = explainer.shap_values(X_explain)

    except Exception as e:
        import traceback
        print(f"[SHAP] Failed to generate explanations for {model_name}: {e}")
        traceback.print_exc()
        return False, []

    # 2. Generate Plots
    os.makedirs("shap_plots", exist_ok=True)
    
    # Plot A: Global Summary (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False, plot_type="dot")
    summary_path = f"shap_plots/{dataset_id}_{model_name}_summary.png"
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    # Plot B: Feature Importance (Bar)
    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False, plot_type="bar")
    bar_path = f"shap_plots/{dataset_id}_{model_name}_bar.png"
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()

    # 3. Log to W&B
    wandb.log({
        f"shap/{dataset_id}_summary": wandb.Image(summary_path),
        f"shap/{dataset_id}_importance": wandb.Image(bar_path)
    })
    
    # Extract Top 3 Features
    top_3_features = ["Unknown"]
    try:
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
        elif len(shap_values.shape) > 2: # handle multiclass arrays
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        top_idx = np.argsort(mean_abs_shap)[-3:][::-1]
        if hasattr(X_explain, 'columns'):
            top_3_features = X_explain.columns[top_idx].tolist()
        else:
            top_3_features = [f"Feature {i}" for i in top_idx]
    except Exception as e:
        print(f"[SHAP] Could not extract top features: {e}")
        
    print(f"[SHAP] Successfully logged plots for {model_name}.")
    return True, top_3_features
