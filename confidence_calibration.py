import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

# 1. Load Data
df = pd.read_csv("confidence_data.csv")

# 2. Normalize Actual Utility to [0, 1] so it matches the Confidence probability space
min_u, max_u = df['actual_utility'].min(), df['actual_utility'].max()
if max_u > min_u:
    df['norm_utility'] = (df['actual_utility'] - min_u) / (max_u - min_u)
else:
    df['norm_utility'] = 1.0

# 3. Define Weight Grid to Test
weight_grid = [
    (0.60, 0.20, 0.20)
]

def calculate_ece(confidences, actuals, n_bins=5):
    """Calculates Expected Calibration Error for continuous utilities."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    bin_data = []
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        # Inclusive upper bound for the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
            
        if in_bin.sum() > 0:
            bin_conf = confidences[in_bin].mean()
            bin_acc = actuals[in_bin].mean()
            bin_weight = in_bin.sum() / total_samples
            
            ece += np.abs(bin_acc - bin_conf) * bin_weight
            bin_data.append((bin_conf, bin_acc, in_bin.sum()))
            
    return ece, bin_data

# 4. Run Calibration
best_weights = (0.60, 0.20, 0.20)
confs = (best_weights[0] * df['c_sim']) + (best_weights[1] * df['c_cons']) + (best_weights[2] * df['c_agree'])
confs = np.clip(confs, 0.0, 1.0).values
actuals = df['norm_utility'].values

best_ece, best_bin_data = calculate_ece(confs, actuals, n_bins=5)

print(f"\n✅ BEST WEIGHTS: {best_weights} with ECE: {best_ece:.4f}")

# 5. Plot Reliability Curve for Best Weights
confs = (best_weights[0] * df['c_sim']) + (best_weights[1] * df['c_cons']) + (best_weights[2] * df['c_agree'])
confs = np.clip(confs, 0.0, 1.0).values

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (y=x)")

bin_confs = [b[0] for b in best_bin_data]
bin_accs = [b[1] for b in best_bin_data]
plt.plot(bin_confs, bin_accs, "o-", color="blue", label=f"MetaAutoML (ECE={best_ece:.4f})")

plt.xlabel("Confidence Score C(D)")
plt.ylabel("Actual Normalized Utility")
plt.title(f"Reliability Diagram (Best Weights: {best_weights})")
plt.legend()
plt.grid(True)
plt.savefig("reliability_curve.png")
print("Saved reliability_curve.png")
plt.show()

# 6. Log to W&B
wandb.init(project="MetaAutoML-Confidence", name="calibration_search")
wandb.log({
    "best_ece": best_ece,
    "best_w_sim": best_weights[0],
    "best_w_cons": best_weights[1],
    "best_w_agree": best_weights[2],
    "reliability_curve": wandb.Image("reliability_curve.png")
})
wandb.finish()
