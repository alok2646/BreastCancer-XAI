import os
import pandas as pd

# ============================================================
# Final Results Summary
# ============================================================

OUTPUT_DIR = "outputs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        "92.89%",
        "88.68%",
        "85.95%",
        "87.29%",
        "97.11%"
    ]
}

df = pd.DataFrame(results)

print("\nFinal Results Summary")
print("=" * 35)
print(df)

csv_path = os.path.join(
    OUTPUT_DIR,
    "results_summary.csv"
)

df.to_csv(csv_path, index=False)

print("\n✓ Results Summary Saved")
print(csv_path)