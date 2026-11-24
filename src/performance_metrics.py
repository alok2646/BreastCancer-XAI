import os
import matplotlib.pyplot as plt

# ============================================================
# Breast Cancer XAI - Performance Metrics
# ============================================================

OUTPUT_DIR = "outputs/performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Final Evaluation Metrics
metrics = {
    "Accuracy": 92.89,
    "Precision": 88.68,
    "Recall": 85.95,
    "F1 Score": 87.29,
    "ROC-AUC": 97.11
}

names = list(metrics.keys())
values = list(metrics.values())

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(names, values)

# Write values on bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f"{value:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.ylim(0, 105)

plt.title(
    "Breast Cancer Classification Performance",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Performance (%)")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "performance_metrics.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("✓ Performance Metrics Figure Saved")
print(output_path)
print("=" * 60)

# ============================================================
# Print Results
# ============================================================

print("\nFinal Evaluation Metrics")
print("=" * 35)

for key, value in metrics.items():
    print(f"{key:<12} : {value:.2f}%")