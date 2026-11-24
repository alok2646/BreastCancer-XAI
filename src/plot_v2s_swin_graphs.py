# ==========================================================
# EfficientNetV2-S + Swin-Tiny
# Training Graphs
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

HISTORY_FILE = (
    "results/v2s_swin/"
    "v2s_swin_training_history.csv"
)

OUTPUT_DIR = (
    "results/v2s_swin/"
    "graphs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD TRAINING HISTORY
# ==========================================================

history = pd.read_csv(
    HISTORY_FILE
)

print("\nTraining history loaded:")
print(history.to_string())


# ==========================================================
# EPOCH LABEL
# ==========================================================

# Global epoch number
history["global_epoch"] = range(
    1,
    len(history) + 1
)


# ==========================================================
# LOSS GRAPH
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history["global_epoch"],
    history["train_loss"],
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history["global_epoch"],
    history["val_loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss vs Validation Loss\n"
    "EfficientNetV2-S + Swin-Tiny"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    history["global_epoch"]
)

plt.tight_layout()


loss_path = os.path.join(
    OUTPUT_DIR,
    "v2s_swin_training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# ACCURACY GRAPH
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history["global_epoch"],
    history["train_accuracy"],
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history["global_epoch"],
    history["val_accuracy"],
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Training Accuracy vs Validation Accuracy\n"
    "EfficientNetV2-S + Swin-Tiny"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    history["global_epoch"]
)

plt.tight_layout()


accuracy_path = os.path.join(
    OUTPUT_DIR,
    "v2s_swin_training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# COMPLETED
# ==========================================================

print("\n")
print("=" * 65)
print("V2S + SWIN TRAINING GRAPHS SAVED")
print("=" * 65)

print(
    "\nLoss Graph:"
)

print(
    loss_path
)

print(
    "\nAccuracy Graph:"
)

print(
    accuracy_path
)

print(
    "\nTraining Graph generation completed successfully!"
)

print("=" * 65)