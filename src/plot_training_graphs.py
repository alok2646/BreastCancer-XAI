import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

HISTORY_FILE = "saved_models/hybrid_stable_training_history.csv"

OUTPUT_DIR = "results/hybrid_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD HISTORY
# ============================================================

if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(
        f"Training history not found:\n{HISTORY_FILE}"
    )

df = pd.read_csv(HISTORY_FILE)

print("\nTraining history loaded:")
print(df)


# ============================================================
# CREATE GLOBAL EPOCH NUMBER
# ============================================================

df["global_epoch"] = range(1, len(df) + 1)


# ============================================================
# GRAPH 1: LOSS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    df["global_epoch"],
    df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    df["global_epoch"],
    df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Hybrid EfficientNet-B3 + Swin-Tiny\n"
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2: ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    df["global_epoch"],
    df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    df["global_epoch"],
    df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Hybrid EfficientNet-B3 + Swin-Tiny\n"
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("TRAINING GRAPHS SAVED")
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

print("=" * 65)