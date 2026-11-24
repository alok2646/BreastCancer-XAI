# ==========================================================
# Breast Cancer Model - Final Test Evaluation
# IEEE Paper Evaluation Pipeline
# ==========================================================

import os
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt

from model import BreastCancerNet
from dataset_patient import test_loader, test_dataset

# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/patient_best_model.pth"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================================
# Device Information
# ==========================================================

print("=" * 60)
print("BREAST CANCER MODEL EVALUATION")
print("=" * 60)

print("Using Device:", DEVICE)
print("Model Path:", MODEL_PATH)

print("=" * 60)

# ==========================================================
# Load Trained Model
# ==========================================================

print("\nLoading trained model...")

model = BreastCancerNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model Loaded Successfully!")


# ==========================================================
# Store Predictions
# ==========================================================

all_labels = []

all_predictions = []

all_probabilities = []


# ==========================================================
# Test Evaluation
# ==========================================================

print("\nStarting Test Evaluation...")
print("=" * 60)


with torch.no_grad():

    for images, labels in test_loader:

        # Move data to device
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # Forward Pass
        outputs = model(images)


        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Prediction
        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        # Store True Labels
        all_labels.extend(
            labels.cpu().numpy()
        )


        # Store Predictions
        all_predictions.extend(
            predictions.cpu().numpy()
        )


        # Store Malignant Probability
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


print("Test Evaluation Completed Successfully!")
print("=" * 60)


# ==========================================================
# Convert to NumPy Arrays
# ==========================================================

all_labels = np.array(all_labels)

all_predictions = np.array(all_predictions)

all_probabilities = np.array(all_probabilities)


# ==========================================================
# Calculate Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)


# ==========================================================
# Print Final Metrics
# ==========================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy  : {accuracy * 100:.2f}%")

print(f"Precision      : {precision * 100:.2f}%")

print(f"Recall         : {recall * 100:.2f}%")

print(f"F1 Score       : {f1 * 100:.2f}%")

print("=" * 60)


# ==========================================================
# Classification Report
# ==========================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4,
        zero_division=0
    )
)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(cm)


# ==========================================================
# ROC Curve and AUC Score
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)

roc_auc = auc(
    fpr,
    tpr
)


print("\n" + "=" * 60)

print(f"AUC SCORE : {roc_auc:.4f}")

print("=" * 60)


# ==========================================================
# Save Confusion Matrix Values
# ==========================================================

np.savetxt(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.txt"
    ),
    cm,
    fmt="%d"
)


# ==========================================================
# Plot Confusion Matrix
# ==========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

classes = test_dataset.classes

plt.xticks(
    np.arange(len(classes)),
    classes
)

plt.yticks(
    np.arange(len(classes)),
    classes
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Plot ROC Curve
# ==========================================================

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Save Metrics
# ==========================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "evaluation_results.txt"
)


with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "BREAST CANCER MODEL TEST EVALUATION\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Test Accuracy : {accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Precision     : {precision * 100:.4f}%\n"
    )

    f.write(
        f"Recall        : {recall * 100:.4f}%\n"
    )

    f.write(
        f"F1 Score      : {f1 * 100:.4f}%\n"
    )

    f.write(
        f"AUC Score     : {roc_auc:.6f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes,
            digits=4,
            zero_division=0
        )
    )


# ==========================================================
# Completion Message
# ==========================================================

print("\n" + "=" * 60)

print("EVALUATION COMPLETED SUCCESSFULLY!")

print("=" * 60)

print(
    f"Results Saved in: {RESULTS_DIR}"
)

print("\nGenerated Files:")

print("1. evaluation_results.txt")

print("2. confusion_matrix.txt")

print("3. confusion_matrix.png")

print("4. roc_curve.png")

print("=" * 60)