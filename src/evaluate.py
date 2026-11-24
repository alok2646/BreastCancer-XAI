import os
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model import BreastCancerNet
from dataset_patient import test_loader
# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2

MODEL_PATH = "saved_models/patient_best_model.pth"
CLASS_NAMES = [
    "benign",
    "malignant"
]

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(" Breast Cancer Model Evaluation ")
print("=" * 60)

print("Device      :", DEVICE)
print("Model Path  :", MODEL_PATH)
print("Output Path :", OUTPUT_DIR)
print("=" * 60)

# ==========================================================
# Load Model
# ==========================================================

model = BreastCancerNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("\nModel Loaded Successfully!")
print("=" * 60)

# ==========================================================
# Evaluation Variables
# ==========================================================

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []
all_probabilities = []

running_loss = 0.0

# ==========================================================
# Evaluate Model
# ==========================================================

print("Evaluating Model...")
print("=" * 60)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        probabilities = torch.softmax(outputs, dim=1)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

print("Evaluation Completed!")
print("=" * 60)

# ==========================================================
# Calculate Metrics
# ==========================================================

test_loss = running_loss / len(test_loader)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

auc = roc_auc_score(
    all_labels,
    all_probabilities
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss      : {test_loss:.4f}")
print(f"Accuracy       : {accuracy*100:.2f}%")
print(f"Precision      : {precision*100:.2f}%")
print(f"Recall         : {recall*100:.2f}%")
print(f"F1 Score       : {f1*100:.2f}%")
print(f"ROC-AUC Score  : {auc:.4f}")

print("=" * 60)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()
plt.close()

print("\nConfusion Matrix Saved Successfully!")

print("=" * 60)