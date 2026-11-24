"""
FINAL TEST EVALUATION
Hybrid EfficientNet-B3 + Swin-Tiny

Uses:
    saved_models/hybrid_stable_best_model.pth

Test set:
    dataset_patient_split/test

Metrics:
    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC
    Confusion Matrix
    Classification Report
"""

import os
import csv
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import matplotlib.pyplot as plt

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

BATCH_SIZE = 4

NUM_WORKERS = 0

DATASET_ROOT = "dataset_patient_split"

TEST_DIR = os.path.join(
    DATASET_ROOT,
    "test"
)

MODEL_PATH = os.path.join(
    "saved_models",
    "hybrid_stable_best_model.pth"
)

OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CLASS_NAMES = [
    "benign",
    "malignant"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("HYBRID MODEL - FINAL TEST EVALUATION")
print("=" * 70)

print(
    "Device       :",
    DEVICE
)

print(
    "Model        :",
    MODEL_PATH
)

print(
    "Test Dataset :",
    TEST_DIR
)

print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# TEST TRANSFORM
# Same normalization used by training
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(

    root=TEST_DIR,

    transform=test_transform
)


print(
    "\nClasses:",
    test_dataset.classes
)

print(
    "Class mapping:",
    test_dataset.class_to_idx
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


print(
    "Test batches:",
    len(test_loader)
)

print("=" * 70)


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

print("\nLoading Hybrid Model...")

model = HybridBreastCancerNet(

    num_classes=2,

    dropout=0.30,

    pretrained=False

)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)


model.load_state_dict(
    checkpoint,
    strict=True
)


model = model.to(DEVICE)

model.eval()


print(
    "Hybrid model loaded successfully!"
)

print(
    "EfficientNet features:",
    model.efficient_features
)

print(
    "Swin features:",
    model.swin_features
)

print(
    "Fusion features:",
    model.fusion_features
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# STORAGE
# ============================================================

all_labels = []

all_predictions = []

all_probabilities = []

running_loss = 0.0


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION STARTED")
print("=" * 70)


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # Forward pass
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        running_loss += loss.item()


        # Probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Prediction
        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        # Store
        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


        # Progress
        if (
            (batch_index + 1) % 1000 == 0
            or
            batch_index == 0
        ):

            print(
                f"Processed batches: "
                f"{batch_index + 1}/{len(test_loader)}"
            )


# ============================================================
# NUMPY ARRAYS
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# TEST LOSS
# ============================================================

test_loss = (
    running_loss /
    len(test_loader)
)


# ============================================================
# METRICS
# ============================================================

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


roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL HYBRID TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss      : {test_loss:.4f}"
)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Precision      : {precision * 100:.2f}%"
)

print(
    f"Recall         : {recall * 100:.2f}%"
)

print(
    f"F1 Score       : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print("=" * 70)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    all_labels,

    all_predictions,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0

)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        "Hybrid EfficientNet-B3 + Swin-Tiny\n"
    )

    file.write(
        "Final Test Evaluation\n\n"
    )

    file.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.4f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.4f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.4f}%\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.6f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_test_metrics.csv"
)


with open(
    metrics_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Test Loss",
        test_loss
    ])

    writer.writerow([
        "Accuracy",
        accuracy * 100
    ])

    writer.writerow([
        "Precision",
        precision * 100
    ])

    writer.writerow([
        "Recall",
        recall * 100
    ])

    writer.writerow([
        "F1 Score",
        f1 * 100
    ])

    writer.writerow([
        "ROC-AUC",
        roc_auc
    ])


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "Hybrid Model - Confusion Matrix"
)

plt.colorbar()


plt.xticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Hybrid Model (AUC = {roc_auc:.4f})"
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
    "Hybrid Model - ROC Curve"
)

plt.legend()

plt.tight_layout()


roc_path = os.path.join(
    OUTPUT_DIR,
    "hybrid_roc_curve.png"
)


plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)

print(
    "Metrics CSV       :",
    metrics_path
)

print(
    "Classification    :",
    report_path
)

print(
    "Confusion Matrix  :",
    cm_path
)

print(
    "ROC Curve         :",
    roc_path
)

print("=" * 70)