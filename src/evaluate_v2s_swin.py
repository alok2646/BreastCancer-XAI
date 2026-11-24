# ==========================================================
# Breast Cancer XAI
# EfficientNetV2-S + Swin-Tiny
# FINAL TEST EVALUATION
# ==========================================================

import os
import csv
import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/hybrid_v2s_swin_best_model.pth"

TEST_DIR = "dataset_patient_split/test"

OUTPUT_DIR = "results/v2s_swin/evaluation"

BATCH_SIZE = 4

NUM_WORKERS = 0

CLASS_NAMES = [
    "benign",
    "malignant"
]

IMAGE_SIZE = 224


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# HEADER
# ==========================================================

print("\n" + "=" * 70)
print("HYBRID EFFICIENTNET-V2-S + SWIN-TINY")
print("FINAL TEST EVALUATION")
print("=" * 70)

print("Device       :", DEVICE)
print("Model        :", MODEL_PATH)
print("Test Dataset :", TEST_DIR)
print("Batch Size   :", BATCH_SIZE)

print("=" * 70)


# ==========================================================
# TEST TRANSFORM
# ==========================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================================
# LOAD TEST DATASET
# ==========================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nClasses       :", test_dataset.classes)
print("Class Mapping :", test_dataset.class_to_idx)

print(
    "Test Images   :",
    len(test_dataset)
)

print(
    "Test Batches  :",
    len(test_loader)
)

print("=" * 70)


# ==========================================================
# IMPORT HYBRID MODEL
# ==========================================================

print("\nLoading Hybrid V2-S + Swin Model...")


from hybrid_v2s_swin import HybridV2SSwinNet


model = HybridV2SSwinNet(
    num_classes=2
).to(DEVICE)


# ==========================================================
# LOAD BEST CHECKPOINT
# ==========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ----------------------------------------------------------
# Support both normal state_dict and checkpoint dictionary
# ----------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)


model.eval()


print("Hybrid model loaded successfully!")

print(
    "EfficientNetV2-S features : 1280"
)

print(
    "Swin-Tiny features        : 768"
)

print(
    "Fusion features           : 2048"
)

print("=" * 70)


# ==========================================================
# LOSS FUNCTION
# ==========================================================

criterion = nn.CrossEntropyLoss()


# ==========================================================
# STORAGE
# ==========================================================

all_labels = []

all_predictions = []

all_probabilities = []

running_loss = 0.0


# ==========================================================
# TEST EVALUATION
# ==========================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION STARTED")
print("=" * 70)


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # --------------------------------------------------
        # Forward Pass
        # --------------------------------------------------

        outputs = model(
            images
        )


        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        running_loss += loss.item()


        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of malignant class
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            batch_index == 1
            or batch_index % 1000 == 0
            or batch_index == len(test_loader)
        ):

            print(
                f"Processed batches: "
                f"{batch_index}/{len(test_loader)}"
            )


# ==========================================================
# CONVERT TO NUMPY
# ==========================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ==========================================================
# FINAL LOSS
# ==========================================================

test_loss = (
    running_loss
    /
    len(test_loader)
)


# ==========================================================
# METRICS
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


roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL V2S + SWIN TEST RESULTS")
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


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ==========================================================
# SAVE CLASSIFICATION REPORT
# ==========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EFFICIENTNETV2-S + SWIN-TINY\n"
    )

    file.write(
        "FINAL TEST CLASSIFICATION REPORT\n\n"
    )

    file.write(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ==========================================================
# SAVE CONFUSION MATRIX AS CSV
# ==========================================================

cm_csv_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

with open(
    cm_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "",
        "Predicted Benign",
        "Predicted Malignant"
    ])

    writer.writerow([
        "Actual Benign",
        cm[0][0],
        cm[0][1]
    ])

    writer.writerow([
        "Actual Malignant",
        cm[1][0],
        cm[1][1]
    ])


# ==========================================================
# CONFUSION MATRIX IMAGE
# ==========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - EfficientNetV2-S + Swin-Tiny"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)


threshold = cm.max() / 2.0


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )


plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()


cm_image_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# ROC CURVE
# ==========================================================

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
    linewidth=2,
    label=f"ROC-AUC = {roc_auc:.4f}"
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
    "ROC Curve - EfficientNetV2-S + Swin-Tiny"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


roc_path = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# SAVE METRICS CSV
# ==========================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "v2s_swin_test_metrics.csv"
)

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Test Loss",
        f"{test_loss:.6f}"
    ])

    writer.writerow([
        "Accuracy",
        f"{accuracy:.6f}"
    ])

    writer.writerow([
        "Precision",
        f"{precision:.6f}"
    ])

    writer.writerow([
        "Recall",
        f"{recall:.6f}"
    ])

    writer.writerow([
        "F1 Score",
        f"{f1:.6f}"
    ])

    writer.writerow([
        "ROC-AUC",
        f"{roc_auc:.6f}"
    ])


# ==========================================================
# SAVE ALL PREDICTIONS
# ==========================================================

predictions_path = os.path.join(
    OUTPUT_DIR,
    "v2s_swin_test_predictions.csv"
)

with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "index",
        "true_label",
        "predicted_label",
        "malignant_probability"
    ])

    for index in range(
        len(all_labels)
    ):

        writer.writerow([
            index,
            CLASS_NAMES[
                all_labels[index]
            ],
            CLASS_NAMES[
                all_predictions[index]
            ],
            f"{all_probabilities[index]:.6f}"
        ])


# ==========================================================
# FINAL OUTPUT
# ==========================================================

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
    cm_image_path
)

print(
    "ROC Curve         :",
    roc_path
)

print(
    "Predictions CSV   :",
    predictions_path
)

print("=" * 70)