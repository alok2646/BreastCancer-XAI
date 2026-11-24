import os
import numpy as np
import pandas as pd

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
    confusion_matrix
)

from hybrid_b3_swin_multiscale import HybridB3SwinMultiScaleNet


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "dataset_base_final"

MODEL_PATH = "saved_models/b3_swin_multiscale_best_model.pth"

RESULT_DIR = "results/b3_swin_multiscale"

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_CLASSES = 2

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("B3 + SWIN ATTENTION FUSION — FINAL TEST EVALUATION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

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
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

print("\nTest samples:", len(test_dataset))

print(
    "Classes:",
    test_dataset.class_to_idx
)


# ============================================================
# TEST LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = HybridB3SwinMultiScaleNet(
    num_classes=NUM_CLASSES,
    dropout=0.30
).to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

print("✓ Best model loaded successfully")


# ============================================================
# EVALUATION
# ============================================================

model.eval()

criterion = nn.CrossEntropyLoss()

running_loss = 0.0

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

test_loss = (
    running_loss /
    len(test_dataset)
)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# CONFUSION MATRIX VALUES
# ============================================================

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


# ============================================================
# SPECIFICITY
# ============================================================

specificity = TN / (TN + FP)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss       : {test_loss:.4f}"
)

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Precision       : {precision * 100:.2f}%"
)

print(
    f"Recall          : {recall * 100:.2f}%"
)

print(
    f"Specificity     : {specificity * 100:.2f}%"
)

print(
    f"F1 Score        : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC         : {roc_auc:.4f}"
)


print("\nConfusion Matrix:")

print(cm)

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{

    "test_loss": test_loss,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "specificity": specificity,

    "f1_score": f1,

    "roc_auc": roc_auc,

    "TN": TN,

    "FP": FP,

    "FN": FN,

    "TP": TP

}])


metrics_path = os.path.join(
    RESULT_DIR,
    "b3_swin_multiscale_test_metrics.csv"
)


metrics.to_csv(
    metrics_path,
    index=False
)


print(
    "\n✓ Metrics saved:"
)

print(
    metrics_path
)


print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)