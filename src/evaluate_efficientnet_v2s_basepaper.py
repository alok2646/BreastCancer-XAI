import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from efficientnet_v2s_basepaper import EfficientNetV2SBasePaper


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path("dataset_base_split")
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path(
    "saved_models/efficientnet_v2s_basepaper_best.pth"
)

RESULT_DIR = Path(
    "results/efficientnet_v2s_basepaper"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BASE PAPER EFFICIENTNETV2S TEST EVALUATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TEST TRANSFORM
# ============================================================
# 50x50 + [0,1]
# ============================================================

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nClasses:")
print(test_dataset.class_to_idx)

print("\nTest images:", len(test_dataset))


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = EfficientNetV2SBasePaper(
    num_classes=2
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("\nBest model loaded:")
print(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

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
            probabilities[:, 1]
            .cpu()
            .numpy()
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# SPECIFICITY
# ============================================================

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision        : {precision * 100:.2f}%"
)

print(
    f"Recall/Sensitivity: {recall * 100:.2f}%"
)

print(
    f"Specificity      : {specificity * 100:.2f}%"
)

print(
    f"F1-Score         : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC          : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = (
    RESULT_DIR /
    "efficientnet_v2s_basepaper_test_metrics.csv"
)

with open(
    metrics_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Test Accuracy",
        accuracy
    ])

    writer.writerow([
        "Precision",
        precision
    ])

    writer.writerow([
        "Recall Sensitivity",
        recall
    ])

    writer.writerow([
        "Specificity",
        specificity
    ])

    writer.writerow([
        "F1 Score",
        f1
    ])

    writer.writerow([
        "ROC AUC",
        roc_auc
    ])

    writer.writerow([
        "TN",
        tn
    ])

    writer.writerow([
        "FP",
        fp
    ])

    writer.writerow([
        "FN",
        fn
    ])

    writer.writerow([
        "TP",
        tp
    ])


print("\nMetrics saved:")
print(metrics_file)

print("\nEvaluation completed successfully!")