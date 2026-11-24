import os
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from efficientnet_v2s_basepaper import EfficientNetV2SBasePaper


# ============================================================
# BASE PAPER TRAINING CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset_base_split")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = Path("saved_models")
RESULT_DIR = Path("results/efficientnet_v2s_basepaper")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 0

EPOCHS = 100

LEARNING_RATE = 0.0001

L1_LAMBDA = 0.006
L2_LAMBDA = 0.006


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BASE PAPER EFFICIENTNETV2S TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TRANSFORMS
# ============================================================
#
# Base paper input:
# 50 x 50
#
# Image normalization:
# [0, 1]
#
# ToTensor() converts uint8 [0,255]
# to float [0,1].
#
# IMPORTANT:
# No ImageNet mean/std normalization here.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)


print("\nClasses:")
print(train_dataset.class_to_idx)

print("\nDataset:")
print("Train:", len(train_dataset))
print("Test :", len(test_dataset))
print("Total:", len(train_dataset) + len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = EfficientNetV2SBasePaper(
    num_classes=2
).to(device)


# ============================================================
# LOSS
# ============================================================
#
# Paper:
# categorical cross-entropy
#
# PyTorch CrossEntropyLoss is the equivalent
# for class-index labels and internally applies
# log-softmax.
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================
#
# Paper:
# Adamax
# learning rate = 0.0001
# ============================================================

optimizer = torch.optim.Adamax(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# L1 + L2 REGULARIZATION
# ============================================================

def regularization_loss(model):

    l1_loss = torch.tensor(
        0.0,
        device=device
    )

    l2_loss = torch.tensor(
        0.0,
        device=device
    )

    # Paper regularization is applied to
    # the fully connected layer.

    l1_loss = torch.sum(
        torch.abs(model.fc.weight)
    )

    l2_loss = torch.sum(
        model.fc.weight ** 2
    )

    return (
        L1_LAMBDA * l1_loss
        +
        L2_LAMBDA * l2_loss
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []

best_train_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = criterion(
            outputs,
            labels
        )

        reg_loss = regularization_loss(
            model
        )

        loss = (
            classification_loss
            + reg_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        100.0 * correct / total
    )

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Accuracy: {epoch_accuracy:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "train_accuracy": epoch_accuracy
    })


    # ========================================================
    # SAVE BEST TRAINING MODEL
    # ========================================================

    if epoch_accuracy > best_train_accuracy:

        best_train_accuracy = epoch_accuracy

        torch.save(
            model.state_dict(),
            MODEL_DIR /
            "efficientnet_v2s_basepaper_best.pth"
        )

        print(
            f"  -> Best model saved "
            f"({best_train_accuracy:.2f}%)"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_DIR /
    "efficientnet_v2s_basepaper_final.pth"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = (
    RESULT_DIR /
    "efficientnet_v2s_basepaper_training_history.csv"
)

with open(
    history_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "loss",
            "train_accuracy"
        ]
    )

    writer.writeheader()

    writer.writerows(history)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best training accuracy: "
    f"{best_train_accuracy:.2f}%"
)

print(
    "Best model:",
    MODEL_DIR /
    "efficientnet_v2s_basepaper_best.pth"
)

print(
    "Final model:",
    MODEL_DIR /
    "efficientnet_v2s_basepaper_final.pth"
)

print(
    "History:",
    history_file
)