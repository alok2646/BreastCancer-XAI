"""
Hybrid EfficientNet-B3 + Swin-Tiny
Breast Cancer Classification Training

Patient-Level Dataset
Train / Validation / Test

Model:
    EfficientNet-B3 + Swin-Tiny
"""

import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.amp import autocast, GradScaler

from hybrid_model import HybridBreastCancerNet


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

NUM_EPOCHS = 30

# RTX 3050 memory-safe batch size
BATCH_SIZE = 4

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

IMAGE_SIZE = 224

DATASET_ROOT = "dataset_patient_split"

SAVE_DIR = "saved_models"

BEST_MODEL = os.path.join(
    SAVE_DIR,
    "hybrid_best_model.pth"
)

FINAL_MODEL = os.path.join(
    SAVE_DIR,
    "hybrid_final_model.pth"
)

CHECKPOINT = os.path.join(
    SAVE_DIR,
    "hybrid_training_checkpoint.pth"
)

HISTORY_FILE = os.path.join(
    SAVE_DIR,
    "hybrid_training_history.csv"
)


os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
# Reproducibility
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "val"
)

TEST_DIR = os.path.join(
    DATASET_ROOT,
    "test"
)


# ==========================================================
# Transforms
# Same normalization as existing project
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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


# ==========================================================
# Load Patient-Level Datasets
# ==========================================================

print("=" * 70)
print("HYBRID BREAST CANCER TRAINING")
print("=" * 70)

print("\nDevice       :", DEVICE)

print("Dataset Root :", os.path.abspath(DATASET_ROOT))

print("Batch Size   :", BATCH_SIZE)

print("Epochs       :", NUM_EPOCHS)

print("Learning Rate:", LEARNING_RATE)

print("=" * 70)


train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)


# ==========================================================
# Dataset Information
# ==========================================================

print("\nClasses:")
print(train_dataset.classes)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

print("\nTrain Images:", len(train_dataset))

print("Val Images  :", len(val_dataset))

print("Test Images :", len(test_dataset))


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

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


print("\nTrain Batches:", len(train_loader))

print("Val Batches  :", len(val_loader))

print("Test Batches :", len(test_loader))

print("=" * 70)


# ==========================================================
# Create Hybrid Model
# ==========================================================

print("\nLoading Hybrid Model...")

model = HybridBreastCancerNet(

    num_classes=2,

    dropout=0.30,

    pretrained=True

)

model = model.to(DEVICE)


print("\nHybrid Model Ready!")

print(
    "EfficientNet Features:",
    model.efficient_features
)

print(
    "Swin Features:",
    model.swin_features
)

print(
    "Fusion Features:",
    model.fusion_features
)


# ==========================================================
# Loss
# ==========================================================

criterion = nn.CrossEntropyLoss()


# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ==========================================================
# Scheduler
# ==========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3
)


# ==========================================================
# AMP
# ==========================================================

use_amp = torch.cuda.is_available()

scaler = GradScaler(
    "cuda",
    enabled=use_amp
)


# ==========================================================
# Training History
# ==========================================================

history = []

best_val_accuracy = 0.0

start_epoch = 0


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()


    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        # Clear temporary tensors
        del images
        del labels
        del outputs
        del loss


    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100.0 *
        correct /
        total
    )

    epoch_time = (
        time.time() -
        start_time
    )


    return (
        epoch_loss,
        epoch_accuracy,
        epoch_time
    )


# ==========================================================
# Validation
# ==========================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            with autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        running_loss /
        len(val_loader)
    )

    val_accuracy = (
        100.0 *
        correct /
        total
    )


    return (
        val_loss,
        val_accuracy
    )


# ==========================================================
# Training Loop
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)


for epoch in range(
    start_epoch,
    NUM_EPOCHS
):


    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )


    # -------------------------
    # Training
    # -------------------------

    train_loss, train_accuracy, epoch_time = (
        train_one_epoch()
    )


    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_accuracy = (
        validate()
    )


    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # -------------------------
    # Print Results
    # -------------------------

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.2e}"
    )

    print(
        f"Time       : {epoch_time:.2f} sec"
    )


    # -------------------------
    # History
    # -------------------------

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "learning_rate": current_lr

    })


    # -------------------------
    # Save Best Model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(

            model.state_dict(),

            BEST_MODEL

        )


        print(
            "\n*** BEST HYBRID MODEL SAVED ***"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy:.2f}%"
        )


    # -------------------------
    # Clear CUDA Cache
    # -------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ==========================================================
# Save Final Model
# ==========================================================

torch.save(

    model.state_dict(),

    FINAL_MODEL

)


# ==========================================================
# Save Training History
# ==========================================================

import csv

with open(
    HISTORY_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(

        f,

        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate"
        ]
    )

    writer.writeheader()

    writer.writerows(history)


# ==========================================================
# Training Complete
# ==========================================================

print("\n")
print("=" * 70)
print("HYBRID TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"\nBest Model  : {BEST_MODEL}"
)

print(
    f"Final Model : {FINAL_MODEL}"
)

print(
    f"History     : {HISTORY_FILE}"
)

print("=" * 70)