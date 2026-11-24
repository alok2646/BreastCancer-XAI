"""
STABLE HYBRID TRAINING
EfficientNet-B3 + Swin-Tiny
Breast Cancer Classification

Baseline:
    hybrid_best_model.pth
    Best Validation Accuracy = 89.50%

Strategy:
    Stage 1 -> Freeze both backbones, train fusion classifier
    Stage 2 -> Unfreeze only final backbone blocks, fine-tune carefully

Safety:
    - NaN / Inf detection
    - Gradient clipping
    - Lower fine-tuning LR
    - Best baseline checkpoint is NEVER overwritten
    - Early stopping
"""

import os
import csv
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


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

DATASET_ROOT = "dataset_patient_split"

SAVE_DIR = "saved_models"

# ------------------------------------------------------------
# IMPORTANT:
# This is your already trained 89.50% baseline.
# DO NOT overwrite this file.
# ------------------------------------------------------------

BASELINE_MODEL = os.path.join(
    SAVE_DIR,
    "hybrid_best_model.pth"
)

# New files only
STABLE_BEST_MODEL = os.path.join(
    SAVE_DIR,
    "hybrid_stable_best_model.pth"
)

STABLE_FINAL_MODEL = os.path.join(
    SAVE_DIR,
    "hybrid_stable_final_model.pth"
)

HISTORY_FILE = os.path.join(
    SAVE_DIR,
    "hybrid_stable_training_history.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Stage 1:
# Only fusion classifier is trainable.
STAGE1_EPOCHS = 3

STAGE1_LR = 1e-4


# Stage 2:
# Carefully fine-tune final blocks.
STAGE2_EPOCHS = 5

STAGE2_LR = 1e-5


WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

PATIENCE = 3


# Your existing best validation accuracy
BASELINE_VAL_ACC = 89.50


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "val"
)


# ============================================================
# TRANSFORMS
# ============================================================

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


# ============================================================
# DATASETS
# ============================================================

print("=" * 70)
print("STABLE HYBRID TRAINING")
print("EfficientNet-B3 + Swin-Tiny")
print("=" * 70)

print(
    "\nDevice:",
    DEVICE
)

print(
    "Baseline:",
    f"{BASELINE_VAL_ACC:.2f}% validation accuracy"
)


if not os.path.exists(BASELINE_MODEL):

    raise FileNotFoundError(
        "\nBaseline model not found:\n"
        f"{BASELINE_MODEL}\n\n"
        "Do NOT start stable training until "
        "hybrid_best_model.pth exists."
    )


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print(
    "\nClasses:",
    train_dataset.classes
)

print(
    "Class mapping:",
    train_dataset.class_to_idx
)

print(
    "Train images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


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


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


print(
    "\nTrain batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading Hybrid Model...")

model = HybridBreastCancerNet(

    num_classes=2,

    dropout=0.30,

    pretrained=True

)


# ============================================================
# LOAD 89.50% BASELINE
# ============================================================

print(
    "\nLoading existing 89.50% baseline..."
)

checkpoint = torch.load(
    BASELINE_MODEL,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint,
    strict=True
)

print(
    "Baseline checkpoint loaded successfully."
)


model = model.to(DEVICE)


print(
    "\nEfficientNet features:",
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
# AMP
# ============================================================

USE_AMP = torch.cuda.is_available()

scaler = GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# HELPER:
# FREEZE EVERYTHING
# ============================================================

def freeze_backbones():

    for param in model.efficientnet.parameters():
        param.requires_grad = False

    for param in model.swin.parameters():
        param.requires_grad = False


# ============================================================
# HELPER:
# UNFREEZE FINAL BLOCKS
# ============================================================

def unfreeze_final_blocks():

    # EfficientNet:
    # Unfreeze only final two feature blocks.
    for block in model.efficientnet.features[-2:]:
        for param in block.parameters():
            param.requires_grad = True


    # Swin:
    # Unfreeze only final two feature stages.
    for block in model.swin.features[-2:]:
        for param in block.parameters():
            param.requires_grad = True


    # Swin final normalization layer
    if hasattr(model.swin, "norm"):

        for param in model.swin.norm.parameters():
            param.requires_grad = True


# ============================================================
# HELPER:
# FUSION CLASSIFIER ONLY
# ============================================================

def trainable_classifier_only():

    freeze_backbones()

    for param in model.fusion_classifier.parameters():
        param.requires_grad = True


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

def print_trainable_parameters():

    total = 0
    trainable = 0

    for param in model.parameters():

        total += param.numel()

        if param.requires_grad:
            trainable += param.numel()


    print(
        f"\nTotal parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Frozen parameters: "
        f"{total - trainable:,}"
    )


# ============================================================
# VALIDATE
# ============================================================

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
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            # ------------------------------------------------
            # NaN / Inf protection
            # ------------------------------------------------

            if not torch.isfinite(loss):

                print(
                    "\nWARNING: Non-finite validation loss detected."
                )

                return float("inf"), 0.0


            running_loss += loss.item()


            predictions = outputs.argmax(
                dim=1
            )


            total += labels.size(0)

            correct += (
                predictions == labels
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


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()


    for images, labels in train_loader:

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


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        # ----------------------------------------------------
        # NaN / Inf detection
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print(
                "\n"
                + "=" * 70
            )

            print(
                "NON-FINITE LOSS DETECTED!"
            )

            print(
                "Current batch skipped."
            )

            print(
                "=" * 70
            )


            optimizer.zero_grad(
                set_to_none=True
            )

            continue


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        # ----------------------------------------------------
        # Unscale before gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100.0 *
        correct /
        total
    )


    elapsed = (
        time.time() -
        start_time
    )


    return (
        epoch_loss,
        epoch_accuracy,
        elapsed
    )


# ============================================================
# HISTORY
# ============================================================

history = []


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_accuracy = BASELINE_VAL_ACC

best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# FUNCTION TO RUN TRAINING
# ============================================================

def run_stage(
    stage_name,
    num_epochs
):

    global best_val_accuracy
    global best_epoch
    global epochs_without_improvement


    print("\n")
    print("=" * 70)
    print(stage_name)
    print("=" * 70)


    for epoch in range(
        1,
        num_epochs + 1
    ):

        print(
            f"\n{stage_name} "
            f"| Epoch [{epoch}/{num_epochs}]"
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_accuracy, elapsed = (
            train_one_epoch()
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate()
        )


        current_lr = optimizer.param_groups[0]["lr"]


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
            f"LR         : {current_lr:.2e}"
        )

        print(
            f"Time       : {elapsed:.2f} sec"
        )


        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        history.append({

            "stage": stage_name,

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "val_loss": val_loss,

            "val_accuracy": val_accuracy,

            "learning_rate": current_lr

        })


        # ----------------------------------------------------
        # SAVE ONLY IF BETTER THAN 89.50%
        # ----------------------------------------------------

        if (
            np.isfinite(val_accuracy)
            and
            val_accuracy > best_val_accuracy
        ):

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            epochs_without_improvement = 0


            torch.save(
                model.state_dict(),
                STABLE_BEST_MODEL
            )


            print(
                "\n*** NEW STABLE BEST MODEL SAVED ***"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_val_accuracy:.2f}%"
            )


        else:

            epochs_without_improvement += 1


            print(
                "\nNo improvement."
            )

            print(
                f"Best remains: "
                f"{best_val_accuracy:.2f}%"
            )


        # ----------------------------------------------------
        # NaN STOP
        # ----------------------------------------------------

        if not np.isfinite(val_loss):

            print(
                "\n"
                + "=" * 70
            )

            print(
                "STOPPING THIS STAGE"
            )

            print(
                "Validation loss became non-finite."
            )

            print(
                "=" * 70
            )

            break


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping:"
            )

            print(
                f"No validation improvement "
                f"for {PATIENCE} epochs."
            )

            break


        # ----------------------------------------------------
        # Clear GPU cache
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# STAGE 1
# ============================================================

trainable_classifier_only()

print(
    "\nSTAGE 1:"
)

print(
    "Backbones: FROZEN"
)

print(
    "Fusion classifier: TRAINABLE"
)

print_trainable_parameters()


optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=STAGE1_LR,

    weight_decay=WEIGHT_DECAY
)


run_stage(
    "STAGE 1 - CLASSIFIER TRAINING",
    STAGE1_EPOCHS
)


# ============================================================
# STAGE 2
# ============================================================

print("\nPreparing Stage 2...")

freeze_backbones()

# Fusion classifier remains trainable
for param in model.fusion_classifier.parameters():
    param.requires_grad = True

# Unfreeze only final backbone blocks
unfreeze_final_blocks()


print(
    "\nSTAGE 2:"
)

print(
    "Final backbone blocks: TRAINABLE"
)

print(
    "Fusion classifier: TRAINABLE"
)

print_trainable_parameters()


optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=STAGE2_LR,

    weight_decay=WEIGHT_DECAY
)


run_stage(
    "STAGE 2 - CONTROLLED FINE-TUNING",
    STAGE2_EPOCHS
)


# ============================================================
# SAVE FINAL STABLE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    STABLE_FINAL_MODEL
)


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "stage",
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


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STABLE HYBRID TRAINING COMPLETED")
print("=" * 70)

print(
    f"Original baseline : "
    f"{BASELINE_VAL_ACC:.2f}%"
)

print(
    f"Best stable result: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best epoch        : "
    f"{best_epoch}"
)

print(
    "\nOriginal baseline preserved:"
)

print(
    BASELINE_MODEL
)

print(
    "\nNew stable best model:"
)

print(
    STABLE_BEST_MODEL
)

print(
    "\nNew stable final model:"
)

print(
    STABLE_FINAL_MODEL
)

print(
    "\nTraining history:"
)

print(
    HISTORY_FILE
)

print("=" * 70)