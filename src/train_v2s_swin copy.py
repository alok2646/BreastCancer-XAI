"""
FAST + STABLE TRAINING
EfficientNetV2-S + Swin-Tiny

Dataset:
    dataset_patient_split

Classes:
    0 = benign
    1 = malignant

Features:
    EfficientNetV2-S = 1280
    Swin-Tiny       = 768
    Fusion          = 2048

Training:
    Stage 1 -> Frozen backbones
    Stage 2 -> Controlled fine-tuning

Safety:
    - Automatic class-weighted loss
    - NaN / Inf detection
    - Gradient clipping
    - Early stopping
    - Best model checkpoint
    - Training history CSV
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

from hybrid_v2s_swin import HybridV2SSwinNet


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

BATCH_SIZE = 4

NUM_WORKERS = 0


DATASET_ROOT = "dataset_patient_split"

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "val"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SAVE_DIR = "saved_models"

RESULT_DIR = "results/v2s_swin"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


BEST_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_v2s_swin_best_model.pth"
)

FINAL_MODEL_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_v2s_swin_final_model.pth"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "v2s_swin_training_history.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Stage 1
STAGE1_EPOCHS = 3

STAGE1_LR = 1e-4


# Stage 2
STAGE2_EPOCHS = 5

STAGE2_LR = 1e-5


WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0

PATIENCE = 2


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
# HEADER
# ============================================================

print("=" * 75)

print(
    "HYBRID EFFICIENTNET-V2-S + SWIN-TINY"
)

print(
    "FAST + STABLE TRAINING"
)

print("=" * 75)

print(
    "\nDevice:",
    DEVICE
)

print(
    "Dataset:",
    os.path.abspath(DATASET_ROOT)
)

print(
    "Batch Size:",
    BATCH_SIZE
)

print(
    "Stage 1 Epochs:",
    STAGE1_EPOCHS
)

print(
    "Stage 2 Epochs:",
    STAGE2_EPOCHS
)

print(
    "Early Stopping Patience:",
    PATIENCE
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

train_dataset = datasets.ImageFolder(

    TRAIN_DIR,

    transform=train_transform
)


val_dataset = datasets.ImageFolder(

    VAL_DIR,

    transform=val_transform
)


print("\nClasses:")

print(
    train_dataset.classes
)

print(
    "\nClass Mapping:"
)

print(
    train_dataset.class_to_idx
)


print(
    "\nTrain Images:",
    len(train_dataset)
)

print(
    "Validation Images:",
    len(val_dataset)
)


# ============================================================
# CLASS BALANCE
# ============================================================

train_targets = np.array(
    train_dataset.targets
)


class_counts = np.bincount(
    train_targets,
    minlength=2
)


print("\nClass Distribution:")

print(
    "Benign:",
    class_counts[0]
)

print(
    "Malignant:",
    class_counts[1]
)


# ============================================================
# AUTOMATIC CLASS WEIGHTS
# ============================================================

total_samples = class_counts.sum()

num_classes = len(class_counts)


class_weights = (
    total_samples /
    (
        num_classes *
        class_counts
    )
)


print(
    "\nAutomatic Class Weights:"
)

print(
    "Benign weight    :",
    f"{class_weights[0]:.4f}"
)

print(
    "Malignant weight :",
    f"{class_weights[1]:.4f}"
)


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


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
    "\nTrain Batches:",
    len(train_loader)
)

print(
    "Validation Batches:",
    len(val_loader)
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading Hybrid V2-S + Swin Model...")

model = HybridV2SSwinNet(

    num_classes=2,

    dropout=0.30,

    pretrained=True
)


model = model.to(DEVICE)


print(
    "\nModel Ready!"
)

print(
    "EfficientNetV2-S Features:",
    model.efficient_features
)

print(
    "Swin-Tiny Features:",
    model.swin_features
)

print(
    "Fusion Features:",
    model.fusion_features
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights_tensor
)


# ============================================================
# AMP
# ============================================================

USE_AMP = torch.cuda.is_available()


scaler = GradScaler(

    "cuda",

    enabled=USE_AMP
)


# ============================================================
# FREEZE BACKBONES
# ============================================================

def freeze_backbones():

    for param in model.efficientnet.parameters():

        param.requires_grad = False


    for param in model.swin.parameters():

        param.requires_grad = False


# ============================================================
# STAGE 1 TRAINABLE PARAMETERS
# ============================================================

def prepare_stage1():

    freeze_backbones()


    for param in model.fusion_classifier.parameters():

        param.requires_grad = True


# ============================================================
# STAGE 2 TRAINABLE PARAMETERS
# ============================================================

def prepare_stage2():

    freeze_backbones()


    # Fusion classifier
    for param in model.fusion_classifier.parameters():

        param.requires_grad = True


    # EfficientNetV2-S final blocks
    for block in model.efficientnet.features[-2:]:

        for param in block.parameters():

            param.requires_grad = True


    # Swin final blocks
    for block in model.swin.features[-2:]:

        for param in block.parameters():

            param.requires_grad = True


    # Swin final normalization
    if hasattr(
        model.swin,
        "norm"
    ):

        for param in model.swin.norm.parameters():

            param.requires_grad = True


# ============================================================
# PARAMETER COUNT
# ============================================================

def print_parameter_status():

    total = 0

    trainable = 0


    for param in model.parameters():

        total += param.numel()

        if param.requires_grad:

            trainable += param.numel()


    print(
        "\nTotal Parameters:",
        f"{total:,}"
    )

    print(
        "Trainable Parameters:",
        f"{trainable:,}"
    )

    print(
        "Frozen Parameters:",
        f"{total - trainable:,}"
    )


# ============================================================
# VALIDATION
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


            if not torch.isfinite(loss):

                return (
                    float("inf"),
                    0.0
                )


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

    skipped_batches = 0


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
        # NaN / Inf protection
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            skipped_batches += 1

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
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )


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


    elapsed = (
        time.time() -
        start_time
    )


    valid_batches = (
        len(train_loader) -
        skipped_batches
    )


    if valid_batches == 0:

        return (
            float("inf"),
            0.0,
            elapsed
        )


    epoch_loss = (
        running_loss /
        valid_batches
    )


    epoch_accuracy = (
        100.0 *
        correct /
        total
    )


    if skipped_batches > 0:

        print(
            f"\nSkipped batches: "
            f"{skipped_batches}"
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


def save_history():

    with open(
        HISTORY_PATH,
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


        writer.writerows(
            history
        )


# ============================================================
# BEST MODEL
# ============================================================

best_val_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# RUN STAGE
# ============================================================

def run_stage(
    stage_name,
    num_epochs
):

    global best_val_accuracy

    global best_epoch

    global epochs_without_improvement


    print("\n")

    print("=" * 75)

    print(stage_name)

    print("=" * 75)


    for epoch in range(
        1,
        num_epochs + 1
    ):

        print(
            f"\n{stage_name} | "
            f"Epoch [{epoch}/{num_epochs}]"
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy, elapsed = (
            train_one_epoch()
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate()
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


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
        # HISTORY
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


        # Save history immediately
        save_history()


        # ----------------------------------------------------
        # BEST MODEL
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
                BEST_MODEL_PATH
            )


            print(
                "\n*** NEW BEST MODEL SAVED ***"
            )

            print(
                f"Best Val Accuracy: "
                f"{best_val_accuracy:.2f}%"
            )

        else:

            epochs_without_improvement += 1


            print(
                "\nNo validation improvement."
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
                "\nValidation loss became NaN/Inf."
            )

            print(
                "Stopping this stage."
            )

            break


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\n*** EARLY STOPPING ***"
            )

            print(
                f"No validation improvement "
                f"for {PATIENCE} epochs."
            )

            break


        # ----------------------------------------------------
        # GPU CACHE
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# STAGE 1
# ============================================================

prepare_stage1()


print("\n")

print(
    "STAGE 1:"
)

print(
    "EfficientNetV2-S : FROZEN"
)

print(
    "Swin-Tiny       : FROZEN"
)

print(
    "Fusion Classifier: TRAINABLE"
)


print_parameter_status()


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

print("\n")

print(
    "Preparing Stage 2..."
)


prepare_stage2()


print(
    "\nSTAGE 2:"
)

print(
    "Final EfficientNetV2-S blocks : TRAINABLE"
)

print(
    "Final Swin-Tiny blocks        : TRAINABLE"
)

print(
    "Fusion Classifier             : TRAINABLE"
)


print_parameter_status()


optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=STAGE2_LR,

    weight_decay=WEIGHT_DECAY
)


# Reset patience for Stage 2
epochs_without_improvement = 0


run_stage(
    "STAGE 2 - CONTROLLED FINE-TUNING",
    STAGE2_EPOCHS
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 75)

print(
    "V2S + SWIN TRAINING COMPLETED"
)

print("=" * 75)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    "\nBest Model:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nFinal Model:"
)

print(
    FINAL_MODEL_PATH
)

print(
    "\nTraining History:"
)

print(
    HISTORY_PATH
)

print("=" * 75)