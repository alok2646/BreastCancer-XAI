import os
import random
import time
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
    roc_auc_score
)

from hybrid_b3_swin import HybridB3SwinNet


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATASET_DIR = "dataset_base_final"

BATCH_SIZE = 4
NUM_WORKERS = 0

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5

LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30
NUM_CLASSES = 2

PATIENCE = 2

MODEL_DIR = "saved_models"
RESULT_DIR = "results/b3_swin"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("EFFICIENTNET-B3 + SWIN-TINY")
print("TRANSFER LEARNING + CONTROLLED FINE-TUNING")
print("=" * 70)

print(f"\nDevice       : {DEVICE}")
print(f"Dataset      : {DATASET_DIR}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Stage 1      : {STAGE1_EPOCHS} epochs")
print(f"Stage 2      : {STAGE2_EPOCHS} epochs")
print(f"Seed         : {SEED}")


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

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
    os.path.join(DATASET_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=val_test_transform
)


print("\nClasses:")
print(train_dataset.classes)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

print(f"\nTrain Images      : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Test Images       : {len(test_dataset)}")


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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nBatches:")
print(f"Train      : {len(train_loader)}")
print(f"Validation : {len(val_loader)}")
print(f"Test       : {len(test_loader)}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B3 + Swin-Tiny...")

model = HybridB3SwinNet(
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
    pretrained=True
)

model = model.to(DEVICE)

print("Model loaded successfully.")


# ============================================================
# PARAMETER HELPERS
# ============================================================

def freeze_all(model):

    for param in model.parameters():
        param.requires_grad = False


def stage1_setup(model):

    # Freeze complete backbones
    freeze_all(model)

    # Train only fusion classifier
    for param in model.classifier.parameters():
        param.requires_grad = True


def stage2_setup(model):

    # First freeze everything
    freeze_all(model)

    # --------------------------------------------------------
    # EfficientNet-B3
    # Unfreeze final feature blocks
    # --------------------------------------------------------

    for param in model.efficientnet.features[-2:].parameters():
        param.requires_grad = True

    # --------------------------------------------------------
    # Swin-Tiny
    # Unfreeze final transformer stage
    # --------------------------------------------------------

    for param in model.swin.features[-1].parameters():
        param.requires_grad = True

    # Swin final normalization
    for param in model.swin.norm.parameters():
        param.requires_grad = True

    # --------------------------------------------------------
    # Fusion classifier
    # --------------------------------------------------------

    for param in model.classifier.parameters():
        param.requires_grad = True


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    return total, trainable, frozen


# ============================================================
# LOSS
# ============================================================

# Dataset is already balanced.
# Therefore no class-weighted loss is used.

criterion = nn.CrossEntropyLoss()


# ============================================================
# AMP
# ============================================================

USE_AMP = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            targets,
            probabilities[:, 1]
        )

    except ValueError:

        auc = 0.0

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    total_samples = 0

    for images, labels in loader:

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

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        total_samples
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_targets.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        total_samples
    )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        auc
    ) = calculate_metrics(
        np.array(all_targets),
        np.array(all_predictions),
        all_probabilities
    )

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# HISTORY
# ============================================================

history = []

best_val_accuracy = 0.0
best_model_path = os.path.join(
    MODEL_DIR,
    "b3_swin_best_model.pth"
)


def save_history():

    history_path = os.path.join(
        RESULT_DIR,
        "b3_swin_training_history.csv"
    )

    if len(history) == 0:
        return

    keys = history[0].keys()

    with open(
        history_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# STAGE 1
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — CLASSIFIER TRAINING")
print("=" * 70)

stage1_setup(model)

total, trainable, frozen = count_parameters(
    model
)

print(f"\nTotal Parameters     : {total:,}")
print(f"Trainable Parameters : {trainable:,}")
print(f"Frozen Parameters    : {frozen:,}")

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE_STAGE1,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

for epoch in range(STAGE1_EPOCHS):

    start_time = time.time()

    print(
        f"\nSTAGE 1 | "
        f"Epoch [{epoch + 1}/{STAGE1_EPOCHS}]"
    )

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate(
        model,
        val_loader
    )

    scheduler.step(val_acc)

    elapsed = time.time() - start_time

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc * 100:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc * 100:.2f}%")
    print(f"Val Prec   : {val_precision * 100:.2f}%")
    print(f"Val Recall : {val_recall * 100:.2f}%")
    print(f"Val F1     : {val_f1 * 100:.2f}%")
    print(f"Val AUC    : {val_auc:.4f}")
    print(f"LR         : {current_lr:.2e}")
    print(f"Time       : {elapsed:.2f} sec")

    history.append({
        "stage": "stage1",
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "learning_rate": current_lr
    })

    save_history()

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"\n*** NEW BEST MODEL SAVED ***"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )


# ============================================================
# STAGE 2
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 — CONTROLLED FINE-TUNING")
print("=" * 70)

stage2_setup(model)

total, trainable, frozen = count_parameters(
    model
)

print(f"\nTotal Parameters     : {total:,}")
print(f"Trainable Parameters : {trainable:,}")
print(f"Frozen Parameters    : {frozen:,}")

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE_STAGE2,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

epochs_without_improvement = 0

for epoch in range(STAGE2_EPOCHS):

    start_time = time.time()

    print(
        f"\nSTAGE 2 | "
        f"Epoch [{epoch + 1}/{STAGE2_EPOCHS}]"
    )

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate(
        model,
        val_loader
    )

    scheduler.step(val_acc)

    elapsed = time.time() - start_time

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc * 100:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc * 100:.2f}%")
    print(f"Val Prec   : {val_precision * 100:.2f}%")
    print(f"Val Recall : {val_recall * 100:.2f}%")
    print(f"Val F1     : {val_f1 * 100:.2f}%")
    print(f"Val AUC    : {val_auc:.4f}")
    print(f"LR         : {current_lr:.2e}")
    print(f"Time       : {elapsed:.2f} sec")

    history.append({
        "stage": "stage2",
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "learning_rate": current_lr
    })

    save_history()

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        torch.save(
            model.state_dict(),
            best_model_path
        )

        epochs_without_improvement = 0

        print(
            "\n*** NEW BEST MODEL SAVED ***"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo validation improvement."
        )

        print(
            f"Best remains: "
            f"{best_val_accuracy * 100:.2f}%"
        )

        if epochs_without_improvement >= PATIENCE:

            print(
                "\n*** EARLY STOPPING ***"
            )

            break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best validation model...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_auc
) = validate(
    model,
    test_loader
)

print(f"\nTest Loss      : {test_loss:.4f}")
print(
    f"Test Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)
print(
    f"Test Precision : "
    f"{test_precision * 100:.2f}%"
)
print(
    f"Test Recall    : "
    f"{test_recall * 100:.2f}%"
)
print(
    f"Test F1        : "
    f"{test_f1 * 100:.2f}%"
)
print(
    f"Test ROC-AUC   : "
    f"{test_auc:.4f}"
)


# ============================================================
# SAVE FINAL TEST METRICS
# ============================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "b3_swin_test_metrics.csv"
)

with open(
    metrics_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Value"
    ])

    writer.writerow([
        "Accuracy",
        test_accuracy
    ])

    writer.writerow([
        "Precision",
        test_precision
    ])

    writer.writerow([
        "Recall",
        test_recall
    ])

    writer.writerow([
        "F1",
        test_f1
    ])

    writer.writerow([
        "ROC-AUC",
        test_auc
    ])


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "b3_swin_final_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest model : {best_model_path}"
)

print(
    f"Final model: {final_model_path}"
)

print(
    f"History    : "
    f"{RESULT_DIR}/b3_swin_training_history.csv"
)

print(
    f"Metrics    : "
    f"{metrics_path}"
)