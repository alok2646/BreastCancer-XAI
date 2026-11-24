import os
import random
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
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset_base_final"

BATCH_SIZE = 400
NUM_WORKERS = 0

NUM_CLASSES = 2

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5

LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

PATIENCE = 2

SEED = 42

MODEL_DIR = "saved_models"
RESULT_DIR = "results/b3_swin_multiscale"

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("B3 + SWIN ATTENTION FUSION TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([

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
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)


print("\nDataset information:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

print("\nClasses:")
print(train_dataset.class_to_idx)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

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
    dropout=DROPOUT
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nTotal parameters: {total_params:,}"
)


# ============================================================
# STAGE 1
# FREEZE BACKBONES
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: FUSION + CLASSIFIER TRAINING")
print("=" * 70)

# Freeze EfficientNet
for param in model.efficientnet.parameters():
    param.requires_grad = False

# Freeze Swin
for param in model.swin.parameters():
    param.requires_grad = False

# Keep these trainable:
# projection layers
# attention fusion
# classifier

for module in [
    model.eff_projection,
    model.swin_projection,
    model.attention_fusion,
    model.classifier
]:

    for param in module.parameters():
        param.requires_grad = True


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Trainable parameters Stage 1: "
    f"{trainable_params:,}"
)


criterion = nn.CrossEntropyLoss()


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


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_preds = []
    all_labels = []

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
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

        running_loss += (
            loss.item() *
            images.size(0)
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(train_loader.dataset)
    )

    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )

    return epoch_loss, epoch_acc


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = (
        running_loss /
        len(val_loader.dataset)
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probs
        )
    except ValueError:
        auc = 0.0

    return (
        loss,
        acc,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# TRAIN STAGE 1
# ============================================================

history = []

best_val_acc = 0.0

best_model_path = os.path.join(
    MODEL_DIR,
    "b3_swin_multiscale_best_model.pth"
)

stage1_best = 0


for epoch in range(1, STAGE1_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch()

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate()

    scheduler.step(val_acc)

    print(
        f"\nStage 1 | Epoch "
        f"{epoch}/{STAGE1_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc : {train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Val Acc   : {val_acc * 100:.2f}%"
    )

    print(
        f"Val Prec  : {val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall : {val_recall * 100:.2f}%"
    )

    print(
        f"Val F1    : {val_f1 * 100:.2f}%"
    )

    print(
        f"Val AUC   : {val_auc:.4f}"
    )

    history.append({
        "stage": "Stage1",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc
    })

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        stage1_best = epoch

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "✓ Best model saved"
        )


# ============================================================
# STAGE 2
# PARTIAL FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: PARTIAL BACKBONE FINE-TUNING")
print("=" * 70)


# First freeze everything
for param in model.parameters():
    param.requires_grad = False


# ------------------------------------------------------------
# EfficientNet-B3
# Unfreeze last feature blocks
# ------------------------------------------------------------

for block in model.efficientnet.features[-2:]:

    for param in block.parameters():
        param.requires_grad = True


# ------------------------------------------------------------
# Swin-Tiny
# Unfreeze final feature blocks
# ------------------------------------------------------------

for block in model.swin.features[-3:]:

    for param in block.parameters():
        param.requires_grad = True


# ------------------------------------------------------------
# Fusion layers remain trainable
# ------------------------------------------------------------

for module in [
    model.eff_projection,
    model.swin_projection,
    model.attention_fusion,
    model.classifier
]:

    for param in module.parameters():
        param.requires_grad = True


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Trainable parameters Stage 2: "
    f"{trainable_params:,}"
)


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


# ============================================================
# EARLY STOPPING
# ============================================================

patience_counter = 0


for epoch in range(1, STAGE2_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch()

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc
    ) = validate()

    scheduler.step(val_acc)

    print(
        f"\nStage 2 | Epoch "
        f"{epoch}/{STAGE2_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc : {train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Val Acc   : {val_acc * 100:.2f}%"
    )

    print(
        f"Val Prec  : {val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall : {val_recall * 100:.2f}%"
    )

    print(
        f"Val F1    : {val_f1 * 100:.2f}%"
    )

    print(
        f"Val AUC   : {val_auc:.4f}"
    )

    history.append({
        "stage": "Stage2",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc
    })

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "✓ Best model updated"
        )

        patience_counter = 0

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    RESULT_DIR,
    "b3_swin_multiscale_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

model.eval()

all_preds = []
all_labels = []
all_probs = []

test_loss = 0.0


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

        test_loss += (
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

        all_probs.extend(
            probabilities.cpu().numpy()
        )

        all_preds.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_loss /= len(
    test_loader.dataset
)


test_accuracy = accuracy_score(
    all_labels,
    all_preds
)

test_precision = precision_score(
    all_labels,
    all_preds,
    zero_division=0
)

test_recall = recall_score(
    all_labels,
    all_preds,
    zero_division=0
)

test_f1 = f1_score(
    all_labels,
    all_preds,
    zero_division=0
)

test_auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    all_preds
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\nTest Results")
print("-" * 50)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Accuracy      : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision     : {test_precision * 100:.2f}%"
)

print(
    f"Recall        : {test_recall * 100:.2f}%"
)

print(
    f"F1 Score      : {test_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC       : {test_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE TEST METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "test_loss": test_loss,

    "accuracy": test_accuracy,

    "precision": test_precision,

    "recall": test_recall,

    "f1_score": test_f1,

    "roc_auc": test_auc,

    "TN": cm[0, 0],
    "FP": cm[0, 1],
    "FN": cm[1, 0],
    "TP": cm[1, 1]

}])


metrics_path = os.path.join(
    RESULT_DIR,
    "b3_swin_multiscale_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "b3_swin_multiscale_final_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Test Recall: "
    f"{test_recall * 100:.2f}%"
)

print(
    f"Test F1: "
    f"{test_f1 * 100:.2f}%"
)

print(
    f"Test ROC-AUC: "
    f"{test_auc:.4f}"
)

print("\nFiles:")
print(
    "Best model :",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)

print(
    "History    :",
    history_path
)

print(
    "Metrics    :",
    metrics_path
)

print("=" * 70)