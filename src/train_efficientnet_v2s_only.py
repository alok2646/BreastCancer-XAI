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

from efficientnet_v2s_only import EfficientNetV2SOnly


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset_base_final"

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_CLASSES = 2
DROPOUT = 0.30

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5

LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 2

SEED = 42

MODEL_DIR = "saved_models"
RESULT_DIR = "results/efficientnet_v2s"

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
print("EFFICIENTNET-V2-S ONLY TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)

print(
    "\nClasses:",
    train_dataset.class_to_idx
)


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

model = EfficientNetV2SOnly(
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nTotal parameters: "
    f"{total_params:,}"
)


# ============================================================
# STAGE 1
# FREEZE V2-S BACKBONE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: CLASSIFIER TRAINING")
print("=" * 70)


for param in model.backbone.features.parameters():
    param.requires_grad = False


for param in model.backbone.classifier.parameters():
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
    lr=LR_STAGE1,
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
# TRAIN FUNCTION
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

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    loss = (
        running_loss /
        len(train_loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    return loss, accuracy


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

    loss = (
        running_loss /
        len(val_loader.dataset)
    )

    accuracy = accuracy_score(
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

best_val_acc = 0.0

best_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_v2s_best_model.pth"
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

for epoch in range(
    1,
    STAGE1_EPOCHS + 1
):

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
        f"Val Recall: {val_recall * 100:.2f}%"
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
print("STAGE 2: PARTIAL V2-S FINE-TUNING")
print("=" * 70)


# Freeze everything first

for param in model.parameters():
    param.requires_grad = False


# Unfreeze last two V2-S feature blocks

for block in model.backbone.features[-2:]:

    for param in block.parameters():
        param.requires_grad = True


# Classifier remains trainable

for param in model.backbone.classifier.parameters():
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
    lr=LR_STAGE2,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


patience_counter = 0


# ============================================================
# STAGE 2 TRAINING
# ============================================================

for epoch in range(
    1,
    STAGE2_EPOCHS + 1
):

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
        f"Val Recall: {val_recall * 100:.2f}%"
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
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = os.path.join(
    RESULT_DIR,
    "efficientnet_v2s_training_history.csv"
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
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


model.eval()

test_loss = 0.0

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

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


test_loss /= len(
    test_dataset
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


TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


specificity = TN / (
    TN + FP
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENTNET-V2-S FINAL TEST RESULTS")
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

metrics_df = pd.DataFrame([{

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
    "efficientnet_v2s_test_metrics.csv"
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
    "efficientnet_v2s_final_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


# ============================================================
# SUMMARY
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
    f"{accuracy * 100:.2f}%"
)

print(
    f"Test F1: "
    f"{f1 * 100:.2f}%"
)

print(
    f"Test ROC-AUC: "
    f"{roc_auc:.4f}"
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