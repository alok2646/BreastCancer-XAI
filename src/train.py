import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast

from model import BreastCancerNet
from dataset_patient import train_loader, val_loader

# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SAVE_DIR = "saved_models"
BEST_MODEL = os.path.join(SAVE_DIR, "patient_best_model.pth")
CHECKPOINT = os.path.join(SAVE_DIR, "training_checkpoint.pth")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# Model
# ==========================================================

model = BreastCancerNet().to(DEVICE)

# ==========================================================
# Loss Function
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
# Learning Rate Scheduler
# ==========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# ==========================================================
# Mixed Precision Training
# ==========================================================

scaler = GradScaler()
start_epoch = 0
best_accuracy = 0.0

if os.path.exists(CHECKPOINT):

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_accuracy = checkpoint["best_accuracy"]

    print("=" * 60)
    print(f"RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print("=" * 60)

print("=" * 60)
print("Training Configuration")
print("=" * 60)

print("Device          :", DEVICE)
print("Epochs          :", NUM_EPOCHS)
print("Learning Rate   :", LEARNING_RATE)
print("Weight Decay    :", WEIGHT_DECAY)
print("Optimizer       : AdamW")
print("Loss            : CrossEntropyLoss")
print("Scheduler       : ReduceLROnPlateau")
print("AMP             :", torch.cuda.is_available())

print("=" * 60)

# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100 * correct / total

    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_time


# ==========================================================
# Validation
# ==========================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad(), torch.amp.autocast("cuda"):

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_acc = 100 * correct / total

    return val_loss, val_acc

# ==========================================================
# Training Loop
# ==========================================================

best_accuracy = 0.0

print("=" * 60)
print("Training Started...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    train_loss, train_acc, train_time = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_loss)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    print(f"Time       : {train_time:.2f} sec")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(model.state_dict(), "saved_models/patient_best_model.pth")

        print("Best Model Saved!")

print("\n" + "=" * 60)
print("Training Completed Successfully")
print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
print("=" * 60)

# ==========================================================
# Save Final Model
# ==========================================================

torch.save(model.state_dict(), "saved_models/patient_final_model.pth")

print("\nFinal Model Saved Successfully!")

# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print(" Breast Cancer XAI Training ")
    print("=" * 60)

    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {NUM_EPOCHS}")
    
    print(f"LearningRate: {LEARNING_RATE}")

    print("=" * 60)


if __name__ == "__main__":

    main()