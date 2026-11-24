# ==========================================================
# Patient-Level Dataset Loader
# Experimental Dataset
# Original dataset.py is NOT modified
# ==========================================================

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================================
# Configuration
# ==========================================================

DATASET_ROOT = Path("dataset_patient_split")

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 100
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Print Configuration
# ==========================================================

print("=" * 60)
print("PATIENT-LEVEL DATASET LOADER")
print("=" * 60)

print("Dataset Root :", DATASET_ROOT.resolve())
print("Train Path   :", TRAIN_DIR.resolve())
print("Val Path     :", VAL_DIR.resolve())
print("Test Path    :", TEST_DIR.resolve())

print()
print("Image Size   :", IMAGE_SIZE)
print("Batch Size   :", BATCH_SIZE)
print("Workers      :", NUM_WORKERS)
print("Device       :", DEVICE)

print("=" * 60)


# ==========================================================
# Training Transform
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


# ==========================================================
# Validation Transform
# ==========================================================

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


# ==========================================================
# Test Transform
# ==========================================================

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
# Load Datasets
# ==========================================================

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

print()
print("=" * 60)
print("DATASETS LOADED")
print("=" * 60)

print("Classes       :", train_dataset.classes)
print("Class Mapping :", train_dataset.class_to_idx)

print()

print("Train Images :", len(train_dataset))
print("Val Images   :", len(val_dataset))
print("Test Images  :", len(test_dataset))

print("=" * 60)


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=False
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=False
)


# ==========================================================
# DataLoader Information
# ==========================================================

print()
print("=" * 60)
print("DATALOADERS CREATED")
print("=" * 60)

print("Train Batches :", len(train_loader))
print("Val Batches   :", len(val_loader))
print("Test Batches  :", len(test_loader))

print("=" * 60)


# ==========================================================
# Verify One Batch
# ==========================================================

images, labels = next(iter(train_loader))

print()
print("=" * 60)
print("BATCH VERIFICATION")
print("=" * 60)

print("Image Shape :", images.shape)
print("Label Shape :", labels.shape)

print()
print("First 10 Labels:")
print(labels[:10])

print("=" * 60)
print("Patient-Level Dataset Loader Ready!")
print("=" * 60)