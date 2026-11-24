import os
import random
import shutil
from pathlib import Path

# ============================================================
# BASE-PAPER STYLE DATA SELECTION
# 7,000 BENIGN + 7,000 MALIGNANT
# ============================================================

SEED = 42
random.seed(SEED)

# Original IDC dataset
SOURCE_DIR = Path("IDC_regular_ps50_idx5")

# New selected dataset
OUTPUT_DIR = Path("dataset_base_14k")

BENIGN_DIR = OUTPUT_DIR / "benign"
MALIGNANT_DIR = OUTPUT_DIR / "malignant"

BENIGN_DIR.mkdir(parents=True, exist_ok=True)
MALIGNANT_DIR.mkdir(parents=True, exist_ok=True)


def collect_images(class_name):
    """
    Recursively collect all image files belonging
    to the requested class.
    """

    class_id = "0" if class_name == "benign" else "1"

    images = []

    for patient_dir in SOURCE_DIR.iterdir():

        if not patient_dir.is_dir():
            continue

        class_dir = patient_dir / class_id

        if not class_dir.exists():
            continue

        for image_path in class_dir.glob("*.png"):
            images.append(image_path)

    return images


# ============================================================
# COLLECT ORIGINAL IMAGES
# ============================================================

print("\nCollecting original IDC images...")

benign_images = collect_images("benign")
malignant_images = collect_images("malignant")

print(f"Available benign images    : {len(benign_images)}")
print(f"Available malignant images : {len(malignant_images)}")


# ============================================================
# RANDOMLY SELECT 7,000 EACH
# ============================================================

NUM_PER_CLASS = 7000

if len(benign_images) < NUM_PER_CLASS:
    raise ValueError("Not enough benign images!")

if len(malignant_images) < NUM_PER_CLASS:
    raise ValueError("Not enough malignant images!")


selected_benign = random.sample(
    benign_images,
    NUM_PER_CLASS
)

selected_malignant = random.sample(
    malignant_images,
    NUM_PER_CLASS
)


# ============================================================
# COPY SELECTED IMAGES
# ============================================================

print("\nCopying selected images...")

for i, image_path in enumerate(selected_benign):

    destination = BENIGN_DIR / f"benign_{i:05d}.png"

    shutil.copy2(
        image_path,
        destination
    )


for i, image_path in enumerate(selected_malignant):

    destination = MALIGNANT_DIR / f"malignant_{i:05d}.png"

    shutil.copy2(
        image_path,
        destination
    )


# ============================================================
# FINAL CHECK
# ============================================================

benign_count = len(list(BENIGN_DIR.glob("*.png")))
malignant_count = len(list(MALIGNANT_DIR.glob("*.png")))

print("\n============================================")
print("BASE DATASET SELECTION COMPLETED")
print("============================================")

print(f"Benign images    : {benign_count}")
print(f"Malignant images : {malignant_count}")
print(f"Total images     : {benign_count + malignant_count}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSeed:", SEED)