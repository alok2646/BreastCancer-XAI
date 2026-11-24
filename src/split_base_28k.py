import random
import shutil
from pathlib import Path

# ============================================================
# BASE-PAPER STYLE 70/30 SPLIT
# Original + Augmented pair ALWAYS stay together
# ============================================================

SEED = 42
random.seed(SEED)

SOURCE_DIR = Path("dataset_base_28k")
OUTPUT_DIR = Path("dataset_base_split")

CLASSES = ["benign", "malignant"]

TRAIN_RATIO = 0.70


for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    train_class = OUTPUT_DIR / "train" / class_name
    test_class = OUTPUT_DIR / "test" / class_name

    train_class.mkdir(parents=True, exist_ok=True)
    test_class.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find ORIGINAL images only
    # --------------------------------------------------------
    original_images = [
        p for p in source_class.glob("*.png")
        if not p.stem.endswith("_aug")
    ]

    print(f"\n{class_name}")
    print(f"Original images/pairs: {len(original_images)}")

    # --------------------------------------------------------
    # Shuffle originals
    # --------------------------------------------------------
    random.shuffle(original_images)

    train_count = int(len(original_images) * TRAIN_RATIO)

    train_originals = original_images[:train_count]
    test_originals = original_images[train_count:]

    # --------------------------------------------------------
    # Copy original + augmented image together
    # --------------------------------------------------------
    for original_path in train_originals:

        augmented_path = (
            original_path.parent /
            f"{original_path.stem}_aug.png"
        )

        shutil.copy2(
            original_path,
            train_class / original_path.name
        )

        shutil.copy2(
            augmented_path,
            train_class / augmented_path.name
        )

    for original_path in test_originals:

        augmented_path = (
            original_path.parent /
            f"{original_path.stem}_aug.png"
        )

        shutil.copy2(
            original_path,
            test_class / original_path.name
        )

        shutil.copy2(
            augmented_path,
            test_class / augmented_path.name
        )

    print(f"Train pairs: {len(train_originals)}")
    print(f"Test pairs : {len(test_originals)}")


# ============================================================
# FINAL COUNT
# ============================================================

print("\n========================================")
print("70/30 SPLIT COMPLETED")
print("========================================")

for split in ["train", "test"]:

    total = 0

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split / class_name
        count = len(list(folder.glob("*.png")))

        print(f"{split}/{class_name}: {count}")

        total += count

    print(f"{split} total: {total}")

print("\nOutput:", OUTPUT_DIR)