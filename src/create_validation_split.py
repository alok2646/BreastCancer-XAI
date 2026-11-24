import random
import shutil
from pathlib import Path

# ============================================================
# FINAL 70/15/15 SPLIT
# SOURCE = dataset_base_28k
#
# Per class:
# 7000 original + 7000 augmented = 14000 images
#
# 70% -> Train = 9800 images
# 15% -> Val   = 2100 images
# 15% -> Test  = 2100 images
# ============================================================

SEED = 42
random.seed(SEED)

SOURCE_DIR = Path("dataset_base_28k")
OUTPUT_DIR = Path("dataset_base_final")

CLASSES = ["benign", "malignant"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CHECK RATIOS
# ============================================================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0


# ============================================================
# REMOVE OLD INCORRECT OUTPUT
# ============================================================

if OUTPUT_DIR.exists():
    print("\nRemoving previous dataset_base_final...")
    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE FRESH OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (
            OUTPUT_DIR /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    print("\n========================================")
    print(f"Processing: {class_name}")
    print("========================================")

    # --------------------------------------------------------
    # Find ORIGINAL images only
    # --------------------------------------------------------

    original_images = [
        p for p in source_class.glob("*.png")
        if not p.stem.endswith("_aug")
    ]

    print(
        f"Original images/pairs found: "
        f"{len(original_images)}"
    )

    if len(original_images) != 7000:
        raise ValueError(
            f"Expected 7000 original images "
            f"for {class_name}, but found "
            f"{len(original_images)}"
        )

    # --------------------------------------------------------
    # Shuffle with fixed seed
    # --------------------------------------------------------

    random.shuffle(original_images)

    # --------------------------------------------------------
    # Calculate number of ORIGINAL PAIRS
    # --------------------------------------------------------

    train_pairs = int(
        len(original_images) * TRAIN_RATIO
    )

    val_pairs = int(
        len(original_images) * VAL_RATIO
    )

    test_pairs = (
        len(original_images)
        - train_pairs
        - val_pairs
    )

    print(f"Train pairs: {train_pairs}")
    print(f"Val pairs  : {val_pairs}")
    print(f"Test pairs : {test_pairs}")

    # --------------------------------------------------------
    # Split original images
    # --------------------------------------------------------

    train_originals = original_images[
        :train_pairs
    ]

    val_originals = original_images[
        train_pairs:
        train_pairs + val_pairs
    ]

    test_originals = original_images[
        train_pairs + val_pairs:
    ]


    # ========================================================
    # FUNCTION TO COPY ORIGINAL + AUGMENTED PAIR
    # ========================================================

    def copy_pairs(image_list, split):

        destination = (
            OUTPUT_DIR /
            split /
            class_name
        )

        for original_path in image_list:

            # Original image
            shutil.copy2(
                original_path,
                destination /
                original_path.name
            )

            # Corresponding augmented image
            augmented_path = (
                source_class /
                f"{original_path.stem}_aug.png"
            )

            if not augmented_path.exists():
                raise FileNotFoundError(
                    f"Augmented image missing:\n"
                    f"{augmented_path}"
                )

            shutil.copy2(
                augmented_path,
                destination /
                augmented_path.name
            )


    # ========================================================
    # COPY TRAIN / VAL / TEST
    # ========================================================

    copy_pairs(
        train_originals,
        "train"
    )

    copy_pairs(
        val_originals,
        "val"
    )

    copy_pairs(
        test_originals,
        "test"
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 50)
print("FINAL DATASET CREATED SUCCESSFULLY")
print("=" * 50)

grand_total = 0

for split in ["train", "val", "test"]:

    split_total = 0

    print(f"\n{split.upper()}")

    for class_name in CLASSES:

        folder = (
            OUTPUT_DIR /
            split /
            class_name
        )

        count = len(
            list(folder.glob("*.png"))
        )

        print(
            f"{class_name:10s}: {count}"
        )

        split_total += count

    print(
        f"{split:10s} total: {split_total}"
    )

    grand_total += split_total


print("\n" + "=" * 50)
print(f"GRAND TOTAL: {grand_total}")
print("=" * 50)

print("\nOutput directory:")
print(OUTPUT_DIR)