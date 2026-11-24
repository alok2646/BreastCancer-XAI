from pathlib import Path
import random
import os
import shutil

# ==========================================================
# Patient-Level Dataset Split
# SAFE EXPERIMENT
# Original dataset will NOT be modified
# ==========================================================

random.seed(42)

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

SOURCE = Path("IDC_regular_ps50_idx5")

OUTPUT = Path("dataset_patient_split")

# ----------------------------------------------------------
# Split Ratios
# ----------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6

# ----------------------------------------------------------
# Safety Check
# ----------------------------------------------------------

if not SOURCE.exists():

    raise FileNotFoundError(
        f"Original dataset not found: {SOURCE.resolve()}"
    )

if OUTPUT.exists() and any(OUTPUT.iterdir()):

    raise RuntimeError(
        f"\n{OUTPUT} already exists and is not empty.\n"
        "Delete/rename the experimental folder manually "
        "only if you intentionally want to recreate it."
    )

# ----------------------------------------------------------
# Create folders
# ----------------------------------------------------------

for split in ["train", "val", "test"]:

    for cls in ["benign", "malignant"]:

        (OUTPUT / split / cls).mkdir(
            parents=True,
            exist_ok=True
        )

# ----------------------------------------------------------
# Get Patient IDs
# ----------------------------------------------------------

patients = [
    p for p in SOURCE.iterdir()
    if p.is_dir()
]

print("=" * 70)
print("PATIENT-LEVEL DATASET SPLIT")
print("=" * 70)

print("Source:", SOURCE.resolve())
print("Output:", OUTPUT.resolve())

print()
print("Total Patients:", len(patients))

# ----------------------------------------------------------
# Shuffle Patients
# ----------------------------------------------------------

random.shuffle(patients)

total_patients = len(patients)

train_end = int(total_patients * TRAIN_RATIO)

val_end = int(
    total_patients * (TRAIN_RATIO + VAL_RATIO)
)

train_patients = patients[:train_end]

val_patients = patients[train_end:val_end]

test_patients = patients[val_end:]

# ----------------------------------------------------------
# Display Patient Split
# ----------------------------------------------------------

print()
print("Patient Split")
print("-" * 70)

print("Train Patients:", len(train_patients))
print("Val Patients  :", len(val_patients))
print("Test Patients :", len(test_patients))

print("-" * 70)

# ----------------------------------------------------------
# Function: Create Hard Link
# ----------------------------------------------------------

def create_link(source_file, destination_file):

    try:

        os.link(
            source_file,
            destination_file
        )

    except OSError:

        # Fallback if hard-link is not supported
        shutil.copy2(
            source_file,
            destination_file
        )


# ----------------------------------------------------------
# Function: Process Patients
# ----------------------------------------------------------

def process_patients(patient_list, split):

    benign_count = 0
    malignant_count = 0

    for patient in patient_list:

        # ----------------------------------------------
        # Class 0 = Benign / Non-IDC
        # ----------------------------------------------

        class0 = patient / "0"

        if class0.exists():

            for image in class0.glob("*.png"):

                destination = (
                    OUTPUT
                    / split
                    / "benign"
                    / image.name
                )

                create_link(
                    image,
                    destination
                )

                benign_count += 1

        # ----------------------------------------------
        # Class 1 = Malignant / IDC
        # ----------------------------------------------

        class1 = patient / "1"

        if class1.exists():

            for image in class1.glob("*.png"):

                destination = (
                    OUTPUT
                    / split
                    / "malignant"
                    / image.name
                )

                create_link(
                    image,
                    destination
                )

                malignant_count += 1

    return benign_count, malignant_count


# ----------------------------------------------------------
# Process Train
# ----------------------------------------------------------

print()
print("Creating TRAIN dataset...")

train_benign, train_malignant = process_patients(
    train_patients,
    "train"
)

print(
    f"Train -> Benign: {train_benign:,} | "
    f"Malignant: {train_malignant:,}"
)

# ----------------------------------------------------------
# Process Validation
# ----------------------------------------------------------

print()
print("Creating VALIDATION dataset...")

val_benign, val_malignant = process_patients(
    val_patients,
    "val"
)

print(
    f"Val -> Benign: {val_benign:,} | "
    f"Malignant: {val_malignant:,}"
)

# ----------------------------------------------------------
# Process Test
# ----------------------------------------------------------

print()
print("Creating TEST dataset...")

test_benign, test_malignant = process_patients(
    test_patients,
    "test"
)

print(
    f"Test -> Benign: {test_benign:,} | "
    f"Malignant: {test_malignant:,}"
)

# ----------------------------------------------------------
# Final Statistics
# ----------------------------------------------------------

total_benign = (
    train_benign
    + val_benign
    + test_benign
)

total_malignant = (
    train_malignant
    + val_malignant
    + test_malignant
)

total_images = (
    total_benign
    + total_malignant
)

print()
print("=" * 70)
print("PATIENT-LEVEL SPLIT COMPLETED")
print("=" * 70)

print()
print("TRAIN")
print("  Patients :", len(train_patients))
print("  Benign   :", f"{train_benign:,}")
print("  Malignant:", f"{train_malignant:,}")
print()

print("VALIDATION")
print("  Patients :", len(val_patients))
print("  Benign   :", f"{val_benign:,}")
print("  Malignant:", f"{val_malignant:,}")
print()

print("TEST")
print("  Patients :", len(test_patients))
print("  Benign   :", f"{test_benign:,}")
print("  Malignant:", f"{test_malignant:,}")
print()

print("=" * 70)

print("TOTAL BENIGN    :", f"{total_benign:,}")
print("TOTAL MALIGNANT :", f"{total_malignant:,}")
print("TOTAL IMAGES    :", f"{total_images:,}")

print("=" * 70)

# ----------------------------------------------------------
# Expected Dataset Count
# ----------------------------------------------------------

EXPECTED_TOTAL = 277524

if total_images == EXPECTED_TOTAL:

    print()
    print("✅ SUCCESS!")
    print(
        "All 277,524 images are present."
    )

else:

    print()
    print("⚠️ WARNING!")
    print(
        f"Expected {EXPECTED_TOTAL:,} images "
        f"but found {total_images:,}."
    )

print()
print("Original dataset was NOT modified.")
print("Current dataset/ was NOT modified.")
print("This is a separate experimental dataset.")

print("=" * 70)