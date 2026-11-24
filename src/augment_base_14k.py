import os
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ============================================================
# BASE PAPER STYLE AUGMENTATION
# 7,000 -> 14,000 PER CLASS
# ============================================================

SOURCE_DIR = Path("dataset_base_14k")
OUTPUT_DIR = Path("dataset_base_28k")

CLASSES = ["benign", "malignant"]

# One augmented image for every original image
augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.20, 0.20),
        shear=(-11.46, 11.46),
        scale=(0.80, 1.20)
    ),
])


for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name
    output_class = OUTPUT_DIR / class_name

    output_class.mkdir(parents=True, exist_ok=True)

    images = list(source_class.glob("*.png"))

    print(f"\nProcessing {class_name}: {len(images)} images")

    for i, image_path in enumerate(images):

        # ----------------------------------------------------
        # Copy original image
        # ----------------------------------------------------
        image = Image.open(image_path).convert("RGB")

        original_output = output_class / image_path.name
        image.save(original_output)

        # ----------------------------------------------------
        # Generate ONE augmented version
        # ----------------------------------------------------
        augmented_image = augmentation(image)

        augmented_output = (
            output_class /
            f"{image_path.stem}_aug.png"
        )

        augmented_image.save(augmented_output)

        if (i + 1) % 500 == 0:
            print(
                f"{class_name}: "
                f"{i + 1}/{len(images)} processed"
            )


# ============================================================
# FINAL COUNT
# ============================================================

print("\n========================================")
print("AUGMENTATION COMPLETED")
print("========================================")

for class_name in CLASSES:

    count = len(
        list(
            (OUTPUT_DIR / class_name).glob("*.png")
        )
    )

    print(f"{class_name}: {count}")

total = sum(
    len(list((OUTPUT_DIR / c).glob("*.png")))
    for c in CLASSES
)

print(f"Total images: {total}")
print(f"Output: {OUTPUT_DIR}")