import os
import shutil
import torch
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from hybrid_model import HybridBreastCancerNet

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

TEST_DIR = "dataset_patient_split/test"

OUTPUT_DIR = "results/xai_hybrid/selected_samples"

NUM_IMAGES_PER_CATEGORY = 5

BATCH_SIZE = 4

CLASS_NAMES = [
    "benign",
    "malignant"
]

CATEGORIES = [
    "correct_benign",
    "correct_malignant",
    "false_positive",
    "false_negative"
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for category in CATEGORIES:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            category
        ),
        exist_ok=True
    )


# ============================================================
# TEST TRANSFORM
# SAME AS HYBRID TEST EVALUATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("HYBRID B3 + SWIN — XAI SAMPLE SELECTION")
print("=" * 70)

print("Device       :", DEVICE)
print("Model        :", MODEL_PATH)
print("Test Dataset :", TEST_DIR)
print("Test Images  :", len(dataset))
print("Classes      :", dataset.classes)
print("Mapping      :", dataset.class_to_idx)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Hybrid model...")


model = HybridBreastCancerNet(
    num_classes=2
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()


print("Hybrid model loaded successfully!")


# ============================================================
# PREDICTIONS
# ============================================================

records = []

processed = 0


print("\n" + "=" * 70)
print("PREDICTION STARTED")
print("=" * 70)


with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        for i in range(
            images.size(0)
        ):

            true_label = labels[i].item()

            predicted_label = predictions[i].item()

            confidence = probabilities[
                i,
                predicted_label
            ].item()


            image_path = dataset.samples[
                processed
            ][0]


            # =================================================
            # CATEGORY
            # =================================================

            if (
                true_label == 0
                and predicted_label == 0
            ):

                category = "correct_benign"


            elif (
                true_label == 1
                and predicted_label == 1
            ):

                category = "correct_malignant"


            elif (
                true_label == 0
                and predicted_label == 1
            ):

                category = "false_positive"


            else:

                category = "false_negative"


            records.append({

                "image_path":
                    image_path,

                "category":
                    category,

                "actual_class":
                    CLASS_NAMES[true_label],

                "predicted_class":
                    CLASS_NAMES[predicted_label],

                "confidence":
                    confidence

            })


            processed += 1


        if processed % 5000 == 0:

            print(
                f"Processed: "
                f"{processed}/{len(dataset)}"
            )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

all_predictions_path = os.path.join(
    OUTPUT_DIR,
    "all_xai_predictions.csv"
)

df.to_csv(
    all_predictions_path,
    index=False
)


# ============================================================
# CATEGORY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY DISTRIBUTION")
print("=" * 70)


for category in CATEGORIES:

    count = len(
        df[
            df["category"] == category
        ]
    )

    print(
        f"{category:20s}: {count}"
    )


# ============================================================
# SELECT REPRESENTATIVE SAMPLES
# ============================================================

selected_records = []


for category in CATEGORIES:

    category_df = df[
        df["category"] == category
    ].copy()


    # Highest confidence first

    category_df = category_df.sort_values(
        by="confidence",
        ascending=False
    )


    selected = category_df.head(
        NUM_IMAGES_PER_CATEGORY
    )


    category_dir = os.path.join(
        OUTPUT_DIR,
        category
    )


    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):


        source_path = row[
            "image_path"
        ]


        extension = os.path.splitext(
            source_path
        )[1]


        destination_name = (
            f"{category}_"
            f"{rank:02d}"
            f"{extension}"
        )


        destination_path = os.path.join(
            category_dir,
            destination_name
        )


        shutil.copy2(
            source_path,
            destination_path
        )


        selected_records.append({

            "category":
                category,

            "rank":
                rank,

            "source_image":
                source_path,

            "selected_image":
                destination_path,

            "actual_class":
                row["actual_class"],

            "predicted_class":
                row["predicted_class"],

            "confidence":
                row["confidence"]

        })


# ============================================================
# SAVE SELECTED SAMPLE CSV
# ============================================================

selected_df = pd.DataFrame(
    selected_records
)


selected_csv_path = os.path.join(
    OUTPUT_DIR,
    "selected_xai_samples.csv"
)


selected_df.to_csv(
    selected_csv_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XAI SAMPLE SELECTION COMPLETED")
print("=" * 70)

print(
    "\nTotal predictions:",
    len(df)
)

print("\nSelected samples:")

for category in CATEGORIES:

    count = len(
        selected_df[
            selected_df["category"]
            == category
        ]
    )

    print(
        f"{category:20s}: {count}"
    )


print("\nTotal selected:",
      len(selected_df)
)


print("\nSaved:")

print(
    "All predictions :",
    all_predictions_path
)

print(
    "Selected samples:",
    selected_csv_path
)

print(
    "Images directory:",
    OUTPUT_DIR
)

print("=" * 70)