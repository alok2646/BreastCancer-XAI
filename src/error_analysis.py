# ==========================================================
# Breast Cancer - Detailed Error Analysis
# Correct vs Incorrect Prediction Sample Selection
# ==========================================================

import os
import csv
import random

import torch
import torch.nn.functional as F

from model import BreastCancerNet
from dataset_patient import test_loader, test_dataset


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/patient_best_model.pth"

RESULTS_DIR = "results/error_analysis"

SAMPLES_PER_CATEGORY = 10

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================
# Class Information
# ==========================================================

CLASS_NAMES = test_dataset.classes

print("\n" + "=" * 60)
print("BREAST CANCER ERROR ANALYSIS")
print("=" * 60)

print("Device       :", DEVICE)
print("Model Path   :", MODEL_PATH)
print("Classes      :", CLASS_NAMES)
print("Samples Each :", SAMPLES_PER_CATEGORY)

print("=" * 60)


# ==========================================================
# Load Model
# ==========================================================

print("\nLoading model...")

model = BreastCancerNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model Loaded Successfully!")


# ==========================================================
# Storage for Predictions
# ==========================================================

correct_benign = []

correct_malignant = []

false_positive = []

false_negative = []


# ==========================================================
# Analyze Test Dataset
# ==========================================================

print("\nAnalyzing Test Dataset...")

dataset_index = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        batch_size = labels.size(0)


        for i in range(batch_size):

            true_label = labels[i].item()

            predicted_label = predictions[i].item()

            confidence = probabilities[
                i,
                predicted_label
            ].item()

            # Get Original Image Path
            image_path = test_dataset.samples[
                dataset_index
            ][0]


            sample_info = {

                "dataset_index": dataset_index,

                "image_path": image_path,

                "true_label": true_label,

                "predicted_label": predicted_label,

                "true_class": CLASS_NAMES[
                    true_label
                ],

                "predicted_class": CLASS_NAMES[
                    predicted_label
                ],

                "confidence": confidence

            }


            # ----------------------------------------------
            # Correct Benign
            # Actual Benign = 0
            # Predicted Benign = 0
            # ----------------------------------------------

            if (
                true_label == 0
                and predicted_label == 0
            ):

                correct_benign.append(
                    sample_info
                )


            # ----------------------------------------------
            # Correct Malignant
            # Actual Malignant = 1
            # Predicted Malignant = 1
            # ----------------------------------------------

            elif (
                true_label == 1
                and predicted_label == 1
            ):

                correct_malignant.append(
                    sample_info
                )


            # ----------------------------------------------
            # False Positive
            # Actual Benign = 0
            # Predicted Malignant = 1
            # ----------------------------------------------

            elif (
                true_label == 0
                and predicted_label == 1
            ):

                false_positive.append(
                    sample_info
                )


            # ----------------------------------------------
            # False Negative
            # Actual Malignant = 1
            # Predicted Benign = 0
            # ----------------------------------------------

            elif (
                true_label == 1
                and predicted_label == 0
            ):

                false_negative.append(
                    sample_info
                )


            dataset_index += 1


# ==========================================================
# Print Statistics
# ==========================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS RESULTS")
print("=" * 60)

print(
    "Correct Benign    :",
    len(correct_benign)
)

print(
    "Correct Malignant :",
    len(correct_malignant)
)

print(
    "False Positive    :",
    len(false_positive)
)

print(
    "False Negative    :",
    len(false_negative)
)

print("=" * 60)


# ==========================================================
# Verify Total Samples
# ==========================================================

total_samples = (

    len(correct_benign)

    + len(correct_malignant)

    + len(false_positive)

    + len(false_negative)

)

print(
    "Total Samples Analyzed:",
    total_samples
)

print(
    "Expected Test Samples :",
    len(test_dataset)
)

print("=" * 60)


# ==========================================================
# Random Sample Selection
# ==========================================================

def select_samples(samples, num_samples):

    if len(samples) <= num_samples:

        return samples

    return random.sample(
        samples,
        num_samples
    )


selected_correct_benign = select_samples(
    correct_benign,
    SAMPLES_PER_CATEGORY
)

selected_correct_malignant = select_samples(
    correct_malignant,
    SAMPLES_PER_CATEGORY
)

selected_false_positive = select_samples(
    false_positive,
    SAMPLES_PER_CATEGORY
)

selected_false_negative = select_samples(
    false_negative,
    SAMPLES_PER_CATEGORY
)


# ==========================================================
# Save Samples to CSV
# ==========================================================

def save_samples_to_csv(
    samples,
    filename
):

    file_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    with open(
        file_path,
        mode="w",
        newline=""
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[

                "dataset_index",

                "image_path",

                "true_label",

                "predicted_label",

                "true_class",

                "predicted_class",

                "confidence"

            ]

        )

        writer.writeheader()

        writer.writerows(
            samples
        )

    print(
        "Saved:",
        file_path
    )


# ==========================================================
# Save Four Categories
# ==========================================================

save_samples_to_csv(

    selected_correct_benign,

    "correct_benign.csv"

)


save_samples_to_csv(

    selected_correct_malignant,

    "correct_malignant.csv"

)


save_samples_to_csv(

    selected_false_positive,

    "false_positive.csv"

)


save_samples_to_csv(

    selected_false_negative,

    "false_negative.csv"

)


# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 60)

print(
    "ERROR ANALYSIS COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "Results saved in:",
    RESULTS_DIR
)

print()

print(
    "Generated Files:"
)

print(
    "1. correct_benign.csv"
)

print(
    "2. correct_malignant.csv"
)

print(
    "3. false_positive.csv"
)

print(
    "4. false_negative.csv"
)

print("=" * 60)