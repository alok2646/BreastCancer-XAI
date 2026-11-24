# ==========================================================
# Breast Cancer - Error Sample Visualization
# Visualize Correct and Incorrect Predictions
# ==========================================================

import os
import csv

import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset_patient import test_transform


# ==========================================================
# Configuration
# ==========================================================

RESULTS_DIR = "results/error_analysis"

OUTPUT_DIR = "results/error_analysis/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES_PER_CATEGORY = 10


# ==========================================================
# CSV Files
# ==========================================================

CATEGORIES = {

    "correct_benign": "Correct Benign",

    "correct_malignant": "Correct Malignant",

    "false_positive": "False Positive",

    "false_negative": "False Negative"

}


# ==========================================================
# ImageNet Normalization Values
# ==========================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# ==========================================================
# Load CSV Samples
# ==========================================================

def load_samples(category):

    csv_path = os.path.join(
        RESULTS_DIR,
        category + ".csv"
    )

    samples = []

    with open(
        csv_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            samples.append(row)

    return samples


# ==========================================================
# Load Original Image
# ==========================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = test_transform(
        image
    )

    # Denormalize image
    image_tensor = (

        image_tensor * STD

        + MEAN

    )

    image_tensor = torch.clamp(
        image_tensor,
        0,
        1
    )

    # Convert CHW → HWC
    image_numpy = image_tensor.permute(
        1,
        2,
        0
    ).numpy()

    return image_numpy


# ==========================================================
# Visualize Category
# ==========================================================

def visualize_category(

    category,

    title

):

    samples = load_samples(
        category
    )

    samples = samples[
        :SAMPLES_PER_CATEGORY
    ]

    print(
        f"\nVisualizing: {title}"
    )

    # ----------------------------------------------
    # Create Figure
    # ----------------------------------------------

    fig, axes = plt.subplots(

        2,

        5,

        figsize=(18, 8)

    )

    axes = axes.flatten()


    # ----------------------------------------------
    # Display Images
    # ----------------------------------------------

    for i, sample in enumerate(
        samples
    ):

        image_path = sample[
            "image_path"
        ]

        image = load_image(
            image_path
        )

        axes[i].imshow(
            image
        )

        axes[i].axis(
            "off"
        )


        # ------------------------------------------
        # Sample Information
        # ------------------------------------------

        true_class = sample[
            "true_class"
        ]

        predicted_class = sample[
            "predicted_class"
        ]

        confidence = float(

            sample[
                "confidence"
            ]

        )


        axes[i].set_title(

            f"Actual: {true_class}\n"

            f"Predicted: {predicted_class}\n"

            f"Confidence: {confidence:.2%}",

            fontsize=10

        )


    # ----------------------------------------------
    # Main Title
    # ----------------------------------------------

    fig.suptitle(

        title,

        fontsize=18,

        fontweight="bold"

    )


    plt.tight_layout()


    # ----------------------------------------------
    # Save Figure
    # ----------------------------------------------

    output_path = os.path.join(

        OUTPUT_DIR,

        category + ".png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    print(

        "Saved:",

        output_path

    )


    plt.close()


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 60)

    print(

        "ERROR SAMPLE VISUALIZATION"

    )

    print("=" * 60)


    # ----------------------------------------------
    # Visualize All Categories
    # ----------------------------------------------

    for category, title in CATEGORIES.items():

        visualize_category(

            category,

            title

        )


    print("\n" + "=" * 60)

    print(

        "VISUALIZATION COMPLETED SUCCESSFULLY!"

    )

    print("=" * 60)

    print(

        "Results Saved In:",

        OUTPUT_DIR

    )

    print("\nGenerated Images:")

    print(
        "1. correct_benign.png"
    )

    print(
        "2. correct_malignant.png"
    )

    print(
        "3. false_positive.png"
    )

    print(
        "4. false_negative.png"
    )

    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()