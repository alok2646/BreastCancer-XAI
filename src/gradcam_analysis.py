# ==========================================================
# Breast Cancer XAI - Grad-CAM Analysis
# Swin Transformer-Tiny
# ==========================================================

import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from model import BreastCancerNet
from dataset_patient import test_transform

from pytorch_grad_cam import GradCAM


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/patient_best_model.pth"

CSV_DIR = "results/error_analysis"

OUTPUT_DIR = "results/xai/gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Categories
# ==========================================================

CATEGORIES = {

    "correct_benign":

        "Correct Benign",

    "correct_malignant":

        "Correct Malignant",

    "false_positive":

        "False Positive",

    "false_negative":

        "False Negative"

}


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [

    "benign",

    "malignant"

]


# ==========================================================
# Swin Transformer Reshape Function
# ==========================================================

def reshape_transform(tensor):

    """
    Convert Swin Transformer feature maps
    into CNN-style format:

    [Batch, Height, Width, Channels]

    to

    [Batch, Channels, Height, Width]
    """

    result = tensor.permute(

        0,

        3,

        1,

        2

    )

    return result


# ==========================================================
# Load Model
# ==========================================================

print("\n" + "=" * 60)

print("GRAD-CAM ANALYSIS")

print("=" * 60)

print("Device     :", DEVICE)

print("Model Path :", MODEL_PATH)

print("=" * 60)


print("\nLoading Swin Transformer Model...")


model = BreastCancerNet().to(
    DEVICE
)


model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

)


model.eval()


print("Model Loaded Successfully!")


# ==========================================================
# Target Layer
# ==========================================================

# Last Swin Transformer Block

target_layers = [

    model.backbone.features[-1][-1].norm1

]


print(

    "Target Layer:",

    target_layers

)


# ==========================================================
# Initialize Grad-CAM
# ==========================================================

cam = GradCAM(

    model=model,

    target_layers=target_layers,

    reshape_transform=reshape_transform

)


print(

    "Grad-CAM Initialized Successfully!"

)


# ==========================================================
# Load Samples from CSV
# ==========================================================

def load_samples(category):

    csv_path = os.path.join(

        CSV_DIR,

        category + ".csv"

    )


    samples = []


    with open(

        csv_path,

        "r",

        newline=""

    ) as file:


        reader = csv.DictReader(

            file

        )


        for row in reader:

            samples.append(

                row

            )


    return samples


# ==========================================================
# Load Image
# ==========================================================

def load_image(image_path):


    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # ----------------------------------------------
    # Original Image for Visualization
    # ----------------------------------------------

    resized_image = image.resize(

        (224, 224)

    )


    image_numpy = np.array(

        resized_image

    ).astype(

        np.float32

    ) / 255.0


    # ----------------------------------------------
    # Image for Model
    # ----------------------------------------------

    input_tensor = test_transform(

        image

    )


    input_tensor = input_tensor.unsqueeze(

        0

    ).to(

        DEVICE

    )


    return (

        image_numpy,

        input_tensor

    )


# ==========================================================
# Create Heatmap Overlay
# ==========================================================

def create_overlay(

    image,

    grayscale_cam

):


    # Convert heatmap to RGB

    heatmap = plt.cm.jet(

        grayscale_cam

    )[

        :, :,

        :3

    ]


    # Blend Original Image + Heatmap

    overlay = (

        0.55 * image

        +

        0.45 * heatmap

    )


    overlay = np.clip(

        overlay,

        0,

        1

    )


    return overlay


# ==========================================================
# Analyze One Sample
# ==========================================================

def analyze_sample(

    sample,

    category,

    sample_number

):


    image_path = sample[

        "image_path"

    ]


    predicted_label = int(

        sample[

            "predicted_label"

        ]

    )


    true_label = int(

        sample[

            "true_label"

        ]

    )


    confidence = float(

        sample[

            "confidence"

        ]

    )


    # ----------------------------------------------
    # Load Image
    # ----------------------------------------------

    image_numpy, input_tensor = load_image(

        image_path

    )


    # ----------------------------------------------
    # Generate Grad-CAM
    # ----------------------------------------------

    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=None

    )


    grayscale_cam = grayscale_cam[

        0

    ]


    # ----------------------------------------------
    # Create Overlay
    # ----------------------------------------------

    overlay = create_overlay(

        image_numpy,

        grayscale_cam

    )


    # ----------------------------------------------
    # Create Visualization
    # ----------------------------------------------

    fig, axes = plt.subplots(

        1,

        3,

        figsize=(15, 5)

    )


    # Original Image

    axes[0].imshow(

        image_numpy

    )

    axes[0].set_title(

        "Original Image"

    )

    axes[0].axis(

        "off"

    )


    # Grad-CAM Heatmap

    axes[1].imshow(

        grayscale_cam,

        cmap="jet"

    )

    axes[1].set_title(

        "Grad-CAM Heatmap"

    )

    axes[1].axis(

        "off"

    )


    # Overlay

    axes[2].imshow(

        overlay

    )

    axes[2].set_title(

        "Grad-CAM Overlay"

    )

    axes[2].axis(

        "off"

    )


    # ----------------------------------------------
    # Main Title
    # ----------------------------------------------

    fig.suptitle(

        f"{category} | "

        f"Actual: {CLASS_NAMES[true_label]} | "

        f"Predicted: {CLASS_NAMES[predicted_label]} | "

        f"Confidence: {confidence:.2%}",

        fontsize=14,

        fontweight="bold"

    )


    plt.tight_layout()


    # ----------------------------------------------
    # Category Directory
    # ----------------------------------------------

    category_dir = os.path.join(

        OUTPUT_DIR,

        category

    )


    os.makedirs(

        category_dir,

        exist_ok=True

    )


    # ----------------------------------------------
    # Save Image
    # ----------------------------------------------

    output_path = os.path.join(

        category_dir,

        f"{category}_{sample_number}.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(

        "Saved:",

        output_path

    )


# ==========================================================
# Process Category
# ==========================================================

def process_category(

    category,

    title

):


    print("\n" + "=" * 60)

    print(

        "Processing:",

        title

    )

    print("=" * 60)


    samples = load_samples(

        category

    )


    for index, sample in enumerate(

        samples,

        start=1

    ):


        analyze_sample(

            sample,

            category,

            index

        )


# ==========================================================
# Main Function
# ==========================================================

def main():


    print("\nStarting Grad-CAM Analysis...")


    for category, title in CATEGORIES.items():


        process_category(

            category,

            title

        )


    print("\n" + "=" * 60)

    print(

        "GRAD-CAM ANALYSIS COMPLETED SUCCESSFULLY!"

    )

    print("=" * 60)


    print(

        "\nResults saved in:",

        OUTPUT_DIR

    )


    print(

        "\nCategories processed:"

    )


    print(

        "1. Correct Benign"

    )

    print(

        "2. Correct Malignant"

    )

    print(

        "3. False Positive"

    )

    print(

        "4. False Negative"

    )


    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()
