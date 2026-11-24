# ==========================================================
# Breast Cancer XAI - SHAP Analysis
# Swin Transformer
# IEEE Paper Explainability Pipeline
# ==========================================================

import os
import csv
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from PIL import Image

from model import BreastCancerNet
from dataset_patient import test_dataset


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/patient_best_model.pth"

OUTPUT_DIR = "results/xai/shap"

ERROR_ANALYSIS_DIR = "results/error_analysis"

NUM_SAMPLES = 10

BACKGROUND_SAMPLES = 20

CLASS_NAMES = [
    "benign",
    "malignant"
]


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Print Header
# ==========================================================

print("=" * 60)
print("BREAST CANCER XAI - SHAP ANALYSIS")
print("=" * 60)

print("Device:", DEVICE)
print("Model Path:", MODEL_PATH)

print("=" * 60)


# ==========================================================
# Load Model
# ==========================================================

print("\nLoading Swin Transformer Model...")

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
# Load CSV Samples
# ==========================================================

def load_csv_samples(csv_path):

    samples = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            samples.append(row)

    return samples


# ==========================================================
# Load Image
# ==========================================================

def load_image(image_path):

    # Convert Windows / Linux path format

    image_path = image_path.replace(
        "\\",
        os.sep
    )

    image_path = image_path.replace(
        "/",
        os.sep
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# ==========================================================
# Get Dataset Transform
# ==========================================================

test_transform = test_dataset.transform


# ==========================================================
# Prepare Background Dataset
# ==========================================================

print("\nPreparing SHAP Background Dataset...")

background_images = []

background_indices = np.linspace(
    0,
    len(test_dataset) - 1,
    BACKGROUND_SAMPLES,
    dtype=int
)


for index in background_indices:

    image, label = test_dataset[index]

    background_images.append(image)


background_tensor = torch.stack(
    background_images
).to(DEVICE)


print(
    "Background Samples:",
    len(background_images)
)


# ==========================================================
# Initialize SHAP Explainer
# ==========================================================

print("\nInitializing SHAP GradientExplainer...")

explainer = shap.GradientExplainer(
    model,
    background_tensor
)


print(
    "SHAP Explainer Initialized Successfully!"
)


# ==========================================================
# Extract SHAP Values Robustly
# Supports Different SHAP Versions
# ==========================================================

def extract_shap_for_class(
    shap_values,
    predicted_class
):

    # ------------------------------------------------------
    # Older SHAP Version
    # Returns List
    # ------------------------------------------------------

    if isinstance(shap_values, list):

        shap_image = np.array(
            shap_values[predicted_class]
        )

        # Remove batch dimension

        if shap_image.ndim == 4:

            shap_image = shap_image[0]

        return shap_image


    # ------------------------------------------------------
    # Newer SHAP Version
    # Returns NumPy Array
    # ------------------------------------------------------

    shap_array = np.array(
        shap_values
    )


    print(
        "Raw SHAP Shape:",
        shap_array.shape
    )


    # ------------------------------------------------------
    # Case 1
    # Shape:
    # (Batch, Channels, Height, Width, Classes)
    # Example:
    # (1, 3, 224, 224, 2)
    # ------------------------------------------------------

    if shap_array.ndim == 5:

        shap_image = shap_array[
            0,
            :,
            :,
            :,
            predicted_class
        ]

        return shap_image


    # ------------------------------------------------------
    # Case 2
    # Shape:
    # (Classes, Batch, Channels, Height, Width)
    # ------------------------------------------------------

    if shap_array.ndim == 5:

        shap_image = shap_array[
            predicted_class,
            0
        ]

        return shap_image


    # ------------------------------------------------------
    # Case 3
    # Shape:
    # (Batch, Channels, Height, Width)
    # ------------------------------------------------------

    if shap_array.ndim == 4:

        shap_image = shap_array[
            0
        ]

        return shap_image


    # ------------------------------------------------------
    # Case 4
    # Shape:
    # (Channels, Height, Width)
    # ------------------------------------------------------

    if shap_array.ndim == 3:

        return shap_array


    # ------------------------------------------------------
    # Unknown Shape
    # ------------------------------------------------------

    raise ValueError(
        f"Unsupported SHAP Shape: {shap_array.shape}"
    )


# ==========================================================
# Convert SHAP Image to Heatmap
# ==========================================================

def create_shap_heatmap(
    shap_image
):

    # ------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------

    shap_image = np.array(
        shap_image
    )


    # ------------------------------------------------------
    # Expected Shape:
    # Channels, Height, Width
    # ------------------------------------------------------

    if shap_image.ndim == 3:

        # If Channel First

        if shap_image.shape[0] == 3:

            heatmap = np.mean(
                np.abs(shap_image),
                axis=0
            )


        # If Channel Last

        elif shap_image.shape[-1] == 3:

            heatmap = np.mean(
                np.abs(shap_image),
                axis=-1
            )

        else:

            heatmap = np.mean(
                np.abs(shap_image),
                axis=0
            )


    elif shap_image.ndim == 2:

        heatmap = np.abs(
            shap_image
        )


    else:

        raise ValueError(
            f"Unsupported SHAP Image Shape: "
            f"{shap_image.shape}"
        )


    # ------------------------------------------------------
    # Normalize Heatmap
    # ------------------------------------------------------

    heatmap = heatmap - np.min(
        heatmap
    )


    max_value = np.max(
        heatmap
    )


    if max_value > 0:

        heatmap = heatmap / max_value


    return heatmap


# ==========================================================
# Process One Category
# ==========================================================

def process_category(category):

    print("\n" + "=" * 60)

    print(
        f"Processing: {category}"
    )

    print("=" * 60)


    # ------------------------------------------------------
    # CSV Path
    # ------------------------------------------------------

    csv_path = os.path.join(
        ERROR_ANALYSIS_DIR,
        f"{category}.csv"
    )


    # ------------------------------------------------------
    # Check CSV
    # ------------------------------------------------------

    if not os.path.exists(csv_path):

        print(
            f"CSV Not Found: {csv_path}"
        )

        return


    # ------------------------------------------------------
    # Load Samples
    # ------------------------------------------------------

    samples = load_csv_samples(
        csv_path
    )


    samples = samples[
        :NUM_SAMPLES
    ]


    print(
        "Total Samples Selected:",
        len(samples)
    )


    # ------------------------------------------------------
    # Create Category Directory
    # ------------------------------------------------------

    category_dir = os.path.join(
        OUTPUT_DIR,
        category
    )


    os.makedirs(
        category_dir,
        exist_ok=True
    )


    # ======================================================
    # Process Samples
    # ======================================================

    for sample_index, sample in enumerate(samples):

        print(
            f"\nProcessing Image "
            f"{sample_index + 1}"
        )


        # --------------------------------------------------
        # Image Path
        # --------------------------------------------------

        image_path = sample[
            "image_path"
        ]


        # --------------------------------------------------
        # Load Original Image
        # --------------------------------------------------

        image = load_image(
            image_path
        )


        # --------------------------------------------------
        # Apply Dataset Transform
        # --------------------------------------------------

        image_tensor = test_transform(
            image
        )


        image_tensor = image_tensor.unsqueeze(
            0
        ).to(DEVICE)


        # --------------------------------------------------
        # Model Prediction
        # --------------------------------------------------

        with torch.no_grad():

            output = model(
                image_tensor
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


        confidence = probabilities[
            0,
            predicted_class
        ].item()


        # --------------------------------------------------
        # Calculate SHAP Values
        # --------------------------------------------------

        print(
            "Calculating SHAP Values..."
        )


        shap_values = explainer.shap_values(
            image_tensor
        )


        # --------------------------------------------------
        # Debug Information
        # --------------------------------------------------

        print(
            "SHAP Type:",
            type(shap_values)
        )


        if isinstance(shap_values, list):

            print(
                "Number of Classes:",
                len(shap_values)
            )


            for class_index, value in enumerate(
                shap_values
            ):

                print(
                    f"Class {class_index} Shape:",
                    np.array(value).shape
                )


        else:

            print(
                "SHAP Array Shape:",
                np.array(shap_values).shape
            )


        # --------------------------------------------------
        # Extract SHAP for Predicted Class
        # --------------------------------------------------

        shap_image = extract_shap_for_class(
            shap_values,
            predicted_class
        )


        print(
            "Selected SHAP Image Shape:",
            shap_image.shape
        )


        # --------------------------------------------------
        # Convert Image for Visualization
        # --------------------------------------------------

        image_np = image_tensor[
            0
        ].detach().cpu().numpy()


        image_np = np.transpose(
            image_np,
            (1, 2, 0)
        )


        # --------------------------------------------------
        # Undo ImageNet Normalization
        # --------------------------------------------------

        mean = np.array(
            [
                0.485,
                0.456,
                0.406
            ]
        )


        std = np.array(
            [
                0.229,
                0.224,
                0.225
            ]
        )


        image_np = (
            image_np * std
        ) + mean


        image_np = np.clip(
            image_np,
            0,
            1
        )


        # --------------------------------------------------
        # Create SHAP Heatmap
        # --------------------------------------------------

        shap_heatmap = create_shap_heatmap(
            shap_image
        )


        print(
            "Heatmap Shape:",
            shap_heatmap.shape
        )


        # --------------------------------------------------
        # Save Path
        # --------------------------------------------------

        save_path = os.path.join(
            category_dir,
            f"{category}_{sample_index + 1}.png"
        )


        # ==================================================
        # Create Figure
        # ==================================================

        plt.figure(
            figsize=(12, 5)
        )


        # --------------------------------------------------
        # Original Image
        # --------------------------------------------------

        plt.subplot(
            1,
            2,
            1
        )


        plt.imshow(
            image_np
        )


        plt.title(

            f"Original Image\n"
            f"True: {sample['true_class']} | "
            f"Predicted: {sample['predicted_class']}"
        )


        plt.axis(
            "off"
        )


        # --------------------------------------------------
        # SHAP Explanation
        # --------------------------------------------------

        plt.subplot(
            1,
            2,
            2
        )


        plt.imshow(
            image_np
        )


        plt.imshow(
            shap_heatmap,
            cmap="jet",
            alpha=0.5
        )


        plt.title(

            f"SHAP Explanation\n"
            f"Predicted: "
            f"{CLASS_NAMES[predicted_class]} | "
            f"Confidence: "
            f"{confidence:.4f}"
        )


        plt.axis(
            "off"
        )


        # --------------------------------------------------
        # Save Figure
        # --------------------------------------------------

        plt.tight_layout()


        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Saved:",
            save_path
        )


# ==========================================================
# Categories
# ==========================================================

categories = [

    "correct_benign",

    "correct_malignant",

    "false_positive",

    "false_negative"

]


# ==========================================================
# Start Analysis
# ==========================================================

print(
    "\nStarting SHAP Analysis..."
)


for category in categories:

    process_category(
        category
    )


# ==========================================================
# Completion Message
# ==========================================================

print(
    "\n" + "=" * 60
)


print(
    "SHAP ANALYSIS COMPLETED SUCCESSFULLY!"
)


print(
    "=" * 60
)


print(
    "\nResults saved in:",
    OUTPUT_DIR
)


print(
    "\nCategories Processed:"
)


for index, category in enumerate(

    categories,

    start=1

):

    print(
        f"{index}. {category}"
    )


print(
    "\n" + "=" * 60
)