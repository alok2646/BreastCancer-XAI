# ==========================================================
# LIME XAI ANALYSIS
# Breast Cancer Classification using Swin Transformer
#
# Categories:
# 1. Correct Benign
# 2. Correct Malignant
# 3. False Positive
# 4. False Negative
#
# 10 Images from each category
# ==========================================================


# ==========================================================
# Imports
# ==========================================================

import os
import sys
import random

import numpy as np
import pandas as pd
import torch

from PIL import Image

import matplotlib.pyplot as plt

from lime import lime_image


# ==========================================================
# Add Project Root to Python Path
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Import Model
# ==========================================================

from model import BreastCancerNet


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "patient_best_model.pth"
)


ERROR_ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "error_analysis"
)


RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "xai",
    "lime"
)


IMAGE_SIZE = 224

NUM_IMAGES = 10

NUM_SAMPLES = 500

NUM_FEATURES = 10


# ==========================================================
# ImageNet Normalization
# Same as dataset_patient.py
# ==========================================================

MEAN = np.array(
    [
        0.485,
        0.456,
        0.406
    ],
    dtype=np.float32
)


STD = np.array(
    [
        0.229,
        0.224,
        0.225
    ],
    dtype=np.float32
)


# ==========================================================
# Random Seed
# ==========================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# Create Output Directories
# ==========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# Print Configuration
# ==========================================================

print("=" * 60)

print("LIME XAI ANALYSIS")

print("=" * 60)

print()

print("Device:", DEVICE)

print("Model Path:", MODEL_PATH)

print("Results Directory:", RESULTS_DIR)

print("Images Per Category:", NUM_IMAGES)

print("LIME Samples:", NUM_SAMPLES)

print()


# ==========================================================
# Check Model File
# ==========================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        f"Model file not found:\n{MODEL_PATH}"

    )


# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)

print("LOADING SWIN TRANSFORMER MODEL")

print("=" * 60)

print()


model = BreastCancerNet()


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)


# ==========================================================
# Handle Different Checkpoint Formats
# ==========================================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]

        )

        print(
            "Loaded checkpoint using model_state_dict"
        )


    elif "state_dict" in checkpoint:

        model.load_state_dict(

            checkpoint[
                "state_dict"
            ]

        )

        print(
            "Loaded checkpoint using state_dict"
        )


    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "Loaded checkpoint directly"
        )


else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)


model.eval()


print()

print(
    "Model Loaded Successfully!"
)

print()


# ==========================================================
# LIME Prediction Function
# ==========================================================

def lime_predict(images):

    """
    LIME passes images in:

    (Batch, Height, Width, Channels)

    Range approximately:

    0 to 255
    """

    # ------------------------------------------------------
    # Convert to NumPy Array
    # ------------------------------------------------------

    images = np.array(
        images,
        dtype=np.float32
    )


    # ------------------------------------------------------
    # Convert 0-255 to 0-1 if needed
    # ------------------------------------------------------

    if images.max() > 1.0:

        images = images / 255.0


    # ------------------------------------------------------
    # Normalize Using ImageNet Mean and STD
    # ------------------------------------------------------

    images = (
        images - MEAN
    ) / STD


    # ------------------------------------------------------
    # NHWC -> NCHW
    # ------------------------------------------------------

    images = np.transpose(

        images,

        (
            0,
            3,
            1,
            2
        )

    )


    # ------------------------------------------------------
    # Convert to Tensor
    # ------------------------------------------------------

    images_tensor = torch.tensor(

        images,

        dtype=torch.float32

    ).to(
        DEVICE
    )


    # ------------------------------------------------------
    # Model Prediction
    # ------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            images_tensor
        )


        probabilities = torch.softmax(

            outputs,

            dim=1

        )


    # ------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------

    probabilities = probabilities.cpu().numpy()


    return probabilities


# ==========================================================
# Load Image for LIME
# ==========================================================

def load_image_for_lime(
    image_path
):

    """
    Load image as RGB NumPy array.

    Output Shape:

    (224, 224, 3)
    """


    # ------------------------------------------------------
    # Open Image
    # ------------------------------------------------------

    image = Image.open(

        image_path

    ).convert(
        "RGB"
    )


    # ------------------------------------------------------
    # Resize
    # ------------------------------------------------------

    image = image.resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    )


    # ------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------

    image_np = np.array(

        image,

        dtype=np.uint8

    )


    return image_np


# ==========================================================
# Get Model Prediction
# ==========================================================

def get_prediction(
    image_np
):

    """
    Get predicted class
    and confidence.
    """


    probabilities = lime_predict(

        np.expand_dims(

            image_np,

            axis=0

        )

    )


    probabilities = probabilities[
        0
    ]


    predicted_class = int(

        np.argmax(
            probabilities
        )

    )


    confidence = float(

        probabilities[
            predicted_class
        ]

    )


    return (

        predicted_class,

        confidence,

        probabilities

    )


# ==========================================================
# Convert Relative Image Path to Absolute Path
# ==========================================================

def get_absolute_image_path(
    image_path
):

    # ------------------------------------------------------
    # Replace Slashes
    # ------------------------------------------------------

    image_path = str(
        image_path
    )


    image_path = image_path.replace(
        "\\",
        os.sep
    )


    image_path = image_path.replace(
        "/",
        os.sep
    )


    # ------------------------------------------------------
    # Already Absolute
    # ------------------------------------------------------

    if os.path.isabs(
        image_path
    ):

        return image_path


    # ------------------------------------------------------
    # Relative to Project Root
    # ------------------------------------------------------

    return os.path.join(

        PROJECT_ROOT,

        image_path

    )


# ==========================================================
# Category Configuration
# ==========================================================

CATEGORIES = {

    "correct_benign": {

        "csv": "correct_benign.csv",

        "display_name": "Correct Benign"

    },


    "correct_malignant": {

        "csv": "correct_malignant.csv",

        "display_name": "Correct Malignant"

    },


    "false_positive": {

        "csv": "false_positive.csv",

        "display_name": "False Positive"

    },


    "false_negative": {

        "csv": "false_negative.csv",

        "display_name": "False Negative"

    }

}


# ==========================================================
# Process Category
# ==========================================================

def process_category(

    category_name,

    category_info

):


    # ------------------------------------------------------
    # Print Header
    # ------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(

        "Processing:",

        category_info[
            "display_name"
        ]

    )

    print(
        "=" * 60
    )


    # ------------------------------------------------------
    # CSV Path
    # ------------------------------------------------------

    csv_path = os.path.join(

        ERROR_ANALYSIS_DIR,

        category_info[
            "csv"
        ]

    )


    # ------------------------------------------------------
    # Check CSV
    # ------------------------------------------------------

    if not os.path.exists(
        csv_path
    ):

        print()

        print(
            "WARNING: CSV Not Found"
        )

        print(
            csv_path
        )

        return


    # ------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------

    dataframe = pd.read_csv(

        csv_path

    )


    # ------------------------------------------------------
    # Check Empty
    # ------------------------------------------------------

    if len(
        dataframe
    ) == 0:

        print()

        print(
            "WARNING: No samples found!"
        )

        return


    # ------------------------------------------------------
    # Select Images
    # ------------------------------------------------------

    selected_samples = dataframe.head(

        NUM_IMAGES

    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------
    # Create Category Directory
    # ------------------------------------------------------

    category_dir = os.path.join(

        RESULTS_DIR,

        category_name

    )


    os.makedirs(

        category_dir,

        exist_ok=True

    )


    # ------------------------------------------------------
    # Print Sample Count
    # ------------------------------------------------------

    print()

    print(

        "Total Samples Selected:",

        len(
            selected_samples
        )

    )


    # ======================================================
    # Process Each Image
    # ======================================================

    for index, row in selected_samples.iterrows():


        print()

        print(

            "-" * 60

        )


        print(

            f"Processing Image {index + 1}"

        )


        print(

            "-" * 60

        )


        # --------------------------------------------------
        # Get Image Path
        # --------------------------------------------------

        relative_path = row[
            "image_path"
        ]


        image_path = get_absolute_image_path(

            relative_path

        )


        # --------------------------------------------------
        # Check Image
        # --------------------------------------------------

        if not os.path.exists(
            image_path
        ):

            print()

            print(
                "WARNING: Image Not Found"
            )

            print(
                image_path
            )

            continue


        # --------------------------------------------------
        # Load Image
        # --------------------------------------------------

        image_np = load_image_for_lime(

            image_path

        )


        # --------------------------------------------------
        # Get Model Prediction
        # --------------------------------------------------

        predicted_class, confidence, probabilities = (

            get_prediction(
                image_np
            )

        )


        # --------------------------------------------------
        # True Class
        # --------------------------------------------------

        true_class = int(

            row[
                "true_label"
            ]

        )


        # --------------------------------------------------
        # Print Prediction
        # --------------------------------------------------

        print()

        print(

            "True Label:",

            true_class

        )


        print(

            "Predicted Label:",

            predicted_class

        )


        print(

            "Confidence:",

            f"{confidence:.4f}"

        )


        # ==================================================
        # Initialize LIME Explainer
        # ==================================================

        print()

        print(
            "Initializing LIME..."
        )


        explainer = lime_image.LimeImageExplainer(

            random_state=SEED

        )


        # ==================================================
        # Generate Explanation
        # ==================================================

        print(
            "Calculating LIME Explanation..."
        )


        explanation = explainer.explain_instance(

            image_np,

            lime_predict,

            top_labels=2,

            hide_color=0,

            num_samples=NUM_SAMPLES

        )


        # ==================================================
        # Get Explanation for Predicted Class
        # ==================================================

        temp, mask = explanation.get_image_and_mask(

            label=predicted_class,

            positive_only=False,

            num_features=NUM_FEATURES,

            hide_rest=False

        )


        # ==================================================
        # Prepare Image
        # ==================================================

        temp = np.clip(

            temp,

            0,

            255

        ).astype(
            np.uint8
        )


        # ==================================================
        # Create Figure
        # ==================================================

        fig, axes = plt.subplots(

            1,

            3,

            figsize=(18, 6)

        )


        # --------------------------------------------------
        # Original Image
        # --------------------------------------------------

        axes[0].imshow(

            image_np

        )


        axes[0].set_title(

            "Original Image"

        )


        axes[0].axis(
            "off"
        )


        # --------------------------------------------------
        # LIME Explanation Image
        # --------------------------------------------------

        axes[1].imshow(

            temp

        )


        axes[1].set_title(

            "LIME Explanation"

        )


        axes[1].axis(
            "off"
        )


        # --------------------------------------------------
        # LIME Mask
        # --------------------------------------------------

        axes[2].imshow(

            image_np

        )


        axes[2].imshow(

            mask,

            cmap="jet",

            alpha=0.50

        )


        axes[2].set_title(

            "LIME Important Regions"

        )


        axes[2].axis(
            "off"
        )


        # ==================================================
        # Main Title
        # ==================================================

        class_names = [

            "Benign",

            "Malignant"

        ]


        true_class_name = class_names[
            true_class
        ]


        predicted_class_name = class_names[
            predicted_class
        ]


        fig.suptitle(

            (
                f"{category_info['display_name']} | "

                f"True: {true_class_name} | "

                f"Predicted: {predicted_class_name} | "

                f"Confidence: {confidence:.4f}"
            ),

            fontsize=14

        )


        plt.tight_layout()


        # ==================================================
        # Save Figure
        # ==================================================

        save_path = os.path.join(

            category_dir,

            f"{category_name}_{index + 1}.png"

        )


        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )


        plt.close()


        # ==================================================
        # Print Saved Path
        # ==================================================

        print()

        print(

            "Saved:",

            save_path

        )


# ==========================================================
# Main Processing
# ==========================================================

print()

print(
    "=" * 60
)

print(
    "STARTING LIME ANALYSIS"
)

print(
    "=" * 60
)


# ==========================================================
# Process All Categories
# ==========================================================

for category_name, category_info in CATEGORIES.items():

    process_category(

        category_name,

        category_info

    )


# ==========================================================
# Final Message
# ==========================================================

print()

print(
    "=" * 60
)

print(
    "LIME ANALYSIS COMPLETED SUCCESSFULLY!"
)

print(
    "=" * 60
)


print()

print(

    "Results saved in:",

    RESULTS_DIR

)


print()

print(
    "Categories Processed:"
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


print()

print(
    "=" * 60
)