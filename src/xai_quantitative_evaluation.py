"""
QUANTITATIVE XAI EVALUATION
Hybrid EfficientNet-B3 + Swin

Metric:
Deletion Faithfulness

For each XAI method:
1. Load original image
2. Load explanation heatmap
3. Find important regions
4. Mask important regions
5. Re-run model
6. Measure confidence drop

Methods:
- Grad-CAM
- Grad-CAM++
- Guided Backpropagation
- LIME
- SHAP
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/correct_malignant_01.png"
)

IMAGE_NAME = "correct_malignant_01"

BASE = "results/xai_hybrid"

OUTPUT_DIR = os.path.join(
    BASE,
    "quantitative"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"{IMAGE_NAME}_deletion_faithfulness.csv"
)


# ============================================================
# XAI HEATMAP PATHS
# ============================================================

HEATMAPS = {

    "Grad-CAM":
        os.path.join(
            BASE,
            "gradcam",
            IMAGE_NAME,
            "gradcam_heatmap.png"
        ),

    "Grad-CAM++":
        os.path.join(
            BASE,
            "gradcam++",
            IMAGE_NAME,
            "gradcam_plus_plus_heatmap.png"
        ),

    "Guided Backpropagation":
        os.path.join(
            BASE,
            "guided_backprop",
            IMAGE_NAME,
            "guided_backpropagation.png"
        ),

    "LIME":
        os.path.join(
            BASE,
            "lime",
            IMAGE_NAME,
            "lime_heatmap.png"
        ),

    "SHAP":
        os.path.join(
            BASE,
            "shap",
            IMAGE_NAME,
            "shap_heatmap.png"
        )
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("QUANTITATIVE XAI EVALUATION")
print("DELETION FAITHFULNESS")
print("=" * 75)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Image  :", IMAGE_PATH)


model = HybridBreastCancerNet(
    num_classes=2,
    dropout=0.30,
    pretrained=False
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

# IMPORTANT:
# Hybrid XAI pipeline uses 224x224 model input.
# Original selected image is 50x50.
# Resize here so quantitative evaluation uses
# the same input size as Grad-CAM, Grad-CAM++,
# Guided Backpropagation, LIME and SHAP.

image = image.resize(
    (224, 224),
    Image.Resampling.BILINEAR
)

image_np = np.array(
    image
).astype(np.float32) / 255.0


print()
print("Original image size : 50 x 50")
print("Model input size    :", image_np.shape)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_array):

    tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    ).permute(2, 0, 1)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probability = F.softmax(
            output,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(probability)
    )

    confidence = float(
        probability[predicted_class]
    )

    return predicted_class, confidence


# ============================================================
# ORIGINAL PREDICTION
# ============================================================

original_class, original_confidence = predict(
    image_np
)

class_names = [
    "benign",
    "malignant"
]

print()
print("Original prediction")
print("-------------------")
print(
    "Predicted :",
    class_names[original_class]
)

print(
    "Confidence: {:.4f}%".format(
        original_confidence * 100
    )
)


# ============================================================
# LOAD HEATMAP
# ============================================================

def load_heatmap(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Heatmap not found: {path}"
        )

    heatmap = Image.open(
        path
    ).convert("RGB")

    heatmap = np.array(
        heatmap
    ).astype(np.float32)

    # Convert RGB to grayscale
    heatmap = (
        0.299 * heatmap[:, :, 0]
        +
        0.587 * heatmap[:, :, 1]
        +
        0.114 * heatmap[:, :, 2]
    )

    # Normalize
    heatmap = (
        heatmap - heatmap.min()
    ) / (
        heatmap.max()
        -
        heatmap.min()
        +
        1e-8
    )

    return heatmap


# ============================================================
# DELETION FUNCTION
# ============================================================
def delete_top_regions(
    image,
    importance,
    percentage
):
    h, w = image.shape[:2]

    # ============================================================
    # RESIZE XAI HEATMAP TO MODEL INPUT SIZE
    # ============================================================
    # Grad-CAM / Grad-CAM++ / Guided Backpropagation
    # may already produce 224x224 maps.
    #
    # LIME produces a 50x50 explanation map.
    # Therefore resize any mismatched heatmap to 224x224.
    # ============================================================

    if importance.shape != (h, w):
        importance = np.array(
            Image.fromarray(
                importance.astype(np.float32),
                mode="F"
            ).resize(
                (w, h),
                Image.Resampling.BILINEAR
            )
        )

    print(
        f"XAI heatmap resized to: {importance.shape}"
    )

    # ============================================================
    # NUMBER OF PIXELS TO DELETE
    # ============================================================

    total_pixels = h * w

    number_to_delete = int(
        total_pixels * percentage
    )

    # ============================================================
    # FIND MOST IMPORTANT PIXELS
    # ============================================================

    flat_importance = importance.flatten()

    indices = np.argsort(
        flat_importance
    )[::-1]

    selected_indices = indices[
        :number_to_delete
    ]

    # ============================================================
    # CREATE MASK
    # ============================================================

    mask = np.zeros(
        total_pixels,
        dtype=bool
    )

    mask[
        selected_indices
    ] = True

    mask = mask.reshape(
        h,
        w
    )

    # ============================================================
    # CREATE PERTURBED IMAGE
    # ============================================================

    perturbed = image.copy()

    # Replace important regions
    # with mean image intensity.

    mean_pixel = np.mean(
        image,
        axis=(0, 1)
    )

    perturbed[mask] = mean_pixel

    return perturbed

# ============================================================
# DELETION LEVELS
# ============================================================

DELETION_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]


# ============================================================
# EVALUATE EACH XAI METHOD
# ============================================================

results = []


for method, heatmap_path in HEATMAPS.items():

    print()
    print("=" * 75)
    print(method)
    print("=" * 75)

    print(
        "Heatmap:",
        heatmap_path
    )

    importance = load_heatmap(
        heatmap_path
    )

    method_values = []

    for percentage in DELETION_LEVELS:

        perturbed_image = delete_top_regions(
            image_np,
            importance,
            percentage
        )

        predicted_class, confidence = predict(
            perturbed_image
        )

        confidence_drop = (
            original_confidence
            -
            confidence
        )

        results.append(
            {
                "image": IMAGE_NAME,
                "method": method,
                "deletion_percentage":
                    int(percentage * 100),
                "original_confidence":
                    original_confidence,
                "perturbed_confidence":
                    confidence,
                "confidence_drop":
                    confidence_drop,
                "predicted_class":
                    class_names[predicted_class]
            }
        )

        method_values.append(
            confidence_drop
        )

        print(
            "Delete {:>3}% | "
            "Confidence: {:.4f}% | "
            "Drop: {:.4f}%".format(
                int(percentage * 100),
                confidence * 100,
                confidence_drop * 100
            )
        )


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "image",
    "method",
    "deletion_percentage",
    "original_confidence",
    "perturbed_confidence",
    "confidence_drop",
    "predicted_class"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("QUANTITATIVE XAI EVALUATION COMPLETED")
print("=" * 75)

print()
print("Saved:")
print(OUTPUT_CSV)

print()
print(
    "Interpretation:"
)

print(
    "Higher confidence drop after deleting"
)

print(
    "important regions indicates stronger"
)

print(
    "deletion-based faithfulness for this sample."
)

print("=" * 75)