import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import shap

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIGURATION
# ============================================================

# SHAP is memory intensive for Hybrid B3 + Swin.
# Run SHAP on CPU to avoid CUDA Out-Of-Memory.
DEVICE = torch.device("cpu")

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/"
    "correct_malignant_01.png"
)

SAMPLE_ROOT = "results/xai_hybrid/selected_samples"

OUTPUT_DIR = (
    "results/xai_hybrid/shap/"
    "correct_malignant_01"
)

CLASS_NAMES = [
    "benign",
    "malignant"
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("HYBRID EFFICIENTNET-B3 + SWIN")
print("SHAP XAI")
print("=" * 70)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Image  :", IMAGE_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

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

print("Model loaded successfully.")


# ============================================================
# LOAD TARGET IMAGE
# ============================================================

original = Image.open(
    IMAGE_PATH
).convert("RGB")

original_np = np.array(
    original
)

input_tensor = transform(
    original
).unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(
        input_tensor
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


print("\nPrediction")

print(
    "Actual folder :",
    os.path.basename(
        os.path.dirname(IMAGE_PATH)
    )
)

print(
    "Predicted     :",
    CLASS_NAMES[predicted_class]
)

print(
    "Confidence    :",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# CREATE SMALL SHAP BACKGROUND
# ============================================================

print("\nCreating SHAP background...")

background_files = []

categories = [
    "correct_benign",
    "correct_malignant",
    "false_positive",
    "false_negative"
]

for category in categories:

    category_dir = os.path.join(
        SAMPLE_ROOT,
        category
    )

    if not os.path.exists(category_dir):
        continue

    files = sorted([
        f
        for f in os.listdir(category_dir)
        if f.lower().endswith(".png")
    ])

    if len(files) > 0:

        background_files.append(
            os.path.join(
                category_dir,
                files[0]
            )
        )


# ------------------------------------------------------------
# IMPORTANT:
# Only ONE background image to reduce memory usage.
# ------------------------------------------------------------

background_files = background_files[:1]


background_images = []

for file_path in background_files:

    img = Image.open(
        file_path
    ).convert("RGB")

    tensor = transform(
        img
    )

    background_images.append(
        tensor
    )


if len(background_images) == 0:

    raise RuntimeError(
        "No background image found."
    )


background = torch.stack(
    background_images
).to(DEVICE)


print(
    "Background images:",
    len(background_images)
)


# ============================================================
# SHAP GRADIENT EXPLAINER
# ============================================================

print("\nCreating SHAP GradientExplainer...")

explainer = shap.GradientExplainer(
    model,
    background
)

print(
    "SHAP explainer created."
)


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")
print(
    "CPU mode enabled to avoid CUDA OOM."
)

print(
    "Please wait..."
)


# Small nsamples for memory-safe execution
shap_values = explainer.shap_values(
    input_tensor,
    nsamples=20
)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

print(
    "\nProcessing SHAP output..."
)

if isinstance(shap_values, list):

    # Older SHAP format:
    # [class_0_values, class_1_values]

    class_values = shap_values[
        predicted_class
    ]

else:

    class_values = np.asarray(
        shap_values
    )

    # Possible format:
    # [batch, H, W, C, classes]

    if class_values.ndim == 5:

        if class_values.shape[-1] == 2:

            class_values = class_values[
                ..., predicted_class
            ]

        elif class_values.shape[1] == 2:

            class_values = class_values[
                :, predicted_class
            ]


class_values = np.asarray(
    class_values
)


print(
    "Raw SHAP shape:",
    class_values.shape
)


# ============================================================
# REMOVE BATCH DIMENSION
# ============================================================

if class_values.ndim == 4:

    class_values = class_values[0]


# ============================================================
# VALIDATE SHAPE
# ============================================================

if class_values.ndim != 3:

    raise RuntimeError(
        "Unexpected SHAP output shape: "
        f"{class_values.shape}"
    )


# ============================================================
# CHANNEL FORMAT
# ============================================================

# Expected:
# H x W x C
#
# If:
# C x H x W
# convert to H x W x C.

if class_values.shape[0] == 3:

    class_values = np.transpose(
        class_values,
        (1, 2, 0)
    )


# ============================================================
# CREATE SHAP IMPORTANCE MAP
# ============================================================

shap_map = np.mean(
    np.abs(class_values),
    axis=2
)


# ============================================================
# NORMALIZE SHAP MAP
# ============================================================

shap_map = np.maximum(
    shap_map,
    0
)

maximum_value = shap_map.max()

if maximum_value > 0:

    shap_map = (
        shap_map
        / maximum_value
    )


# ============================================================
# RESIZE ORIGINAL IMAGE
# ============================================================

original_resized = cv2.resize(
    original_np,
    (224, 224)
)


# ============================================================
# CREATE SHAP HEATMAP
# ============================================================

heatmap = np.uint8(
    shap_map * 255
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ============================================================
# CREATE SHAP OVERLAY
# ============================================================

original_bgr = cv2.cvtColor(
    original_resized,
    cv2.COLOR_RGB2BGR
)

overlay = cv2.addWeighted(
    original_bgr,
    0.55,
    heatmap,
    0.45,
    0
)


# ============================================================
# SAVE ORIGINAL
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "original.png"
)

cv2.imwrite(
    original_path,
    original_bgr
)


# ============================================================
# SAVE SHAP HEATMAP
# ============================================================

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "shap_heatmap.png"
)

cv2.imwrite(
    heatmap_path,
    heatmap
)


# ============================================================
# SAVE SHAP OVERLAY
# ============================================================

overlay_path = os.path.join(
    OUTPUT_DIR,
    "shap_overlay.png"
)

cv2.imwrite(
    overlay_path,
    overlay
)


# ============================================================
# CREATE COMBINED 3-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# ------------------------------------------------------------
# PANEL 1: ORIGINAL
# ------------------------------------------------------------

axes[0].imshow(
    original_resized
)

axes[0].set_title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)

axes[0].axis("off")


# ------------------------------------------------------------
# PANEL 2: SHAP HEATMAP
# ------------------------------------------------------------

axes[1].imshow(
    cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )
)

axes[1].set_title(
    "SHAP Heatmap",
    fontsize=14,
    fontweight="bold"
)

axes[1].axis("off")


# ------------------------------------------------------------
# PANEL 3: SHAP OVERLAY
# ------------------------------------------------------------

axes[2].imshow(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )
)

axes[2].set_title(
    "SHAP Overlay",
    fontsize=14,
    fontweight="bold"
)

axes[2].axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

actual_class = os.path.basename(
    os.path.dirname(
        IMAGE_PATH
    )
)

fig.suptitle(
    f"Correct Malignant | "
    f"Actual: {actual_class} | "
    f"Predicted: {CLASS_NAMES[predicted_class]} | "
    f"Confidence: {confidence * 100:.2f}%",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.90]
)


# ============================================================
# SAVE COMBINED FIGURE
# ============================================================

combined_path = os.path.join(
    OUTPUT_DIR,
    "shap_combined.png"
)

plt.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SHAP COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "Original :",
    original_path
)

print(
    "Heatmap  :",
    heatmap_path
)

print(
    "Overlay  :",
    overlay_path
)

print(
    "Combined :",
    combined_path
)

print("=" * 70)