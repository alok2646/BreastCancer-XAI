"""
LIME XAI - Hybrid EfficientNet-B3 + Swin
Output:
1. Original Image
2. LIME Heatmap
3. LIME Overlay
4. Combined Figure
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/correct_malignant_01.png"
)

OUTPUT_ROOT = "results/xai_hybrid/lime"

NUM_SAMPLES = 1000
GRID_SIZE = 8


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    image_name
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HYBRID EFFICIENTNET-B3 + SWIN")
print("LIME EXPLAINABILITY")
print("=" * 70)

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
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_np = np.array(image).astype(np.float32) / 255.0

print()
print("Image shape :", image_np.shape)


# ============================================================
# MODEL PREDICTION FUNCTION FOR LIME
# ============================================================

def predict_fn(images):

    batch = []

    for img in images:

        if img.max() > 1.0:
            img = img / 255.0

        tensor = torch.tensor(
            img,
            dtype=torch.float32
        ).permute(2, 0, 1)

        batch.append(tensor)

    batch = torch.stack(batch)

    predictions = []

    with torch.no_grad():

        for start in range(0, len(batch), 8):

            mini_batch = batch[start:start + 8].to(DEVICE)

            outputs = model(mini_batch)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            predictions.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(predictions, axis=0)


# ============================================================
# ORIGINAL PREDICTION
# ============================================================

prediction = predict_fn(
    np.expand_dims(image_np, axis=0)
)[0]

predicted_class = int(np.argmax(prediction))

class_names = [
    "benign",
    "malignant"
]

predicted_name = class_names[predicted_class]

confidence = float(prediction[predicted_class])


print()
print("Prediction")
print("Actual folder :", os.path.basename(os.path.dirname(IMAGE_PATH)))
print("Predicted     :", predicted_name)
print("Confidence    : {:.2f}%".format(confidence * 100))


# ============================================================
# LIME EXPLAINER
# ============================================================

print()
print("LIME segmentation")
print("Grid :", GRID_SIZE, "x", GRID_SIZE)
print("Interpretable regions :", GRID_SIZE * GRID_SIZE)

explainer = lime_image.LimeImageExplainer(
    verbose=False
)


# ============================================================
# CUSTOM GRID SEGMENTATION
# ============================================================

def grid_segmentation(image):

    h, w, _ = image.shape

    segments = np.zeros(
        (h, w),
        dtype=np.int32
    )

    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE

    region = 0

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            y1 = row * cell_h
            y2 = (
                (row + 1) * cell_h
                if row < GRID_SIZE - 1
                else h
            )

            x1 = col * cell_w
            x2 = (
                (col + 1) * cell_w
                if col < GRID_SIZE - 1
                else w
            )

            segments[
                y1:y2,
                x1:x2
            ] = region

            region += 1

    return segments


# ============================================================
# RUN LIME
# ============================================================

print()
print("Creating LIME perturbations...")
print("Number of samples:", NUM_SAMPLES)

explanation = explainer.explain_instance(
    image_np,
    predict_fn,
    top_labels=2,
    hide_color=0,
    num_samples=NUM_SAMPLES,
    segmentation_fn=grid_segmentation
)

print()
print("Fitting local LIME model...")


# ============================================================
# GET EXPLANATION
# ============================================================

label = predicted_class

temp, mask = explanation.get_image_and_mask(
    label=label,
    positive_only=False,
    num_features=GRID_SIZE * GRID_SIZE,
    hide_rest=False
)

# Explanation weights
local_exp = dict(
    explanation.local_exp[label]
)


# ============================================================
# CREATE LIME HEATMAP
# ============================================================

segments = explanation.segments

heatmap = np.zeros_like(
    image_np[:, :, 0],
    dtype=np.float32
)

for segment_id, weight in local_exp.items():

    heatmap[
        segments == segment_id
    ] = weight


# Normalize heatmap
max_abs = np.max(
    np.abs(heatmap)
)

if max_abs > 0:
    heatmap = heatmap / max_abs


# ============================================================
# CREATE POSITIVE/NEGATIVE VISUALIZATION
# ============================================================

lime_rgb = np.zeros(
    (heatmap.shape[0], heatmap.shape[1], 3)
)

positive = heatmap > 0
negative = heatmap < 0

# Positive contribution
lime_rgb[positive, 0] = 1.0
lime_rgb[positive, 1] = 0.15

# Negative contribution
lime_rgb[negative, 2] = 1.0
lime_rgb[negative, 1] = 0.15

# Neutral
neutral = heatmap == 0
lime_rgb[neutral] = 0.05


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = (
    0.55 * image_np
    +
    0.45 * lime_rgb
)

overlay = np.clip(
    overlay,
    0,
    1
)


# ============================================================
# SAVE INDIVIDUAL IMAGES
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "original.png"
)

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "lime_heatmap.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "lime_overlay.png"
)

combined_path = os.path.join(
    OUTPUT_DIR,
    "lime_combined.png"
)


Image.fromarray(
    (image_np * 255).astype(np.uint8)
).save(original_path)

plt.imsave(
    heatmap_path,
    heatmap,
    cmap="seismic",
    vmin=-1,
    vmax=1
)

Image.fromarray(
    (overlay * 255).astype(np.uint8)
).save(overlay_path)


# ============================================================
# COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

fig.suptitle(
    "{} | Predicted: {} | Confidence: {:.2f}%".format(
        image_name,
        predicted_name,
        confidence * 100
    ),
    fontsize=16,
    fontweight="bold"
)


# Original
axes[0].imshow(image_np)
axes[0].set_title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)
axes[0].axis("off")


# Heatmap
im = axes[1].imshow(
    heatmap,
    cmap="seismic",
    vmin=-1,
    vmax=1
)

axes[1].set_title(
    "LIME Heatmap",
    fontsize=14,
    fontweight="bold"
)
axes[1].axis("off")


# Overlay
axes[2].imshow(overlay)
axes[2].set_title(
    "LIME Overlay",
    fontsize=14,
    fontweight="bold"
)
axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE PREDICTION INFO
# ============================================================

info_path = os.path.join(
    OUTPUT_DIR,
    "prediction.txt"
)

with open(
    info_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Hybrid EfficientNet-B3 + Swin\n"
    )

    f.write(
        "XAI Method: LIME\n\n"
    )

    f.write(
        f"Image: {IMAGE_PATH}\n"
    )

    f.write(
        f"Predicted class: {predicted_name}\n"
    )

    f.write(
        f"Confidence: {confidence * 100:.2f}%\n"
    )

    f.write(
        f"LIME grid: {GRID_SIZE} x {GRID_SIZE}\n"
    )

    f.write(
        f"LIME samples: {NUM_SAMPLES}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("LIME COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Original :", original_path)
print("Heatmap  :", heatmap_path)
print("Overlay  :", overlay_path)
print("Combined :", combined_path)
print("Info     :", info_path)

print("=" * 70)