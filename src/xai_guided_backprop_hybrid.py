import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from captum.attr import Saliency

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/"
    "correct_malignant_01.png"
)

OUTPUT_DIR = (
    "results/xai_hybrid/guided_backprop/"
    "correct_malignant_01"
)

CLASS_NAMES = [
    "benign",
    "malignant"
]


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRANSFORM
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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HYBRID EFFICIENTNET-B3 + SWIN")
print("GUIDED BACKPROPAGATION / POSITIVE-GRADIENT SALIENCY")
print("=" * 70)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Image  :", IMAGE_PATH)


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
# LOAD IMAGE
# ============================================================

original = Image.open(
    IMAGE_PATH
).convert("RGB")

original_np = np.array(
    original
)

input_tensor = transform(
    original
).unsqueeze(0)

input_tensor = input_tensor.to(
    DEVICE
)

input_tensor.requires_grad_(True)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    prediction_output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        prediction_output,
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
        os.path.dirname(
            IMAGE_PATH
        )
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
# POSITIVE-GRADIENT SALIENCY
# ============================================================

saliency = Saliency(
    model
)


attributions = saliency.attribute(
    input_tensor,
    target=predicted_class,
    abs=False
)


# ============================================================
# POSITIVE GRADIENT ONLY
# ============================================================

attributions = torch.relu(
    attributions
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

attributions = attributions.detach().cpu()

attributions = attributions[0].permute(
    1,
    2,
    0
).numpy()


# ============================================================
# CHANNEL AGGREGATION
# ============================================================

saliency_map = np.max(
    attributions,
    axis=2
)


# ============================================================
# NORMALIZE
# ============================================================

saliency_map = np.maximum(
    saliency_map,
    0
)

if saliency_map.max() > 0:

    saliency_map = (
        saliency_map
        / saliency_map.max()
    )


# ============================================================
# RESIZE ORIGINAL
# ============================================================

original_resized = cv2.resize(
    original_np,
    (224, 224)
)


# ============================================================
# HEATMAP
# ============================================================

heatmap = np.uint8(
    saliency_map * 255
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ============================================================
# OVERLAY
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
# SAVE INDIVIDUAL FILES
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "original.png"
)

guided_path = os.path.join(
    OUTPUT_DIR,
    "guided_backpropagation.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "guided_backpropagation_overlay.png"
)


cv2.imwrite(
    original_path,
    original_bgr
)

cv2.imwrite(
    guided_path,
    heatmap
)

cv2.imwrite(
    overlay_path,
    overlay
)


# ============================================================
# COMBINED 3-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# Original

axes[0].imshow(
    original_resized
)

axes[0].set_title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)

axes[0].axis("off")


# Guided Backprop

axes[1].imshow(
    cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )
)

axes[1].set_title(
    "Guided Backpropagation",
    fontsize=14,
    fontweight="bold"
)

axes[1].axis("off")


# Overlay

axes[2].imshow(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )
)

axes[2].set_title(
    "Guided Backpropagation Overlay",
    fontsize=14,
    fontweight="bold"
)

axes[2].axis("off")


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    f"Correct Malignant | Actual: malignant | "
    f"Predicted: {CLASS_NAMES[predicted_class]} | "
    f"Confidence: {confidence * 100:.2f}%",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.90]
)


# ============================================================
# SAVE COMBINED
# ============================================================

combined_path = os.path.join(
    OUTPUT_DIR,
    "guided_backprop_combined.png"
)


plt.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("GUIDED BACKPROPAGATION COMPLETED")
print("=" * 70)

print(
    "Original :",
    original_path
)

print(
    "Guided   :",
    guided_path
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