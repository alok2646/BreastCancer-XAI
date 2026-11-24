import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
    "results/xai_hybrid/gradcam++/"
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
print("GRAD-CAM++")
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
# TARGET LAYER
# ============================================================

target_layer = model.efficientnet.features[-1]

print(
    "Target layer  :",
    "efficientnet.features[-1]"
)


# ============================================================
# GRAD-CAM++
# ============================================================

cam = GradCAMPlusPlus(
    model=model,
    target_layers=[target_layer]
)


targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


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
    grayscale_cam * 255
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

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_plus_plus_heatmap.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_plus_plus_overlay.png"
)


cv2.imwrite(
    original_path,
    original_bgr
)

cv2.imwrite(
    heatmap_path,
    heatmap
)

cv2.imwrite(
    overlay_path,
    overlay
)


# ============================================================
# COMBINED IEEE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(
    original_resized
)

axes[0].set_title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)

axes[0].axis("off")


axes[1].imshow(
    cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )
)

axes[1].set_title(
    "Grad-CAM++ Heatmap",
    fontsize=14,
    fontweight="bold"
)

axes[1].axis("off")


axes[2].imshow(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )
)

axes[2].set_title(
    "Grad-CAM++ Overlay",
    fontsize=14,
    fontweight="bold"
)

axes[2].axis("off")


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


combined_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_plus_plus_combined.png"
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
print("GRAD-CAM++ COMPLETED SUCCESSFULLY")
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