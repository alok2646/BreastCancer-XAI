import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    GuidedBackpropReLUModel
)

from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image
)

from model import BreastCancerNet
from dataset import test_dataset

# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/best_model.pth"

OUTPUT_DIR = "outputs/gradcam_plus"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "benign",
    "malignant"
]

print("=" * 60)
print(" Breast Cancer XAI - GradCAM++ ")
print("=" * 60)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Output :", OUTPUT_DIR)

print("=" * 60)

# ==========================================================
# Load Trained Model
# ==========================================================

model = BreastCancerNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("✓ Model Loaded Successfully")
print("=" * 60)

# ==========================================================
# Target Layer
# ==========================================================

target_layers = [
    model.backbone.features[-1]
]

print("Target Layer Selected:")
print(target_layers)
print("=" * 60)

# ==========================================================
# Initialize GradCAM++
# ==========================================================

cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)

print("✓ GradCAM++ Initialized Successfully")

# ==========================================================
# Initialize Guided Backpropagation
# ==========================================================

gb_model = GuidedBackpropReLUModel(
    model=model,
    device=DEVICE.type
)

print("✓ Guided Backprop Initialized Successfully")
print("=" * 60)

# ==========================================================
# Load One Test Image
# ==========================================================

image, label = test_dataset[0]

input_tensor = image.unsqueeze(0).to(DEVICE)

print("✓ Test Image Loaded")
print("=" * 60)

# ==========================================================
# Generate GradCAM++
# ==========================================================

grayscale_cam = cam(
    input_tensor=input_tensor
)

grayscale_cam = grayscale_cam[0]

print("✓ GradCAM++ Generated")

# ==========================================================
# Guided Backpropagation
# ==========================================================

gb = gb_model(input_tensor, target_category=None)

# GuidedBackprop already returns HWC image
gb = np.float32(gb)

gb = gb - gb.min()
gb = gb / (gb.max() + 1e-8)

print("✓ Guided Backprop Generated")

# ==========================================================
# Original Image
# ==========================================================

rgb_img = image.permute(1, 2, 0).cpu().numpy()

rgb_img = np.clip(rgb_img, 0, 1)

# ==========================================================
# GradCAM++ Overlay
# ==========================================================

cam_image = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

print("✓ GradCAM++ Overlay Created")

# ==========================================================
# Guided GradCAM
# ==========================================================

guided_gradcam = gb * grayscale_cam[..., np.newaxis]

guided_gradcam -= guided_gradcam.min()

guided_gradcam /= (
    guided_gradcam.max() + 1e-8
)

print("✓ Guided GradCAM Created")
print("=" * 60)
# ==========================================================
# Save Images
# ==========================================================

plt.figure(figsize=(16, 12))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis("off")

# GradCAM++
plt.subplot(2, 2, 2)
plt.imshow(cam_image)
plt.title("GradCAM++")
plt.axis("off")

# Guided Backpropagation
plt.subplot(2, 2, 3)
plt.imshow(gb)
plt.title("Guided Backpropagation")
plt.axis("off")

# Guided GradCAM
plt.subplot(2, 2, 4)
plt.imshow(guided_gradcam)
plt.title("Guided GradCAM")
plt.axis("off")

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ Comparison Figure Saved : {output_path}")

# ==========================================================
# Prediction
# ==========================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[prediction.item()]
actual_class = CLASS_NAMES[label]

print("\n" + "=" * 60)
print("Advanced XAI Result")
print("=" * 60)

print(f"Actual Class      : {actual_class}")
print(f"Predicted Class   : {predicted_class}")
print(f"Confidence Score  : {confidence.item()*100:.2f}%")

print("=" * 60)
print("Advanced XAI Completed Successfully!")
print("=" * 60)
