import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import BreastCancerNet
from dataset import test_dataset

# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/best_model.pth"

OUTPUT_DIR = "outputs/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "benign",
    "malignant"
]

print("=" * 60)
print(" Breast Cancer XAI - GradCAM ")
print("=" * 60)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Output :", OUTPUT_DIR)

print("=" * 60)



# ============================================================
# Load Trained Model
# ============================================================

model = BreastCancerNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("=" * 60)
print("Model Loaded Successfully!")
print("=" * 60)


# ============================================================
# Target Layer for Grad-CAM
# ============================================================

target_layers = [
    model.backbone.features[-1]
]

print("Target Layer Selected:")
print(target_layers)
print("=" * 60)


# ============================================================
# Initialize GradCAM
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

print("GradCAM Initialized Successfully!")
print("=" * 60)

# ============================================================
# Select One Test Image
# ============================================================

image, label = test_dataset[0]

input_tensor = image.unsqueeze(0).to(DEVICE)

print("Image Loaded!")
print("=" * 60)


# ============================================================
# Generate Grad-CAM
# ============================================================

grayscale_cam = cam(
    input_tensor=input_tensor
)

grayscale_cam = grayscale_cam[0]

print("GradCAM Generated Successfully!")
print("=" * 60)

# ============================================================
# Convert Tensor to Image
# ============================================================

rgb_img = image.permute(1, 2, 0).cpu().numpy()

rgb_img = np.clip(rgb_img, 0, 1)


# ============================================================
# Overlay Grad-CAM
# ============================================================

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

print("Heatmap Overlay Created!")
print("=" * 60)


# ============================================================
# Save Result
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_result.png"
)

plt.figure(figsize=(8, 8))
plt.imshow(visualization)
plt.axis("off")
plt.title("Grad-CAM Explanation")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Saved : {output_path}")
print("=" * 60)

# ============================================================
# Prediction Details
# ============================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[predicted.item()]
actual_class = CLASS_NAMES[label]

print("\n" + "=" * 60)
print("Grad-CAM Prediction Result")
print("=" * 60)

print(f"Actual Class      : {actual_class}")
print(f"Predicted Class   : {predicted_class}")
print(f"Confidence Score  : {confidence.item()*100:.2f}%")

print("=" * 60)
print("Grad-CAM Completed Successfully!")
print("=" * 60)