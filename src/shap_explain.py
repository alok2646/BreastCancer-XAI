import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shap

from model import BreastCancerNet
from dataset import test_dataset

# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/best_model.pth"

OUTPUT_DIR = "outputs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "benign",
    "malignant"
]

print("=" * 60)
print(" Breast Cancer XAI - SHAP ")
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

print("✓ Model Loaded Successfully")
print("=" * 60)

# ============================================================
# Background Images for SHAP
# ============================================================

print("Loading Background Images...")

background_images = []

for i in range(2):
    img, _ = test_dataset[i]
    background_images.append(img)

background = torch.stack(background_images).to(DEVICE)

print("Background Shape :", background.shape)

# ============================================================
# Test Image
# ============================================================

image, label = test_dataset[20]

input_tensor = image.unsqueeze(0).to(DEVICE)

print("Test Image Loaded Successfully!")
print("=" * 60)

# ============================================================
# Initialize SHAP GradientExplainer
# ============================================================

print("Initializing SHAP Explainer...")

explainer = shap.GradientExplainer(
    model,
    background
)

print("✓ SHAP Explainer Initialized Successfully")
print("=" * 60)

# ============================================================
# Generate SHAP Values
# ============================================================

print("Generating SHAP Values...")

torch.cuda.empty_cache()

shap_values = explainer.shap_values(
    input_tensor,
    nsamples=5
)

print("✓ SHAP Values Generated Successfully")
print("=" * 60)

# Convert input tensor to numpy
image_np = image.permute(1, 2, 0).cpu().numpy()

image_np = np.clip(image_np, 0, 1)

# Print SHAP information
print("Input Image Shape :", image_np.shape)

if isinstance(shap_values, list):
    print("Number of Classes :", len(shap_values))
    print("SHAP Shape :", np.array(shap_values[0]).shape)
else:
    print("SHAP Shape :", np.array(shap_values).shape)

print("=" * 60)

# ============================================================
# Prepare SHAP Values for Visualization
# ============================================================

print("Preparing SHAP Visualization...")

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_to_plot = shap_values[0]
else:
    shap_to_plot = shap_values

# Convert to numpy if needed
if isinstance(shap_to_plot, torch.Tensor):
    shap_to_plot = shap_to_plot.detach().cpu().numpy()

# Remove batch dimension if present
if len(shap_to_plot.shape) == 4:
    shap_to_plot = shap_to_plot[0]

print("Visualization Shape :", shap_to_plot.shape)
print("=" * 60)

# ============================================================
# Save SHAP Visualization
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(image_np)

plt.title("Input Image")

plt.axis("off")

output_path = os.path.join(
    OUTPUT_DIR,
    "shap_input_image.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ Image Saved : {output_path}")

# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[prediction.item()]
actual_class = CLASS_NAMES[label]

print("\n" + "=" * 60)
print("SHAP Prediction Result")
print("=" * 60)

print(f"Actual Class     : {actual_class}")
print(f"Predicted Class  : {predicted_class}")
print(f"Confidence Score : {confidence.item()*100:.2f}%")

print("=" * 60)
print("SHAP Completed Successfully!")
print("=" * 60)