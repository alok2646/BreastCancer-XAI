import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

from model import BreastCancerNet
from dataset import test_dataset

# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/best_model.pth"

OUTPUT_DIR = "outputs/lime"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "benign",
    "malignant"
]

print("=" * 60)
print(" Breast Cancer XAI - LIME ")
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
# Prediction Function for LIME
# ==========================================================

def predict(images):

    model.eval()

    images = torch.tensor(
        images,
        dtype=torch.float32
    )

    images = images.permute(0, 3, 1, 2)

    images = images.to(DEVICE)

    with torch.no_grad():

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    return probabilities.cpu().numpy()


print("✓ Prediction Function Ready")
print("=" * 60)

# ==========================================================
# Load One Test Image
# ==========================================================

image, label = test_dataset[0]

rgb_image = image.permute(1, 2, 0).cpu().numpy()

rgb_image = np.clip(rgb_image, 0, 1)

print("✓ Test Image Loaded")
print("=" * 60)


# ==========================================================
# Initialize LIME Explainer
# ==========================================================

explainer = lime_image.LimeImageExplainer()

print("✓ LIME Explainer Initialized")
print("=" * 60)


# ==========================================================
# Generate LIME Explanation
# ==========================================================

explanation = explainer.explain_instance(
    rgb_image,
    predict,
    top_labels=2,
    hide_color=0,
    num_samples=1000
)

print("✓ LIME Explanation Generated")
print("=" * 60)

# ==========================================================
# Get LIME Visualization
# ==========================================================

lime_image_result, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)

lime_image_result = np.clip(lime_image_result, 0, 1)

lime_visualization = mark_boundaries(
    lime_image_result,
    mask
)

print("✓ LIME Visualization Created")
print("=" * 60)


# ==========================================================
# Save Result
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "lime_explanation.png"
)

plt.figure(figsize=(8, 8))
plt.imshow(lime_visualization)
plt.axis("off")
plt.title("LIME Explanation")

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✓ Saved : {output_path}")
print("=" * 60)

# ==========================================================
# Prediction
# ==========================================================

input_tensor = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[prediction.item()]
actual_class = CLASS_NAMES[label]

print("\n" + "=" * 60)
print("LIME Prediction Result")
print("=" * 60)

print(f"Actual Class     : {actual_class}")
print(f"Predicted Class  : {predicted_class}")
print(f"Confidence Score : {confidence.item()*100:.2f}%")

print("=" * 60)
print("LIME Completed Successfully!")
print("=" * 60)