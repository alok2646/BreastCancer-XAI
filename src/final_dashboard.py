import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# =====================================================
# Paths
# =====================================================

GRADCAM = "outputs/gradcam/gradcam_result.png"

GRADCAM_PLUS = "outputs/gradcam_plus/comparison.png"

LIME = "outputs/lime/lime_explanation.png"

SHAP = "outputs/shap/shap_input_image.png"

CONFUSION = "outputs/confusion_matrix.png"

OUTPUT_DIR = "outputs/final_dashboard"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path):

    img = Image.open(path)

    return np.array(img)

gradcam = load_image(GRADCAM)

gradcam_plus = load_image(GRADCAM_PLUS)

lime = load_image(LIME)

shap = load_image(SHAP)

confusion = load_image(CONFUSION)

# =====================================================
# Create Dashboard
# =====================================================

plt.figure(figsize=(18, 12))

# -------------------------
# GradCAM
# -------------------------
plt.subplot(2, 3, 1)
plt.imshow(gradcam)
plt.title("Grad-CAM", fontsize=14)
plt.axis("off")

# -------------------------
# GradCAM++
# -------------------------
plt.subplot(2, 3, 2)
plt.imshow(gradcam_plus)
plt.title("Grad-CAM++", fontsize=14)
plt.axis("off")

# -------------------------
# LIME
# -------------------------
plt.subplot(2, 3, 3)
plt.imshow(lime)
plt.title("LIME", fontsize=14)
plt.axis("off")

# -------------------------
# SHAP
# -------------------------
plt.subplot(2, 3, 4)
plt.imshow(shap)
plt.title("SHAP", fontsize=14)
plt.axis("off")

# -------------------------
# Confusion Matrix
# -------------------------
plt.subplot(2, 3, 5)
plt.imshow(confusion)
plt.title("Confusion Matrix", fontsize=14)
plt.axis("off")

# -------------------------
# Prediction Summary
# -------------------------
plt.subplot(2, 3, 6)
plt.axis("off")

summary = """
BREAST CANCER XAI

Prediction : Benign

Confidence : 100%

Methods Used

✓ Grad-CAM
✓ Grad-CAM++
✓ LIME
✓ SHAP

Model
EfficientNet-B3

Dataset
Breast Histopathology
"""

plt.text(
    0.02,
    0.98,
    summary,
    fontsize=13,
    va="top",
    family="monospace",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        boxstyle="round",
        pad=1.0
 )
)
plt.suptitle(
    "Breast Cancer Explainable AI Dashboard",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout()
# ============================================================
# Save Dashboard
# ============================================================

dashboard_path = os.path.join(
    OUTPUT_DIR,
    "breast_cancer_xai_dashboard.png"
)

plt.savefig(
    dashboard_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("✓ FINAL DASHBOARD SAVED")
print(dashboard_path)
print("=" * 60)