import os
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

IMAGE_NAME = "correct_malignant_01"

BASE = "results/xai_hybrid"

ORIGINAL = os.path.join(
    BASE,
    "selected_samples",
    "correct_malignant",
    f"{IMAGE_NAME}.png"
)

GRADCAM = os.path.join(
    BASE,
    "gradcam",
    IMAGE_NAME,
    "gradcam_overlay.png"
)

GRADCAM_PLUS = os.path.join(
    BASE,
    "gradcam++",
    IMAGE_NAME,
    "gradcam_plus_plus_overlay.png"
)

GUIDED = os.path.join(
    BASE,
    "guided_backprop",
    IMAGE_NAME,
    "guided_backpropagation_overlay.png"
)

LIME = os.path.join(
    BASE,
    "lime",
    IMAGE_NAME,
    "lime_overlay.png"
)

SHAP = os.path.join(
    BASE,
    "shap",
    IMAGE_NAME,
    "shap_overlay.png"
)


OUTPUT_DIR = os.path.join(
    BASE,
    "xai_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{IMAGE_NAME}_xai_comparison.png"
)


# ============================================================
# CHECK FILES
# ============================================================

files = {
    "Original": ORIGINAL,
    "Grad-CAM": GRADCAM,
    "Grad-CAM++": GRADCAM_PLUS,
    "Guided Backpropagation": GUIDED,
    "LIME": LIME,
    "SHAP": SHAP
}

print("=" * 80)
print("FINAL XAI COMPARISON")
print("Hybrid EfficientNet-B3 + Swin")
print("=" * 80)

for name, path in files.items():

    if not os.path.exists(path):
        print(f"ERROR: {name} file not found:")
        print(path)
        raise FileNotFoundError(path)

    print(f"{name:<25}: OK")


# ============================================================
# LOAD IMAGES
# ============================================================

images = {}

for name, path in files.items():
    images[name] = Image.open(path).convert("RGB")


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 11)
)

fig.suptitle(
    "XAI Comparison - Hybrid EfficientNet-B3 + Swin\n"
    "Correct Malignant Sample | Predicted: Malignant | Confidence: 99.88%",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# DISPLAY
# ============================================================

titles = [
    "Original Image",
    "Grad-CAM",
    "Grad-CAM++",
    "Guided Backpropagation",
    "LIME",
    "SHAP"
]

order = [
    "Original",
    "Grad-CAM",
    "Grad-CAM++",
    "Guided Backpropagation",
    "LIME",
    "SHAP"
]


for ax, title, key in zip(
    axes.flat,
    titles,
    order
):

    ax.imshow(images[key])

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    ax.axis("off")


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.92]
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 80)
print("XAI COMPARISON CREATED SUCCESSFULLY")
print("=" * 80)

print("Saved:")
print(OUTPUT_FILE)

print("=" * 80)