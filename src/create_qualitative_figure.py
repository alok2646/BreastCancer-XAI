import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SAMPLE = "correct_malignant_01"

BASE_DIR = Path("results/xai_hybrid")

OUTPUT_DIR = (
    BASE_DIR
    / "qualitative_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

original_path = (
    BASE_DIR
    / "gradcam"
    / SAMPLE
    / "original.png"
)

gradcam_path = (
    BASE_DIR
    / "gradcam"
    / SAMPLE
    / "gradcam_overlay.png"
)

gradcam_pp_path = (
    BASE_DIR
    / "gradcam++"
    / SAMPLE
    / "gradcam_plus_plus_overlay.png"
)

guided_path = (
    BASE_DIR
    / "guided_backprop"
    / SAMPLE
    / "guided_backpropagation_overlay.png"
)

lime_path = (
    BASE_DIR
    / "lime"
    / SAMPLE
    / "lime_overlay.png"
)

shap_path = (
    BASE_DIR
    / "shap"
    / SAMPLE
    / "shap_overlay.png"
)


# ============================================================
# CHECK FILES
# ============================================================

files = {
    "Original Image": original_path,
    "Grad-CAM": gradcam_path,
    "Grad-CAM++": gradcam_pp_path,
    "Guided Backpropagation": guided_path,
    "LIME": lime_path,
    "SHAP": shap_path,
}


print("=" * 75)
print("QUALITATIVE XAI ANALYSIS")
print("=" * 75)

for name, path in files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(
        f"{name:25s}: OK"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise RuntimeError(
            f"Unable to read image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


images = {
    name: load_image(path)
    for name, path in files.items()
}


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


titles = [
    "Original Image",
    "Grad-CAM",
    "Grad-CAM++",
    "Guided Backpropagation",
    "LIME",
    "SHAP"
]


names = [
    "Original Image",
    "Grad-CAM",
    "Grad-CAM++",
    "Guided Backpropagation",
    "LIME",
    "SHAP"
]


# ============================================================
# DISPLAY
# ============================================================

for ax, title, name in zip(
    axes.flat,
    titles,
    names
):

    ax.imshow(
        images[name]
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.axis(
        "off"
    )


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Qualitative XAI Analysis – Hybrid EfficientNet-B3 + Swin",
    fontsize=18,
    fontweight="bold",
    y=0.98
)


fig.text(
    0.5,
    0.94,
    "Correct Malignant Sample | "
    "Actual: malignant | "
    "Predicted: malignant | "
    "Confidence: 99.88%",
    ha="center",
    fontsize=12,
    fontweight="bold"
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR
    / f"{SAMPLE}_qualitative_comparison.png"
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.91
    ]
)


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)

print(
    "QUALITATIVE FIGURE CREATED SUCCESSFULLY"
)

print("=" * 75)

print(
    "Saved:",
    output_path
)

print("=" * 75)