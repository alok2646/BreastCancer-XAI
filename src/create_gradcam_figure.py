import os
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = (
    "results/xai_hybrid/gradcam/"
    "correct_malignant_01"
)

OUTPUT_FILE = (
    "results/xai_hybrid/gradcam/"
    "correct_malignant_01/"
    "gradcam_combined.png"
)


# ============================================================
# LOAD IMAGES
# ============================================================

original = Image.open(
    os.path.join(
        INPUT_DIR,
        "original.png"
    )
)

heatmap = Image.open(
    os.path.join(
        INPUT_DIR,
        "gradcam_heatmap.png"
    )
)

overlay = Image.open(
    os.path.join(
        INPUT_DIR,
        "gradcam_overlay.png"
    )
)


# ============================================================
# CREATE SINGLE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# ============================================================
# ORIGINAL
# ============================================================

axes[0].imshow(original)

axes[0].set_title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)

axes[0].axis("off")


# ============================================================
# HEATMAP
# ============================================================

axes[1].imshow(heatmap)

axes[1].set_title(
    "Grad-CAM Heatmap",
    fontsize=14,
    fontweight="bold"
)

axes[1].axis("off")


# ============================================================
# OVERLAY
# ============================================================

axes[2].imshow(overlay)

axes[2].set_title(
    "Grad-CAM Overlay",
    fontsize=14,
    fontweight="bold"
)

axes[2].axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Correct Malignant | Actual: malignant | "
    "Predicted: malignant | Confidence: 99.88%",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.90]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("=" * 70)
print("GRAD-CAM COMBINED FIGURE CREATED")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_FILE
)