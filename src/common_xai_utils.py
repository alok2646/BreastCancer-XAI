"""
COMMON XAI UTILITIES
Hybrid EfficientNet-B3 + Swin Transformer

Purpose:
Keep model loading, image preprocessing and prediction
identical across:

- Grad-CAM
- Grad-CAM++
- Guided Backpropagation
- SHAP
- LIME
"""

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

CLASS_NAMES = [
    "benign",
    "malignant"
]

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SAME TRANSFORM FOR ALL XAI METHODS
# ============================================================

XAI_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_xai_model():

    model = HybridBreastCancerNet(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)

    # IMPORTANT:
    # All XAI methods must use eval mode.
    model.eval()

    return model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_xai_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_tensor = XAI_TRANSFORM(
        resized_image
    )

    return resized_image, image_tensor


# ============================================================
# PREDICTION
# ============================================================

def predict_xai(
    model,
    image_tensor
):

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = float(
        probabilities[
            predicted_class
        ].item()
    )

    return (
        logits,
        probabilities,
        predicted_class,
        confidence
    )


# ============================================================
# PRINT PREDICTION
# ============================================================

def print_xai_prediction(
    probabilities,
    predicted_class,
    confidence
):

    print("\nPrediction")
    print(
        "Benign probability   : "
        f"{probabilities[0].item() * 100:.4f}%"
    )

    print(
        "Malignant probability: "
        f"{probabilities[1].item() * 100:.4f}%"
    )

    print(
        "Predicted class      : "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        "Confidence           : "
        f"{confidence * 100:.4f}%"
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = (
        "results/xai_hybrid/selected_samples/"
        "correct_malignant/"
        "correct_malignant_01.png"
    )

    print("=" * 70)
    print("COMMON XAI PIPELINE TEST")
    print("=" * 70)

    print(
        "Device :",
        DEVICE
    )

    print(
        "Model  :",
        MODEL_PATH
    )

    print(
        "Image  :",
        IMAGE_PATH
    )

    print("\nLoading model...")

    model = load_xai_model()

    print(
        "Model loaded successfully."
    )

    print("\nLoading image...")

    resized_image, image_tensor = (
        load_xai_image(
            IMAGE_PATH
        )
    )

    print(
        "Image tensor shape:",
        tuple(image_tensor.shape)
    )

    print(
        "Image tensor min:",
        float(image_tensor.min())
    )

    print(
        "Image tensor max:",
        float(image_tensor.max())
    )

    (
        logits,
        probabilities,
        predicted_class,
        confidence
    ) = predict_xai(
        model,
        image_tensor
    )

    print_xai_prediction(
        probabilities,
        predicted_class,
        confidence
    )

    print("\n" + "=" * 70)
    print("COMMON XAI PIPELINE TEST COMPLETED")
    print("=" * 70)