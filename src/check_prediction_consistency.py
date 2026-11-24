import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/correct_malignant_01.png"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["benign", "malignant"]


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("PREDICTION CONSISTENCY CHECK")
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

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = image.resize(
    (224, 224)
)

image_tensor = transform(
    image
).unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probabilities = F.softmax(
        output,
        dim=1
    )[0]


predicted_class = int(
    torch.argmax(probabilities).item()
)

predicted_name = CLASS_NAMES[
    predicted_class
]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

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
    f"{predicted_name}"
)

print(
    "Confidence           : "
    f"{probabilities[predicted_class].item() * 100:.4f}%"
)

print("=" * 70)