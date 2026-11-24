import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from hybrid_model import HybridBreastCancerNet


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/hybrid_stable_best_model.pth"

IMAGE_PATH = (
    "results/xai_hybrid/selected_samples/"
    "correct_malignant/"
    "correct_malignant_01.png"
)

OUTPUT_DIR = (
    "results/xai_hybrid/gradcam/"
    "correct_malignant_01"
)

CLASS_NAMES = [
    "benign",
    "malignant"
]

IMG_SIZE = 224


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TEST TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HYBRID EFFICIENTNET-B3 + SWIN")
print("GRAD-CAM")
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


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

original = Image.open(
    IMAGE_PATH
).convert("RGB")

original_np = np.array(
    original
)

input_tensor = transform(
    original
).unsqueeze(0)

input_tensor = input_tensor.to(
    DEVICE
)


print(
    "Input shape:",
    tuple(input_tensor.shape)
)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ONLY INSIDE EFFICIENTNET-B3
# ============================================================

target_layer = None
target_name = None


for name, module in model.efficientnet.named_modules():

    if isinstance(
        module,
        torch.nn.Conv2d
    ):

        target_layer = module
        target_name = (
            "efficientnet." + name
        )


if target_layer is None:

    raise RuntimeError(
        "No Conv2d layer found inside EfficientNet-B3."
    )


print(
    "\nGrad-CAM target layer:",
    target_name
)


# ============================================================
# HOOK STORAGE
# ============================================================

activations = []
gradients = []


def forward_hook(
    module,
    input,
    output
):

    activations.append(
        output
    )


def backward_hook(
    module,
    grad_input,
    grad_output
):

    gradients.append(
        grad_output[0]
    )


forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# FORWARD
# ============================================================

model.zero_grad()

output = model(
    input_tensor
)

probabilities = F.softmax(
    output,
    dim=1
)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = probabilities[
    0,
    predicted_class
].item()


print("\nPrediction")
print(
    "Actual folder :",
    os.path.basename(
        os.path.dirname(
            IMAGE_PATH
        )
    )
)

print(
    "Predicted     :",
    CLASS_NAMES[predicted_class]
)

print(
    "Confidence    :",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# BACKWARD
# ============================================================

target_score = output[
    0,
    predicted_class
]

target_score.backward()


# ============================================================
# GET ACTIVATION + GRADIENT
# ============================================================

activation = activations[0]

gradient = gradients[0]


print(
    "\nActivation shape:",
    tuple(activation.shape)
)

print(
    "Gradient shape  :",
    tuple(gradient.shape)
)


# ============================================================
# GRAD-CAM WEIGHTS
# ============================================================

weights = gradient.mean(
    dim=(2, 3),
    keepdim=True
)


cam = (
    weights * activation
).sum(
    dim=1,
    keepdim=True
)


cam = F.relu(
    cam
)


# ============================================================
# UPSAMPLE
# ============================================================

cam = F.interpolate(
    cam,
    size=(
        IMG_SIZE,
        IMG_SIZE
    ),
    mode="bilinear",
    align_corners=False
)


cam = cam.squeeze()

cam = cam.detach().cpu().numpy()


# ============================================================
# NORMALIZE
# ============================================================

cam = cam - cam.min()

if cam.max() > 0:

    cam = cam / cam.max()


# ============================================================
# HEATMAP
# ============================================================

heatmap = np.uint8(
    cam * 255
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

original_resized = cv2.resize(
    original_np,
    (
        IMG_SIZE,
        IMG_SIZE
    )
)


original_bgr = cv2.cvtColor(
    original_resized,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    original_bgr,
    0.55,
    heatmap,
    0.45,
    0
)


# ============================================================
# SAVE
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "original.png"
)

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_heatmap.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_overlay.png"
)


cv2.imwrite(
    original_path,
    original_bgr
)

cv2.imwrite(
    heatmap_path,
    heatmap
)

cv2.imwrite(
    overlay_path,
    overlay
)


# ============================================================
# SAVE PREDICTION INFORMATION
# ============================================================

info_path = os.path.join(
    OUTPUT_DIR,
    "prediction.txt"
)


with open(
    info_path,
    "w"
) as f:

    f.write(
        "Hybrid EfficientNet-B3 + Swin Transformer\n"
    )

    f.write(
        f"Image: {IMAGE_PATH}\n"
    )

    f.write(
        f"Predicted class: "
        f"{CLASS_NAMES[predicted_class]}\n"
    )

    f.write(
        f"Confidence: "
        f"{confidence * 100:.4f}%\n"
    )

    f.write(
        f"Target layer: "
        f"{target_name}\n"
    )

    f.write(
        f"Activation shape: "
        f"{tuple(activation.shape)}\n"
    )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()

backward_handle.remove()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("GRAD-CAM COMPLETED SUCCESSFULLY")

print("=" * 70)

print(
    "Original :",
    original_path
)

print(
    "Heatmap  :",
    heatmap_path
)

print(
    "Overlay  :",
    overlay_path
)

print(
    "Info     :",
    info_path
)

print("=" * 70)