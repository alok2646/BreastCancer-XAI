
from pathlib import Path
import json
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from efficientnet_b3_only import EfficientNetB3Only

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = ROOT / "dataset_patient_split" / "test"

BACKBONE_CHECKPOINT = (
    ROOT / "saved_models" / "efficientnet_b3_best_model.pth"
)

HYBRID_CHECKPOINT = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "gpu_hybrid_tuned_run" / "best_hybrid_model.pth"
)

OUT = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "xai_explanations"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_QUBITS = 4

# Limit runtime for the first run.
MAX_IMAGES = 2
LIME_SAMPLES = 100
SHAP_SAMPLES = 40
SHAP_GROUPS = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

for folder in ("gradcam", "gradcam_plus_plus", "shap", "lime"):
    (OUT / folder).mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable. Activate venv_quantum.")

if not BACKBONE_CHECKPOINT.exists():
    raise FileNotFoundError(BACKBONE_CHECKPOINT)

if not HYBRID_CHECKPOINT.exists():
    raise FileNotFoundError(HYBRID_CHECKPOINT)

print("GPU:", torch.cuda.get_device_name(0))
print("Output:", OUT)


# =========================================================
# QUANTUM CIRCUIT: MATCH TRAINING IMPLEMENTATION
# =========================================================
def rotation(theta, axis):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)

    if axis == "Y":
        real = torch.stack([c, -s, s, c], dim=-1)
        real = real.reshape(*theta.shape, 2, 2)
        return torch.complex(real, torch.zeros_like(real))

    real = torch.stack([c, z, z, c], dim=-1)
    real = real.reshape(*theta.shape, 2, 2)

    if axis == "X":
        imag = torch.stack([z, -s, -s, z], dim=-1)
    else:
        imag = torch.stack([-s, z, z, s], dim=-1)

    imag = imag.reshape(*theta.shape, 2, 2)
    return torch.complex(real, imag)


def apply_gate(state, matrix, wire):
    batch = state.shape[0]
    shaped = state.reshape(batch, 2, 2, 2, 2)
    moved = shaped.movedim(wire + 1, -1)
    pairs = moved.reshape(batch, -1, 2)

    result = torch.matmul(pairs, matrix.transpose(-1, -2))

    return (
        result.reshape(moved.shape)
        .movedim(-1, wire + 1)
        .reshape(batch, 16)
    )


def apply_cnot(state, control, target):
    indices = torch.arange(16, device=state.device)

    control_bit = 1 << (3 - control)
    target_bit = 1 << (3 - target)

    permutation = indices.clone()
    mask = (indices & control_bit) != 0
    permutation[mask] = indices[mask] ^ target_bit

    return state.index_select(1, permutation)


class GPUQuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, 4, 3))

    def forward(self, angles):
        batch = angles.shape[0]
        zero = torch.zeros(
            batch, dtype=torch.long, device=angles.device
        )

        state = F.one_hot(zero, 16).to(torch.complex64)

        # Classical-to-quantum angle encoding
        for wire in range(4):
            state = apply_gate(
                state, rotation(angles[:, wire], "Y"), wire
            )

        # Trainable RX, RY, RZ rotations
        for wire in range(4):
            for k, axis in enumerate(("X", "Y", "Z")):
                state = apply_gate(
                    state, rotation(self.weights[0, wire, k], axis),
                    wire
                )

        # Entanglement ring
        for wire in range(4):
            state = apply_cnot(state, wire, (wire + 1) % 4)

        probabilities = state.abs().square()
        basis = torch.arange(16, device=angles.device)
        measurements = []

        for wire in range(4):
            bits = (basis >> (3 - wire)) & 1
            signs = (1 - 2 * bits).to(probabilities.dtype)
            measurements.append(probabilities @ signs)

        return torch.stack(measurements, dim=1)


# =========================================================
# HYBRID CLASSIFIER HEAD
# =========================================================
class HybridHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("feature_mean", torch.zeros(1536))
        self.register_buffer("feature_std", torch.ones(1536))

        self.projection = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

        self.quantum = GPUQuantumCircuit()
        self.classifier = nn.Linear(4, 2)

    def forward(self, features):
        features = (
            features - self.feature_mean
        ) / self.feature_std.clamp_min(1e-6)

        angles = torch.pi * torch.tanh(self.projection(features))
        measurements = self.quantum(angles)

        return self.classifier(measurements)


# =========================================================
# LOAD CHECKPOINTS WITHOUT MODIFYING THEM
# =========================================================
print("Loading EfficientNet-B3 checkpoint...")

backbone = EfficientNetB3Only(num_classes=2, dropout=0.30)
backbone_state = torch.load(
    BACKBONE_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)
backbone.load_state_dict(backbone_state, strict=True)

print("Loading hybrid checkpoint...")

checkpoint = torch.load(
    HYBRID_CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

head = HybridHead()
head.load_state_dict(checkpoint["model_state_dict"], strict=True)

if checkpoint.get("feature_dim", 1536) != 1536:
    raise ValueError("Hybrid checkpoint feature dimension is not 1536.")

if checkpoint.get("n_qubits", 4) != 4:
    raise ValueError("Hybrid checkpoint does not use 4 qubits.")

if checkpoint.get("n_layers", 1) != 1:
    raise ValueError("Circuit layer count does not match this script.")

# The image model returns hybrid logits, not EfficientNet-only logits.
class ImageHybridModel(nn.Module):
    def __init__(self, cnn, hybrid_head):
        super().__init__()
        self.cnn = cnn
        self.head = hybrid_head

    def forward(self, images):
        feature_maps = self.cnn.backbone.features(images)
        pooled = self.cnn.backbone.avgpool(feature_maps)
        features = torch.flatten(pooled, 1)
        return self.head(features)


model = ImageHybridModel(backbone, head).to(DEVICE).eval()

for parameter in model.parameters():
    parameter.requires_grad_(True)

print("Both checkpoints loaded successfully.")


# =========================================================
# DATASET AND CLASS MAPPING
# =========================================================
dataset = ImageFolder(str(IMAGE_ROOT), transform=TRANSFORM)

if dataset.classes != ["benign", "malignant"]:
    raise ValueError(
        f"Expected ['benign', 'malignant']; got {dataset.classes}"
    )

print("Classes:", dataset.classes)
print("Test images available:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError("No images found in the test directory.")


def load_original(index):
    path, label = dataset.samples[index]
    image = Image.open(path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return path, label, image, tensor


def save_overlay(image, heatmap, output_path, title):
    original = np.asarray(image.resize((224, 224))).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))

    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.55, colored, 0.45, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# GRAD-CAM AND GRAD-CAM++
# =========================================================
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.cnn.backbone.features[-1]

def generate_cam(cam_class, tensor, target_class, output_path, image, title):
    cam = cam_class(
        model=model,
        target_layers=[target_layer],
    )

    grayscale = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]

    save_overlay(image, grayscale, output_path, title)
    del cam


# =========================================================
# PREDICTION FUNCTION FOR SHAP AND LIME
# =========================================================
@torch.no_grad()
def predict_probabilities(image_batch):
    model.eval()
    tensors = []

    for array in image_batch:
        pil_image = Image.fromarray(
            np.asarray(array).astype(np.uint8)
        ).convert("RGB")
        tensors.append(TRANSFORM(pil_image))

    batch = torch.stack(tensors).to(DEVICE)
    logits = model(batch)

    return torch.softmax(logits, dim=1).cpu().numpy()


# =========================================================
# GROUPED SHAP ON THE HYBRID HEAD
# =========================================================
def generate_grouped_shap(feature_vector, output_path):
    import shap

    model.head.eval()
    feature_vector = feature_vector.detach().float().to(DEVICE)

    # Partition 1536 dimensions into 32 contiguous groups.
    chunks = torch.tensor_split(
        torch.arange(1536, device=DEVICE), SHAP_GROUPS
    )

    group_values = torch.stack([
        feature_vector[0, chunk].mean() for chunk in chunks
    ]).detach()

    # Each group represents a shift relative to this image's features.
    def expand_groups(rows):
        rows = torch.as_tensor(
            rows, dtype=torch.float32, device=DEVICE
        )

        expanded = feature_vector.repeat(rows.shape[0], 1)

        for group_id, chunk in enumerate(chunks):
            delta = rows[:, group_id] - group_values[group_id]
            expanded[:, chunk] += delta[:, None]

        return expanded

    def grouped_predict(rows):
        expanded = expand_groups(rows)

        with torch.no_grad():
            logits = model.head(expanded)
            return torch.softmax(logits, dim=1).cpu().numpy()

    actual = group_values.cpu().numpy().reshape(1, -1)
    baseline = np.zeros_like(actual)

    explainer = shap.KernelExplainer(grouped_predict, baseline)
    shap_values = explainer.shap_values(
        actual,
        nsamples=SHAP_SAMPLES,
    )

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1]).reshape(-1)
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, 1]
        else:
            values = values.reshape(-1)

    order = np.argsort(np.abs(values))[::-1]
    top = order[:15]

    plt.figure(figsize=(9, 6))
    plt.barh(
        [f"Feature group {i + 1}" for i in top[::-1]],
        values[top[::-1]],
    )
    plt.xlabel("SHAP value (malignant output)")
    plt.title("Grouped SHAP — Hybrid Classifier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "method": "Grouped Kernel SHAP",
        "group_count": SHAP_GROUPS,
        "samples": SHAP_SAMPLES,
        "top_group_indices_zero_based": top.tolist(),
        "top_group_shap_values": values[top].tolist(),
        "note": (
            "Grouped local explanation of the hybrid head; "
            "not individual-feature SHAP or a pixel attribution map."
        ),
    }


# =========================================================
# LIME IMAGE EXPLANATION
# =========================================================
def generate_lime(image, output_path, target_class):
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import mark_boundaries

    explainer = LimeImageExplainer(random_state=42)

    explanation = explainer.explain_instance(
        np.asarray(image.resize((224, 224))).astype(np.uint8),
        classifier_fn=predict_probabilities,
        labels=(target_class,),
        top_labels=None,
        hide_color=0,
        num_samples=LIME_SAMPLES,
    )

    image_explanation, mask = explanation.get_image_and_mask(
        label=target_class,
        positive_only=False,
        num_features=8,
        hide_rest=False,
    )

    boundary_image = mark_boundaries(
        image_explanation / 255.0,
        mask,
        color=(1, 1, 0),
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(boundary_image)
    plt.axis("off")
    plt.title(f"LIME regions — {dataset.classes[target_class]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# RUN EXPLANATIONS ON A SMALL TEST SUBSET
# =========================================================
results = []

for index in range(min(MAX_IMAGES, len(dataset))):
    path, actual_label, image, tensor = load_original(index)
    image_id = Path(path).stem

    print(f"\n[{index + 1}] Explaining: {image_id}")

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(probabilities.argmax().item())

    print(
        "Actual:", dataset.classes[actual_label],
        "| Predicted:", dataset.classes[predicted_class],
        "| Probabilities:", probabilities.cpu().tolist(),
    )

    # Both CAM methods explain the predicted class.
    generate_cam(
        GradCAM, tensor, predicted_class,
        OUT / "gradcam" / f"{image_id}_gradcam.png",
        image, f"Grad-CAM: {dataset.classes[predicted_class]}",
    )

    generate_cam(
        GradCAMPlusPlus, tensor, predicted_class,
        OUT / "gradcam_plus_plus" / f"{image_id}_gradcam_pp.png",
        image, f"Grad-CAM++: {dataset.classes[predicted_class]}",
    )

    # Extract features using the same loaded EfficientNet checkpoint.
    with torch.no_grad():
        feature_maps = model.cnn.backbone.features(tensor)
        pooled = model.cnn.backbone.avgpool(feature_maps)
        feature_vector = torch.flatten(pooled, 1)

    try:
        shap_report = generate_grouped_shap(
            feature_vector,
            OUT / "shap" / f"{image_id}_grouped_shap.png",
        )
    except Exception as exc:
        shap_report = {"error": repr(exc)}
        print("SHAP failed:", repr(exc))

    try:
        generate_lime(
            image,
            OUT / "lime" / f"{image_id}_lime.png",
            predicted_class,
        )
        lime_status = "success"
    except Exception as exc:
        lime_status = repr(exc)
        print("LIME failed:", repr(exc))

    results.append({
        "image": str(path),
        "actual_class": dataset.classes[actual_label],
        "predicted_class": dataset.classes[predicted_class],
        "probabilities": probabilities.cpu().tolist(),
        "gradcam": "success",
        "gradcam_plus_plus": "success",
        "shap": shap_report,
        "lime": lime_status,
    })

    print("Saved XAI outputs for:", image_id)

report_path = OUT / "xai_run_report.json"
with open(report_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nXAI run completed.")
print("Report:", report_path)
print("Output folder:", OUT)
