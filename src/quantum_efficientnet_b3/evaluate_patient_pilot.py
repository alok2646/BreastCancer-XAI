
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 1. PATHS AND DEVICE
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "patient_split_features"
)

RUN_DIR = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "patient_pilot_run"
)

CHECKPOINT_PATH = RUN_DIR / "best_pilot_model.pth"
OUTPUT_PATH = RUN_DIR / "pilot_validation_metrics.json"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")


# =========================================================
# 2. LOAD TRAINING AND VALIDATION FEATURES
# =========================================================

train_data = torch.load(
    FEATURE_DIR / "train_features.pt",
    map_location="cpu",
    weights_only=True,
)

val_data = torch.load(
    FEATURE_DIR / "val_features.pt",
    map_location="cpu",
    weights_only=True,
)

classes = val_data["classes"]

if classes != ["benign", "malignant"]:
    raise ValueError(f"Unexpected class order: {classes}")

# Recreate the same 4,000-sample training subset used in the pilot.
generator = torch.Generator().manual_seed(42)
selected = []

for label in (0, 1):
    candidates = torch.where(
        train_data["labels"] == label
    )[0]

    if len(candidates) < 2000:
        raise ValueError(
            f"Insufficient training samples for class {label}"
        )

    chosen = candidates[
        torch.randperm(len(candidates), generator=generator)[:2000]
    ]

    selected.append(chosen)

indices = torch.cat(selected)

X_pilot = train_data["features"][indices].float()

# Same normalization statistics as the pilot training script.
mean = X_pilot.mean(dim=0)
std = X_pilot.std(dim=0, unbiased=False).clamp_min(1e-6)

X_val = val_data["features"].float()
y_val = val_data["labels"].long()

print("Validation features:", tuple(X_val.shape))
print("Validation labels:", tuple(y_val.shape))
print("Validation class counts:", torch.bincount(y_val).tolist())


# =========================================================
# 3. QUANTUM CIRCUIT
# =========================================================

def rotation(theta, axis):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)

    if axis == "Y":
        real = torch.stack(
            [c, -s, s, c], dim=-1
        ).reshape(*theta.shape, 2, 2)

        return torch.complex(real, torch.zeros_like(real))

    real = torch.stack(
        [c, z, z, c], dim=-1
    ).reshape(*theta.shape, 2, 2)

    if axis == "X":
        imag = torch.stack(
            [z, -s, -s, z], dim=-1
        ).reshape(*theta.shape, 2, 2)
    else:
        imag = torch.stack(
            [-s, z, z, s], dim=-1
        ).reshape(*theta.shape, 2, 2)

    return torch.complex(real, imag)


def apply_gate(state, matrix, wire):
    batch = state.shape[0]

    shaped = state.reshape(batch, 2, 2, 2, 2)
    moved = shaped.movedim(wire + 1, -1)
    pairs = moved.reshape(batch, -1, 2)

    result = torch.matmul(
        pairs,
        matrix.transpose(-1, -2),
    )

    return (
        result.reshape(moved.shape)
        .movedim(-1, wire + 1)
        .reshape(batch, 16)
    )


def cnot(state, control, target):
    indices = torch.arange(16, device=state.device)

    control_bit = 1 << (3 - control)
    target_bit = 1 << (3 - target)

    permutation = indices.clone()
    mask = (indices & control_bit) != 0
    permutation[mask] = indices[mask] ^ target_bit

    return state.index_select(1, permutation)


class QuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(
            0.05 * torch.randn(1, 4, 3)
        )

    def forward(self, angles):
        batch = angles.shape[0]

        zero = torch.zeros(
            batch,
            dtype=torch.long,
            device=angles.device,
        )

        state = F.one_hot(zero, 16).to(torch.complex64)

        for wire in range(4):
            state = apply_gate(
                state,
                rotation(angles[:, wire], "Y"),
                wire,
            )

        for wire in range(4):
            for k, axis in enumerate(("X", "Y", "Z")):
                state = apply_gate(
                    state,
                    rotation(self.weights[0, wire, k], axis),
                    wire,
                )

        for wire in range(4):
            state = cnot(state, wire, (wire + 1) % 4)

        probs = state.abs().square()
        basis = torch.arange(16, device=state.device)

        outputs = []

        for wire in range(4):
            bits = (basis >> (3 - wire)) & 1
            signs = (1 - 2 * bits).to(probs.dtype)
            outputs.append(probs @ signs)

        return torch.stack(outputs, dim=1)


# =========================================================
# 4. HYBRID MODEL
# =========================================================

class HybridModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

        self.projection = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 4),
        )

        self.quantum = QuantumCircuit()
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        x = (x - self.mean) / self.std

        angles = torch.pi * torch.tanh(
            self.projection(x)
        )

        quantum_output = self.quantum(angles)

        return self.classifier(quantum_output)


# =========================================================
# 5. LOAD SAVED CHECKPOINT — NO TRAINING
# =========================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model = HybridModel(mean, std).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded pilot checkpoint from epoch:", checkpoint["epoch"])


# =========================================================
# 6. VALIDATION INFERENCE
# =========================================================

loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

all_y = []
all_probs = []

with torch.inference_mode():
    for x, y in loader:
        logits = model(x.to(DEVICE))

        malignant_probs = torch.softmax(
            logits, dim=1
        )[:, 1]

        all_probs.extend(
            malignant_probs.cpu().tolist()
        )

        all_y.extend(y.tolist())

print("Validation predictions completed:", len(all_y))


# =========================================================
# 7. METRIC CALCULATION
# =========================================================

def calculate_metrics(actual, predicted):
    # Rows = actual class; columns = predicted class.
    cm = [[0, 0], [0, 0]]

    for a, p in zip(actual, predicted):
        cm[a][p] += 1

    per_class = {}

    for label, name in enumerate(classes):
        tp = cm[label][label]

        fp = sum(
            cm[row][label]
            for row in range(2)
            if row != label
        )

        fn = sum(
            cm[label][col]
            for col in range(2)
            if col != label
        )

        tn = (
            len(actual) - tp - fp - fn
        )

        precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if tn + fp else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": sum(cm[label]),
        }

    accuracy = sum(
        cm[i][i] for i in range(2)
    ) / len(actual)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "class_metrics": per_class,
    }


# =========================================================
# 8. DEFAULT THRESHOLD = 0.50
# =========================================================

default_predictions = [
    int(prob >= 0.50)
    for prob in all_probs
]

default_results = calculate_metrics(
    all_y,
    default_predictions,
)


# =========================================================
# 9. VALIDATION THRESHOLD ANALYSIS
# =========================================================

threshold_results = []

for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    predictions = [
        int(prob >= threshold)
        for prob in all_probs
    ]

    result = calculate_metrics(
        all_y,
        predictions,
    )

    malignant = result["class_metrics"]["malignant"]

    threshold_results.append({
        "threshold": threshold,
        "accuracy": result["accuracy"],
        "malignant_precision": malignant["precision"],
        "malignant_recall": malignant["recall"],
        "malignant_specificity": malignant["specificity"],
        "malignant_f1": malignant["f1"],
    })


# =========================================================
# 10. SAVE RESULTS TO A NEW JSON FILE
# =========================================================

report = {
    "experiment": "patient-split pilot validation",
    "checkpoint_epoch": checkpoint["epoch"],
    "classes": classes,
    "validation_samples": len(all_y),
    "default_threshold": 0.50,
    "default_threshold_results": default_results,
    "threshold_analysis": threshold_results,
    "test_set_used": False,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n--- VALIDATION RESULTS AT THRESHOLD 0.50 ---")
print(json.dumps(default_results, indent=2))

print("\n--- THRESHOLD COMPARISON ---")

for item in threshold_results:
    print(
        f"Threshold={item['threshold']:.2f} | "
        f"Accuracy={item['accuracy']:.4f} | "
        f"Malignant Precision={item['malignant_precision']:.4f} | "
        f"Malignant Recall={item['malignant_recall']:.4f} | "
        f"Malignant F1={item['malignant_f1']:.4f}"
    )

print("\nSaved:", OUTPUT_PATH)
print("Validation only. No training performed.")
print("Test set was not used.")
