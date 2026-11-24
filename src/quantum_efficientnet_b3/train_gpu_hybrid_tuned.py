
from pathlib import Path
import csv
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "quantum_efficientnet_b3" / "features"
RUN_DIR = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "gpu_hybrid_tuned_run"
)
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 128
EPOCHS = 8
PATIENCE = 3
LR = 3e-4
N_QUBITS = 4
N_LAYERS = 1

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Training stopped.")

if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise RuntimeError(
        f"{RUN_DIR} already contains files. "
        "Choose a new RUN_DIR to avoid overwriting results."
    )

RUN_DIR.mkdir(parents=True, exist_ok=True)

print("GPU:", torch.cuda.get_device_name(0))
print("Output directory:", RUN_DIR)

# --------------------------------------------------
# Load previously extracted features and labels
# --------------------------------------------------
def load_split(name):
    path = FEATURE_DIR / f"{name}_features.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected format: {path}")

    x = data["features"].float()
    y = data["labels"].long()

    if x.ndim != 2 or x.shape[1] != 1536:
        raise ValueError(f"Unexpected feature shape: {x.shape}")

    if len(x) != len(y):
        raise ValueError(f"Feature/label count mismatch: {name}")

    if not torch.isfinite(x).all():
        raise ValueError(f"Non-finite features in {name}")

    return x, y, data["classes"]

X_train, y_train, classes = load_split("train")
X_val, y_val, val_classes = load_split("val")
X_test, y_test, test_classes = load_split("test")

if classes != val_classes or classes != test_classes:
    raise ValueError("Class order differs across splits.")

if classes != ["benign", "malignant"]:
    raise ValueError(f"Unexpected class mapping: {classes}")

if set(y_train.tolist()) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

print("Classes:", classes)
print("Train:", tuple(X_train.shape), torch.bincount(y_train).tolist())
print("Val:", tuple(X_val.shape), torch.bincount(y_val).tolist())
print("Test:", tuple(X_test.shape), torch.bincount(y_test).tolist())

# Train-only normalization statistics; no validation/test leakage.
feature_mean = X_train.mean(dim=0)
feature_std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# --------------------------------------------------
# Differentiable quantum state-vector operations
# --------------------------------------------------
def rotation_matrix(theta, axis):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)

    if axis == "Y":
        real = torch.stack([c, -s, s, c], dim=-1)
        return torch.complex(real.reshape(*theta.shape, 2, 2),
                             torch.zeros_like(
                                 real.reshape(*theta.shape, 2, 2)
                             ))

    if axis == "X":
        real = torch.stack([c, z, z, c], dim=-1)
        imag = torch.stack([z, -s, -s, z], dim=-1)
    elif axis == "Z":
        real = torch.stack([c, z, z, c], dim=-1)
        imag = torch.stack([-s, z, z, s], dim=-1)
    else:
        raise ValueError(axis)

    shape = (*theta.shape, 2, 2)
    return torch.complex(real.reshape(shape), imag.reshape(shape))


def apply_gate(state, matrix, wire):
    batch = state.shape[0]
    shaped = state.reshape(batch, *([2] * N_QUBITS))
    moved = shaped.movedim(wire + 1, -1)
    pairs = moved.reshape(batch, -1, 2)

    result = torch.matmul(pairs, matrix.transpose(-1, -2))
    restored = result.reshape(moved.shape).movedim(-1, wire + 1)

    return restored.reshape(batch, 2 ** N_QUBITS)


def apply_cnot(state, control, target):
    indices = torch.arange(2 ** N_QUBITS, device=state.device)
    control_bit = 1 << (N_QUBITS - 1 - control)
    target_bit = 1 << (N_QUBITS - 1 - target)

    permutation = indices.clone()
    mask = (indices & control_bit) != 0
    permutation[mask] = indices[mask] ^ target_bit

    return state.index_select(1, permutation)


class GPUQuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            0.05 * torch.randn(N_LAYERS, N_QUBITS, 3)
        )

    def forward(self, angles):
        batch = angles.shape[0]
        initial = torch.zeros(batch, device=angles.device, dtype=torch.long)
        state = F.one_hot(initial, 2 ** N_QUBITS).to(torch.complex64)

        # Encode classical features into the quantum state.
        for wire in range(N_QUBITS):
            state = apply_gate(
                state, rotation_matrix(angles[:, wire], "Y"), wire
            )

        # Trainable rotations and an entangling CNOT ring.
        for layer in range(N_LAYERS):
            for wire in range(N_QUBITS):
                for k, axis in enumerate(("X", "Y", "Z")):
                    theta = self.weights[layer, wire, k]
                    state = apply_gate(
                        state, rotation_matrix(theta, axis), wire
                    )

            for wire in range(N_QUBITS):
                state = apply_cnot(
                    state, wire, (wire + 1) % N_QUBITS
                )

        probabilities = state.abs().square()
        basis = torch.arange(2 ** N_QUBITS, device=angles.device)
        measurements = []

        for wire in range(N_QUBITS):
            bit = (basis >> (N_QUBITS - 1 - wire)) & 1
            signs = (1 - 2 * bit).to(probabilities.dtype)
            measurements.append(probabilities @ signs)

        return torch.stack(measurements, dim=1)


# --------------------------------------------------
# Hybrid classifier
# --------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())

        self.projection = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, N_QUBITS),
        )

        self.quantum = GPUQuantumCircuit()
        self.classifier = nn.Linear(N_QUBITS, 2)

    def forward(self, x):
        x = (x - self.feature_mean) / self.feature_std
        angles = math.pi * torch.tanh(self.projection(x))
        measurements = self.quantum(angles)
        return self.classifier(measurements)


model = HybridModel(feature_mean, feature_std).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-3
)
criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Training and validation
# --------------------------------------------------
best_val_loss = float("inf")
best_epoch = 0
bad_epochs = 0
history = []
BEST_PATH = RUN_DIR / "best_hybrid_model.pth"


def evaluate_loss(loader):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

    return total_loss / total


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        if step % 25 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {step}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = total_loss / total
    train_acc = correct / total
    val_loss = evaluate_loss(val_loader)

    print(
        f"\nEpoch {epoch}: train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
    })

    with open(RUN_DIR / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        bad_epochs = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "feature_dim": 1536,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
        }, BEST_PATH)

        print("Saved new best validation checkpoint.")
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.")
        break

# --------------------------------------------------
# Final test evaluation: run once, after model selection
# --------------------------------------------------
checkpoint = torch.load(
    BEST_PATH, map_location=DEVICE, weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        all_labels.extend(y.tolist())
        all_preds.extend(probs.argmax(1).cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

tp = sum(p == 1 and y == 1 for p, y in zip(all_preds, all_labels))
tn = sum(p == 0 and y == 0 for p, y in zip(all_preds, all_labels))
fp = sum(p == 1 and y == 0 for p, y in zip(all_preds, all_labels))
fn = sum(p == 0 and y == 1 for p, y in zip(all_preds, all_labels))

def ratio(a, b):
    return a / b if b else 0.0

accuracy = ratio(tp + tn, len(all_labels))
precision = ratio(tp, tp + fp)
recall = ratio(tp, tp + fn)
specificity = ratio(tn, tn + fp)
f1 = ratio(2 * precision * recall, precision + recall)

# ROC-AUC from average ranks; ties receive average ranks.
ranked = sorted(
    [(score, label) for score, label in zip(all_probs, all_labels)],
    key=lambda item: item[0]
)
positive_count = sum(all_labels)
negative_count = len(all_labels) - positive_count
rank_sum_positive = 0
i = 0

while i < len(ranked):
    j = i + 1
    while j < len(ranked) and ranked[j][0] == ranked[i][0]:
        j += 1

    average_rank = ((i + 1) + j) / 2
    rank_sum_positive += average_rank * sum(
        ranked[k][1] for k in range(i, j)
    )
    i = j

auc = (
    (rank_sum_positive - positive_count * (positive_count + 1) / 2)
    / (positive_count * negative_count)
    if positive_count and negative_count else float("nan")
)

metrics = {
    "classes": classes,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "test_count": len(all_labels),
    "accuracy": accuracy,
    "precision_malignant": precision,
    "recall_malignant": recall,
    "f1_malignant": f1,
    "specificity_benign": specificity,
    "roc_auc_malignant": auc,
    "confusion_matrix": {
        "true_benign_pred_benign": tn,
        "true_benign_pred_malignant": fp,
        "true_malignant_pred_benign": fn,
        "true_malignant_pred_malignant": tp,
    },
}

with open(RUN_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n--- FINAL TEST RESULTS ---")
print(json.dumps(metrics, indent=2))
print("\nBest model:", BEST_PATH)
print("Metrics:", RUN_DIR / "test_metrics.json")
print("Training history:", RUN_DIR / "training_history.csv")
