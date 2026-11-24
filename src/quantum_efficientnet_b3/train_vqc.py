
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "quantum_efficientnet_b3" / "features"
OUT_DIR = ROOT / "results" / "quantum_efficientnet_b3"
MODEL_DIR = ROOT / "saved_models" / "quantum_efficientnet_b3"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
N_QUBITS = 4
N_LAYERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; stopping instead of silently using CPU.")

# Deterministic, balanced subsets for a quick pilot.
def load_balanced_split(name, per_class):
    obj = torch.load(
        FEATURE_DIR / f"{name}_features.pt",
        map_location="cpu",
        weights_only=True
    )
    x, y = obj["features"].float(), obj["labels"].long()
    rng = np.random.default_rng(SEED)
    chosen = []

    for cls in (0, 1):
        idx = torch.where(y == cls)[0].numpy()
        if len(idx) < per_class:
            raise ValueError(f"Not enough examples for class {cls} in {name}")
        chosen.extend(rng.choice(idx, per_class, replace=False).tolist())

    chosen = np.array(chosen)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]

x_train, y_train = load_balanced_split("train", 500)
x_val, y_val = load_balanced_split("val", 150)
x_test, y_test = load_balanced_split("test", 150)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=BATCH_SIZE, shuffle=False
)

# Keep the quantum simulator and its trainable circuit on CPU.
qdev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(qdev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

quantum_layer = qml.qnn.TorchLayer(
    circuit, {"weights": (N_LAYERS, N_QUBITS, 3)}
).cpu()

class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1536, 64),
            nn.GELU(),
            nn.Linear(64, N_QUBITS),
            nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )

    def forward(self, features):
        angles = self.encoder(features) * np.pi
        # Transfer to CPU for the default.qubit simulator.
        measured = quantum_layer(angles.to("cpu"))
        return self.head(measured.to(features.device))

model = QuantumHybrid().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(quantum_layer.parameters()),
    lr=1e-3, weight_decay=1e-4
)

# Smoke test before training.
try:
    xb, yb = next(iter(train_loader))
    logits = model(xb.to(DEVICE))
    loss = criterion(logits, yb.to(DEVICE))
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print("Smoke test passed:", tuple(logits.shape))
except Exception as exc:
    raise RuntimeError(
        "Quantum forward/backward smoke test failed; training stopped."
    ) from exc

def evaluate(loader):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_y.extend(yb.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_prob.extend(probs.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_y, all_pred),
        "precision": precision_score(all_y, all_pred, zero_division=0),
        "recall": recall_score(all_y, all_pred, zero_division=0),
        "f1": f1_score(all_y, all_pred, zero_division=0),
        "roc_auc": roc_auc_score(all_y, all_prob),
        "confusion_matrix": confusion_matrix(all_y, all_pred).tolist()
    }

history = []
best_val = float("inf")
best_path = MODEL_DIR / "vqc_pilot_best.pt"
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    quantum_layer.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)

    val = evaluate(val_loader)
    row = {
        "epoch": epoch,
        "train_loss": running_loss / len(train_loader.dataset),
        **{f"val_{k}": v for k, v in val.items()
           if k != "confusion_matrix"}
    }
    history.append(row)
    print(json.dumps(row))

    if val["loss"] < best_val:
        best_val = val["loss"]
        torch.save({
            "model_state": model.state_dict(),
            "quantum_state": quantum_layer.state_dict(),
            "seed": SEED,
            "epoch": epoch
        }, best_path)

pd.DataFrame(history).to_csv(
    OUT_DIR / "vqc_pilot_training_history.csv", index=False
)

# Reload the selected validation checkpoint before test evaluation.
checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
quantum_layer.load_state_dict(checkpoint["quantum_state"])

test_metrics = evaluate(test_loader)
test_metrics["test_samples"] = len(y_test)
test_metrics["train_samples"] = len(y_train)
test_metrics["validation_samples"] = len(y_val)
test_metrics["elapsed_seconds"] = time.time() - start
test_metrics["device_for_cnn"] = str(DEVICE)
test_metrics["quantum_simulator"] = "PennyLane default.qubit (CPU)"

with open(OUT_DIR / "vqc_pilot_test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

print("\nFINAL PILOT TEST METRICS")
print(json.dumps(test_metrics, indent=2))
print("Saved checkpoint:", best_path)
print("Elapsed seconds:", round(time.time() - start, 1))
