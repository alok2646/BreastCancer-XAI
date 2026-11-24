
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "quantum_efficientnet_b3" / "patient_split_features"
RUN_DIR = ROOT / "results" / "quantum_efficientnet_b3" / "patient_pilot_run"

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64
EPOCHS = 3
N_QUBITS = 4

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise RuntimeError(f"Output directory already exists: {RUN_DIR}")

RUN_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

def load_split(name):
    data = torch.load(
        FEATURE_DIR / f"{name}_features.pt",
        map_location="cpu",
        weights_only=True,
    )
    return data["features"].float(), data["labels"].long(), data["classes"]

X_train, y_train, classes = load_split("train")
X_val, y_val, val_classes = load_split("val")

assert classes == ["benign", "malignant"]
assert val_classes == classes

# Pilot: use a fixed subset for quick runtime testing.
# Sampling separately within each class preserves both classes.
g = torch.Generator().manual_seed(42)
indices = []

for label in (0, 1):
    candidates = torch.where(y_train == label)[0]
    chosen = candidates[torch.randperm(len(candidates), generator=g)[:2000]]
    indices.append(chosen)

pilot_indices = torch.cat(indices)
X_train = X_train[pilot_indices]
y_train = y_train[pilot_indices]

print("GPU:", torch.cuda.get_device_name(0))
print("Pilot training samples:", len(y_train))
print("Pilot class counts:", torch.bincount(y_train).tolist())
print("Validation samples:", len(y_val))

# Normalize with pilot-training data only.
mean = X_train.mean(dim=0)
std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

def rotation(theta, axis):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)

    if axis == "Y":
        real = torch.stack([c, -s, s, c], dim=-1).reshape(*theta.shape, 2, 2)
        return torch.complex(real, torch.zeros_like(real))

    real = torch.stack([c, z, z, c], dim=-1).reshape(*theta.shape, 2, 2)
    if axis == "X":
        imag = torch.stack([z, -s, -s, z], dim=-1).reshape(*theta.shape, 2, 2)
    else:
        imag = torch.stack([-s, z, z, s], dim=-1).reshape(*theta.shape, 2, 2)

    return torch.complex(real, imag)

def apply_gate(state, matrix, wire):
    batch = state.shape[0]
    shaped = state.reshape(batch, 2, 2, 2, 2)
    moved = shaped.movedim(wire + 1, -1)
    pairs = moved.reshape(batch, -1, 2)
    result = torch.matmul(pairs, matrix.transpose(-1, -2))
    return result.reshape(moved.shape).movedim(-1, wire + 1).reshape(batch, 16)

def cnot(state, control, target):
    indices = torch.arange(16, device=state.device)
    control_bit = 1 << (3 - control)
    target_bit = 1 << (3 - target)
    permutation = indices.clone()
    mask = (indices & control_bit) != 0
    permutation[mask] ^= target_bit
    return state.index_select(1, permutation)

class QuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.05 * torch.randn(1, 4, 3))

    def forward(self, angles):
        batch = angles.shape[0]
        zero = torch.zeros(batch, dtype=torch.long, device=angles.device)
        state = F.one_hot(zero, 16).to(torch.complex64)

        for wire in range(4):
            state = apply_gate(state, rotation(angles[:, wire], "Y"), wire)

        for wire in range(4):
            for k, axis in enumerate(("X", "Y", "Z")):
                state = apply_gate(state, rotation(self.weights[0, wire, k], axis), wire)

        for wire in range(4):
            state = cnot(state, wire, (wire + 1) % 4)

        probs = state.abs().square()
        basis = torch.arange(16, device=angles.device)
        outputs = []

        for wire in range(4):
            bits = (basis >> (3 - wire)) & 1
            outputs.append(probs @ (1 - 2 * bits).to(probs.dtype))

        return torch.stack(outputs, dim=1)

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
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
        angles = torch.pi * torch.tanh(self.projection(x))
        return self.classifier(self.quantum(angles))

model = HybridModel().to(DEVICE)

# Class weights are calculated from the pilot training subset.
counts = torch.bincount(y_train, minlength=2).float()
weights = (counts.sum() / (2 * counts)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-3
)

best_val_loss = float("inf")
history = []

def evaluate():
    model.eval()
    total_loss, total, correct = 0.0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()

    return total_loss / total, correct / total

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total, correct = 0.0, 0, 0

    for step, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total += len(y)
        correct += (logits.argmax(1) == y).sum().item()

        if step % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, batch {step}/{len(train_loader)}")

    train_loss = total_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate()

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    history.append(record)
    print(record)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            RUN_DIR / "best_pilot_model.pth",
        )

with open(RUN_DIR / "pilot_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("Pilot complete. No test-set evaluation was performed.")
print("Results saved to:", RUN_DIR)
