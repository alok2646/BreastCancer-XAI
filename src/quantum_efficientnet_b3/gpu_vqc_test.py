
import torch
import torch.nn as nn
import math

assert torch.cuda.is_available(), "CUDA GPU not available"

DEVICE = torch.device("cuda")
N_QUBITS = 4
N_LAYERS = 1

def apply_gate(state, matrix, wire):
    batch = state.shape[0]
    shaped = state.reshape(batch, *([2] * N_QUBITS))
    moved = shaped.movedim(wire + 1, -1)
    pairs = moved.reshape(batch, -1, 2)

    result = torch.matmul(pairs, matrix.transpose(-1, -2))
    restored = result.reshape(moved.shape).movedim(-1, wire + 1)
    return restored.reshape(batch, 2**N_QUBITS)

def rotation_matrix(theta, axis):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)

    if axis == "Y":
        rows = [[c, -s], [s, c]]
        real = torch.stack(
            [torch.stack(r, dim=-1) for r in rows], dim=-2
        )
        return torch.complex(real, torch.zeros_like(real)).to(torch.complex64)

    if axis == "X":
        real = torch.stack([
            torch.stack([c, z], dim=-1),
            torch.stack([z, c], dim=-1)
        ], dim=-2)
        imag = torch.stack([
            torch.stack([z, -s], dim=-1),
            torch.stack([-s, z], dim=-1)
        ], dim=-2)
        return torch.complex(real, imag).to(torch.complex64)

    if axis == "Z":
        real = torch.stack([
            torch.stack([c, z], dim=-1),
            torch.stack([z, c], dim=-1)
        ], dim=-2)
        imag = torch.stack([
            torch.stack([-s, z], dim=-1),
            torch.stack([z, s], dim=-1)
        ], dim=-2)
        return torch.complex(real, imag).to(torch.complex64)

    raise ValueError("Unknown rotation axis")

def apply_cnot(state, control, target):
    indices = torch.arange(16, device=state.device)
    control_bit = 1 << (N_QUBITS - 1 - control)
    target_bit = 1 << (N_QUBITS - 1 - target)

    perm = indices.clone()
    mask = (indices & control_bit) != 0
    perm[mask] = indices[mask] ^ target_bit

    return state.index_select(1, perm)

class GPUQuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            0.05 * torch.randn(N_LAYERS, N_QUBITS, 3)
        )

    def forward(self, angles):
        batch = angles.shape[0]
        zero = torch.zeros(batch, device=DEVICE, dtype=torch.long)
        state = torch.nn.functional.one_hot(zero, 16).to(torch.complex64)

        # Encode four features as RY rotations.
        for wire in range(N_QUBITS):
            state = apply_gate(
                state, rotation_matrix(angles[:, wire], "Y"), wire
            )

        # Trainable rotations and entanglement.
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

        probs = state.abs().square()
        basis = torch.arange(16, device=DEVICE)
        measurements = []

        for wire in range(N_QUBITS):
            bit = (basis >> (N_QUBITS - 1 - wire)) & 1
            signs = (1 - 2 * bit).to(probs.dtype)
            measurements.append(probs @ signs)

        return torch.stack(measurements, dim=1)

model = GPUQuantumCircuit().to(DEVICE)
angles = torch.randn(2, 4, device=DEVICE, requires_grad=True)

output = model(angles)
loss = output.square().mean()
loss.backward()

print("GPU:", torch.cuda.get_device_name(0))
print("Input device:", angles.device)
print("Quantum state simulation device:", output.device)
print("Output shape:", tuple(output.shape))
print("Output:\n", output.detach())
print("Input gradient:", angles.grad is not None)
print("Trainable quantum parameters:", model.weights.numel())
print("Parameter gradient:", model.weights.grad is not None)

assert output.shape == (2, 4)
assert angles.grad is not None
assert model.weights.grad is not None
assert torch.isfinite(output).all()

print("SUCCESS: GPU quantum circuit forward/backward test passed.")
