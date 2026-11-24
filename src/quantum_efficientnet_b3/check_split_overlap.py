
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
base = root / "results" / "quantum_efficientnet_b3" / "features"

splits = {}

for name in ("train", "val", "test"):
    data = torch.load(
        base / f"{name}_features.pt",
        map_location="cpu",
        weights_only=True
    )

    paths = [str(p) for p in data["paths"]]
    normalized = [p.replace("\\", "/").casefold() for p in paths]

    splits[name] = set(normalized)

    print(f"{name}: {len(paths)} paths, {len(set(normalized))} unique")

print("\n--- EXACT PATH OVERLAP ---")

pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]

for a, b in pairs:
    overlap = splits[a] & splits[b]
    print(f"{a} vs {b}: {len(overlap)} overlapping paths")

print("\nNote: Zero path overlap does not prove patient-level separation.")
