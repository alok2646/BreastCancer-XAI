
from pathlib import Path
import torch

BASE = Path("results/quantum_efficientnet_b3/features")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device available:", DEVICE)

for split in ("train", "val", "test"):
    path = BASE / f"{split}_features.pt"

    if not path.exists():
        print(f"❌ Missing: {path}")
        continue

    data = torch.load(path, map_location="cpu", weights_only=True)

    print(f"\n--- {split.upper()} ---")
    print("Data type:", type(data))

    if isinstance(data, torch.Tensor):
        print("Feature shape:", tuple(data.shape))
        print("Dtype:", data.dtype)
        print("Finite values:", bool(torch.isfinite(data).all()))

    elif isinstance(data, dict):
        print("Keys:", list(data.keys()))

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"{key}: shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}"
                )
            else:
                print(f"{key}: type={type(value).__name__}")

    else:
        print("Unexpected format; inspect before training.")
