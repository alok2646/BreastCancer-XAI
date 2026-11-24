
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "dataset_base_final"
CHECKPOINT = ROOT / "saved_models" / "efficientnet_b3_best_model.pth"
OUT_DIR = ROOT / "results" / "quantum_efficientnet_b3" / "features"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same EfficientNet-B3 architecture without downloading weights.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop without CPU fallback.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = FeatureExtractor().to(DEVICE)
    state = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(
            DATA_DIR / split, transform=transform
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        features, labels, paths = [], [], []

        with torch.inference_mode():
            for images, targets in loader:
                images = images.to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    output = model(images)

                features.append(output.float().cpu())
                labels.append(targets.cpu())

        paths = [str(path) for path, _ in dataset.samples]

        save_path = OUT_DIR / f"{split}_features.pt"
        torch.save({
            "features": torch.cat(features),
            "labels": torch.cat(labels),
            "paths": paths,
            "classes": dataset.classes,
            "checkpoint": str(CHECKPOINT),
        }, save_path)

        print(
            f"{split}: {len(dataset)} images | "
            f"features={tuple(torch.cat(features).shape)} | "
            f"saved={save_path}"
        )

    print("GPU feature extraction completed.")

if __name__ == "__main__":
    main()
