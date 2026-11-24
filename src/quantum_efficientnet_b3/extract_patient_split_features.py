
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dataset_patient_split"
OUT_ROOT = (
    ROOT / "results" / "quantum_efficientnet_b3"
    / "patient_split_features"
)

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 16
IMAGE_SIZE = 224

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Stopping.")

if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
    raise RuntimeError(
        f"Output folder already has files: {OUT_ROOT}\n"
        "Choose a fresh output folder to avoid overwriting."
    )

OUT_ROOT.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print("GPU:", torch.cuda.get_device_name(0))
print("Loading ImageNet-pretrained EfficientNet-B3...")

weights = models.EfficientNet_B3_Weights.DEFAULT
backbone = models.efficientnet_b3(weights=weights)

extractor = nn.Sequential(
    backbone.features,
    backbone.avgpool,
    nn.Flatten(1),
).to(DEVICE).eval()

print("Feature dimension: 1536")
print("Dataset:", DATA_ROOT)
print("Output:", OUT_ROOT)

for split in ("train", "val", "test"):
    split_dir = DATA_ROOT / split

    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    dataset = datasets.ImageFolder(
        root=str(split_dir),
        transform=transform,
    )

    if dataset.classes != ["benign", "malignant"]:
        raise ValueError(
            f"Unexpected class mapping in {split}: "
            f"{dataset.classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    features_list = []
    labels_list = []
    paths = []

    print(f"\nExtracting {split}: {len(dataset)} images")

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                features = extractor(images)

            features_list.append(features.float().cpu())
            labels_list.append(labels.long().cpu())

            start = batch_idx * BATCH_SIZE
            end = min(start + len(labels), len(dataset))

            paths.extend(
                dataset.samples[i][0]
                for i in range(start, end)
            )

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"{split}: batch {batch_idx + 1}/"
                    f"{len(loader)}"
                )

    all_features = torch.cat(features_list)
    all_labels = torch.cat(labels_list)

    output_path = OUT_ROOT / f"{split}_features.pt"

    torch.save({
        "features": all_features,
        "labels": all_labels,
        "paths": paths,
        "classes": dataset.classes,
        "checkpoint": "ImageNet-pretrained EfficientNet-B3",
    }, output_path)

    print("Saved:", output_path)
    print("Feature shape:", tuple(all_features.shape))
    print("Label counts:", torch.bincount(all_labels).tolist())

print("\nAll patient-split features extracted successfully.")
