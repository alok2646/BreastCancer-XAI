import torch
import torch.nn as nn
from torchvision import models

# =====================================================
# Configuration
# =====================================================

NUM_CLASSES = 2
DROPOUT = 0.40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Breast Cancer XAI Model")
print("=" * 60)
print("Device :", DEVICE)
print("Classes:", NUM_CLASSES)
print("=" * 60)


# =====================================================
# EfficientNet-B3 Model
# =====================================================

class BreastCancerNet(nn.Module):

    def __init__(self):
        super(BreastCancerNet, self).__init__()

        # Load pretrained EfficientNet-B3
        self.backbone = models.efficientnet_b6(
            weights=models.EfficientNet_B6_Weights.DEFAULT
        )

        # Number of input features
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(

            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(512, NUM_CLASSES)

        )

    def forward(self, x):
        return self.backbone(x)

    # =====================================================
# Model Test
# =====================================================

if __name__ == "__main__":

    print("\nCreating Model...\n")

    model = BreastCancerNet().to(DEVICE)

    print(model)

    print("\n" + "=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    print("=" * 60)

    # Dummy Input
    model.eval()
    
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        output = model(dummy)

    print("Input Shape :", dummy.shape)
    print("Output Shape:", output.shape)

    print("=" * 60)

    print("Model Loaded Successfully.")
    print("GPU :", DEVICE)

    print("=" * 60)

    