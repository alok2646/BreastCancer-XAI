import torch
import torch.nn as nn

from torchvision.models import swin_t, Swin_T_Weights


# ==========================================================
# Configuration
# ==========================================================

NUM_CLASSES = 2
DROPOUT = 0.30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Swin Transformer-Tiny Model
# ==========================================================

class BreastCancerNet(nn.Module):

    def __init__(self):

        super(BreastCancerNet, self).__init__()

        # --------------------------------------------------
        # Load pretrained Swin Transformer-Tiny
        # --------------------------------------------------

        weights = Swin_T_Weights.DEFAULT

        self.backbone = swin_t(weights=weights)

        # --------------------------------------------------
        # Get classifier input features
        # Swin-T output features = 768
        # --------------------------------------------------

        in_features = self.backbone.head.in_features

        # --------------------------------------------------
        # Replace original classifier
        # --------------------------------------------------

        self.backbone.head = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(DROPOUT),

            nn.Linear(256, NUM_CLASSES)

        )

    def forward(self, x):

        return self.backbone(x)


# ==========================================================
# Model Test
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Breast Cancer XAI - Swin Transformer")
    print("=" * 60)

    print("Device :", DEVICE)
    print("Classes:", NUM_CLASSES)

    print("\nCreating Swin Transformer model...\n")

    model = BreastCancerNet().to(DEVICE)

    print(model)

    print("\n" + "=" * 60)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    print("=" * 60)

    # ------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------

    model.eval()

    dummy = torch.randn(
        2,
        3,
        224,
        224
    ).to(DEVICE)

    with torch.no_grad():

        output = model(dummy)

    print("Input Shape  :", dummy.shape)
    print("Output Shape :", output.shape)

    print("=" * 60)

    print("Swin Transformer Model Loaded Successfully!")
    print("GPU :", DEVICE)

    print("=" * 60)