"""
Hybrid EfficientNet-B3 + Swin-Tiny Model
Breast Cancer Classification

Input:
    224 x 224 RGB image

Architecture:
    Image
       |
       +----> EfficientNet-B3 ----> 1536-D features
       |
       +----> Swin-Tiny ----------> 768-D features
                    |
              Feature Fusion
                    |
                2304-D
                    |
              FC 512 + Dropout
                    |
              FC 256 + Dropout
                    |
                FC 2
                    |
          Benign / Malignant
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights,
    swin_t,
    Swin_T_Weights
)


class HybridBreastCancerNet(nn.Module):

    def __init__(
        self,
        num_classes=2,
        dropout=0.30,
        pretrained=True
    ):
        super().__init__()

        # ============================================================
        # 1. EfficientNet-B3 Branch
        # ============================================================

        if pretrained:
            efficient_weights = EfficientNet_B3_Weights.DEFAULT
        else:
            efficient_weights = None

        self.efficientnet = efficientnet_b3(
            weights=efficient_weights
        )

        # EfficientNet-B3 feature dimension = 1536
        efficient_features = self.efficientnet.classifier[1].in_features

        # Remove original classifier
        self.efficientnet.classifier = nn.Identity()


        # ============================================================
        # 2. Swin-Tiny Branch
        # ============================================================

        if pretrained:
            swin_weights = Swin_T_Weights.DEFAULT
        else:
            swin_weights = None

        self.swin = swin_t(
            weights=swin_weights
        )

        # Swin-Tiny feature dimension = 768
        swin_features = self.swin.head.in_features

        # Remove original classification head
        self.swin.head = nn.Identity()


        # ============================================================
        # 3. Feature Fusion
        # ============================================================

        fusion_features = efficient_features + swin_features

        self.fusion_classifier = nn.Sequential(

            nn.Linear(
                fusion_features,
                512
            ),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(dropout),


            nn.Linear(
                512,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(dropout),


            nn.Linear(
                256,
                num_classes
            )
        )


        # Store dimensions for debugging
        self.efficient_features = efficient_features
        self.swin_features = swin_features
        self.fusion_features = fusion_features


    def forward(self, x):

        # ============================================================
        # EfficientNet Feature Extraction
        # ============================================================

        efficient_features = self.efficientnet(x)

        # ============================================================
        # Swin Feature Extraction
        # ============================================================

        swin_features = self.swin(x)

        # ============================================================
        # Feature Fusion
        # ============================================================

        fused_features = torch.cat(
            [efficient_features, swin_features],
            dim=1
        )

        # ============================================================
        # Classification
        # ============================================================

        output = self.fusion_classifier(
            fused_features
        )

        return output


# ================================================================
# Model Test
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HYBRID EFFICIENTNET-B3 + SWIN-TINY MODEL")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    model = HybridBreastCancerNet(
        num_classes=2,
        dropout=0.30,
        pretrained=False
    )

    model = model.to(device)

    # Dummy input
    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    # Forward pass
    with torch.no_grad():

        output = model(dummy_input)

    print("\nModel Information:")
    print("-" * 60)

    print(
        f"EfficientNet-B3 features : "
        f"{model.efficient_features}"
    )

    print(
        f"Swin-Tiny features       : "
        f"{model.swin_features}"
    )

    print(
        f"Fused features           : "
        f"{model.fusion_features}"
    )

    print(
        f"Output shape             : "
        f"{output.shape}"
    )

    print("\nExpected:")
    print("Input  : [2, 3, 224, 224]")
    print("Output : [2, 2]")

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED")
    print("=" * 60)