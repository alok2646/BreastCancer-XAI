"""
HYBRID EFFICIENTNET-V2-S + SWIN-TINY
Breast Cancer Classification

EfficientNetV2-S features : 1280
Swin-Tiny features        : 768
Fusion features           : 2048

Classes:
    0 = benign
    1 = malignant
"""

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    swin_t,
    Swin_T_Weights
)


# ============================================================
# MODEL
# ============================================================

class HybridV2SSwinNet(nn.Module):

    def __init__(
        self,
        num_classes=2,
        dropout=0.30,
        pretrained=True
    ):

        super().__init__()


        # ====================================================
        # EFFICIENTNET-V2-S
        # ====================================================

        if pretrained:

            efficient_weights = (
                EfficientNet_V2_S_Weights.DEFAULT
            )

        else:

            efficient_weights = None


        self.efficientnet = efficientnet_v2_s(
            weights=efficient_weights
        )


        # EfficientNetV2-S classifier input = 1280
        self.efficient_features = (
            self.efficientnet.classifier[1].in_features
        )


        # Remove original classifier
        self.efficientnet.classifier = nn.Identity()


        # ====================================================
        # SWIN-TINY
        # ====================================================

        if pretrained:

            swin_weights = (
                Swin_T_Weights.DEFAULT
            )

        else:

            swin_weights = None


        self.swin = swin_t(
            weights=swin_weights
        )


        # Swin-Tiny feature dimension = 768
        self.swin_features = (
            self.swin.head.in_features
        )


        # Remove original classification head
        self.swin.head = nn.Identity()


        # ====================================================
        # FUSION
        # ====================================================

        self.fusion_features = (
            self.efficient_features
            +
            self.swin_features
        )


        # ====================================================
        # FUSION CLASSIFIER
        # ====================================================

        self.fusion_classifier = nn.Sequential(

            nn.Linear(
                self.fusion_features,
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


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # EfficientNetV2-S
        # ----------------------------------------------------

        efficient_features = (
            self.efficientnet(x)
        )


        # ----------------------------------------------------
        # Swin-Tiny
        # ----------------------------------------------------

        swin_features = (
            self.swin(x)
        )


        # ----------------------------------------------------
        # Feature Fusion
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                efficient_features,
                swin_features
            ],
            dim=1
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        output = self.fusion_classifier(
            fused_features
        )


        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HYBRID EFFICIENTNET-V2-S + SWIN-TINY")
    print("=" * 70)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\nDevice:",
        device
    )


    print(
        "\nLoading model..."
    )


    model = HybridV2SSwinNet(
        num_classes=2,
        dropout=0.30,
        pretrained=True
    )


    model = model.to(device)


    print(
        "\nModel Information:"
    )

    print(
        "-" * 60
    )

    print(
        "EfficientNetV2-S features :",
        model.efficient_features
    )

    print(
        "Swin-Tiny features        :",
        model.swin_features
    )

    print(
        "Fusion features           :",
        model.fusion_features
    )


    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)


    with torch.no_grad():

        output = model(
            dummy_input
        )


    print(
        "Input shape               :",
        dummy_input.shape
    )

    print(
        "Output shape              :",
        output.shape
    )


    print(
        "\nExpected:"
    )

    print(
        "Input  : [2, 3, 224, 224]"
    )

    print(
        "Output : [2, 2]"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL TEST COMPLETED"
    )

    print(
        "=" * 70
    )