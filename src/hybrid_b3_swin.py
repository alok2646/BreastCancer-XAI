import torch
import torch.nn as nn
from torchvision import models


class HybridB3SwinNet(nn.Module):

    def __init__(
        self,
        num_classes=2,
        dropout=0.30,
        pretrained=True
    ):
        super().__init__()

        # ====================================================
        # 1. EfficientNet-B3
        # ====================================================

        if pretrained:
            b3_weights = models.EfficientNet_B3_Weights.DEFAULT
        else:
            b3_weights = None

        self.efficientnet = models.efficientnet_b3(
            weights=b3_weights
        )

        # EfficientNet-B3 feature dimension = 1536
        b3_features = self.efficientnet.classifier[1].in_features

        # Remove original classifier
        self.efficientnet.classifier = nn.Identity()


        # ====================================================
        # 2. Swin-Tiny
        # ====================================================

        if pretrained:
            swin_weights = models.Swin_T_Weights.DEFAULT
        else:
            swin_weights = None

        self.swin = models.swin_t(
            weights=swin_weights
        )

        # Swin-Tiny feature dimension = 768
        swin_features = self.swin.head.in_features

        # Remove original classifier
        self.swin.head = nn.Identity()


        # ====================================================
        # 3. Feature Fusion
        # ====================================================

        fusion_features = (
            b3_features +
            swin_features
        )


        # ====================================================
        # 4. Classification Head
        # ====================================================

        self.classifier = nn.Sequential(

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


        # Save feature dimensions
        self.b3_features = b3_features
        self.swin_features = swin_features
        self.fusion_features = fusion_features


    def forward(self, x):

        # EfficientNet-B3 features
        b3_output = self.efficientnet(x)

        # Swin-Tiny features
        swin_output = self.swin(x)

        # Feature fusion
        fused_features = torch.cat(
            [b3_output, swin_output],
            dim=1
        )

        # Final classification
        output = self.classifier(
            fused_features
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("EFFICIENTNET-B3 + SWIN-TINY HYBRID MODEL")
    print("=" * 60)

    model = HybridB3SwinNet(
        num_classes=2,
        dropout=0.30,
        pretrained=True
    )

    model = model.to(device)

    # Dummy input
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

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel Ready!")
    print(
        f"EfficientNet-B3 Features : "
        f"{model.b3_features}"
    )

    print(
        f"Swin-Tiny Features       : "
        f"{model.swin_features}"
    )

    print(
        f"Fusion Features          : "
        f"{model.fusion_features}"
    )

    print(
        f"Output Shape             : "
        f"{output.shape}"
    )

    print(
        f"Total Parameters         : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable Parameters     : "
        f"{trainable_parameters:,}"
    )

    print("\nModel test successful!")