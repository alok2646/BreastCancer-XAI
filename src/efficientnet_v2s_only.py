import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)


class EfficientNetV2SOnly(nn.Module):

    def __init__(self, num_classes=2, dropout=0.30):

        super().__init__()

        # Pretrained EfficientNetV2-S
        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        # EfficientNetV2-S feature dimension
        in_features = self.backbone.classifier[1].in_features

        print(
            "EfficientNetV2-S feature dimension:",
            in_features
        )

        # Replace original classifier
        self.backbone.classifier = nn.Sequential(

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features,
                512
            ),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                512,
                256
            ),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("EFFICIENTNET-V2-S ONLY")
    print("=" * 60)

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = EfficientNetV2SOnly(
        num_classes=2,
        dropout=0.30
    ).to(device)

    x = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():

        output = model(x)

    print(
        "Input shape :",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        "\nModel test successful."
    )