import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)


# ============================================================
# BASE PAPER: EfficientNetV2S
# ============================================================

class EfficientNetV2SBasePaper(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # ----------------------------------------------------
        # EfficientNetV2S
        # Without top/classifier
        # ImageNet pretrained weights
        # ----------------------------------------------------
        backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        # Remove original classification head
        self.backbone = backbone.features

        # EfficientNetV2-S output feature channels = 1280
        self.feature_dim = 1280

        # ----------------------------------------------------
        # Max Pooling
        # Paper: applies max pooling to backbone output
        # ----------------------------------------------------
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # ----------------------------------------------------
        # Batch Normalization
        #
        # Paper:
        # normalization axis = -1
        # decay = 0.99
        # epsilon = 0.001
        #
        # PyTorch momentum is equivalent to:
        # 1 - decay = 0.01
        # ----------------------------------------------------
        self.batch_norm = nn.BatchNorm1d(
            self.feature_dim,
            eps=0.001,
            momentum=0.01
        )

        # ----------------------------------------------------
        # Fully Connected Layer
        # 256 neurons + ReLU
        # ----------------------------------------------------
        self.fc = nn.Linear(
            self.feature_dim,
            256
        )

        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # Dropout = 0.45
        # ----------------------------------------------------
        self.dropout = nn.Dropout(
            p=0.45
        )

        # ----------------------------------------------------
        # Output Layer
        # 2 classes
        #
        # IMPORTANT:
        # We return logits during training.
        # CrossEntropyLoss internally handles softmax.
        # ----------------------------------------------------
        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        # EfficientNetV2S feature extraction
        x = self.backbone(x)

        # Max pooling
        x = self.max_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Batch normalization
        x = self.batch_norm(x)

        # Dense 256
        x = self.fc(x)

        # ReLU
        x = self.relu(x)

        # Dropout
        x = self.dropout(x)

        # Output logits
        x = self.classifier(x)

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = EfficientNetV2SBasePaper(
        num_classes=2
    ).to(device)

    # Base paper uses 50x50 images
    dummy_input = torch.randn(
        2,
        3,
        50,
        50
    ).to(device)

    with torch.no_grad():

        output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters    :", total_params)
    print("Trainable parameters:", trainable_params)

    print("\nModel test successful!")