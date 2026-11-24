import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights,
    swin_t,
    Swin_T_Weights
)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of EfficientNet-B3 and Swin-Tiny features.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class HybridB3SwinMultiScaleNet(nn.Module):

    def __init__(self, num_classes=2, dropout=0.30):

        super().__init__()

        # =========================================================
        # 1. EfficientNet-B3
        # =========================================================

        self.efficientnet = efficientnet_b3(
            weights=EfficientNet_B3_Weights.DEFAULT
        )

        # Remove original classifier
        self.efficientnet.classifier = nn.Identity()

        # EfficientNet-B3 feature dimension = 1536
        self.eff_dim = 1536

        # =========================================================
        # 2. Swin-Tiny
        # =========================================================

        self.swin = swin_t(
            weights=Swin_T_Weights.DEFAULT
        )

        # Remove original classifier
        self.swin.head = nn.Identity()

        # Swin-Tiny feature dimension = 768
        self.swin_dim = 768

        # =========================================================
        # 3. Projection layers
        # =========================================================

        self.eff_projection = nn.Sequential(
            nn.Linear(self.eff_dim, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        self.swin_projection = nn.Sequential(
            nn.Linear(self.swin_dim, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        # =========================================================
        # 4. Attention-based Fusion
        # =========================================================

        self.fusion_dim = 1536

        self.attention_fusion = AttentionFusion(
            self.fusion_dim
        )

        # =========================================================
        # 5. Fusion classifier
        # =========================================================

        self.classifier = nn.Sequential(

            nn.Linear(self.fusion_dim, 512),

            nn.LayerNorm(512),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(512, 256),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # EfficientNet features
        eff_features = self.efficientnet(x)

        # Swin features
        swin_features = self.swin(x)

        # Project both feature spaces
        eff_features = self.eff_projection(
            eff_features
        )

        swin_features = self.swin_projection(
            swin_features
        )

        # Concatenate projected features
        fused_features = torch.cat(
            [eff_features, swin_features],
            dim=1
        )

        # Attention-based feature refinement
        fused_features = self.attention_fusion(
            fused_features
        )

        # Classification
        output = self.classifier(
            fused_features
        )

        return output


# =============================================================
# MODEL TEST
# =============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = HybridB3SwinMultiScaleNet(
        num_classes=2,
        dropout=0.30
    ).to(device)

    # Dummy input
    x = torch.randn(
        2, 3, 224, 224
    ).to(device)

    with torch.no_grad():

        output = model(x)

    print("Input shape :", x.shape)
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

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    print("\nModel test successful.")