import timm
import torch.nn as nn
from torchvision.ops import MLP


class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNet_B0, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )

        self.mlp = MLP(
            in_channels=1280,
            hidden_channels=[720, 120, num_classes],
            dropout=0.3,
        )

    def forward(self, x):
        x = self.model(x)
        x = self.mlp(x)
        return x
