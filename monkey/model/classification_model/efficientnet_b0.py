import timm
import torch.nn as nn


class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNet_B0, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x
