import timm
import torch.nn as nn
from torchvision.ops import MLP
import torch


class EfficientNet_B0(nn.Module):
    def __init__(
        self, input_channels=3, num_classes=2, pretrained=True
    ):
        super(EfficientNet_B0, self).__init__()
        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )

        if input_channels != 3:
            # Replace first conv layer
            new_conv_stem = nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.feature_extractor.conv_stem.out_channels,
                kernel_size=self.feature_extractor.conv_stem.kernel_size,
                stride=self.feature_extractor.conv_stem.stride,
                padding=self.feature_extractor.conv_stem.padding,
                bias=self.feature_extractor.conv_stem.bias
                is not None,
            )
            with torch.no_grad():
                new_conv_stem.weight[:, :3] = (
                    self.feature_extractor.conv_stem.weight
                )
                new_conv_stem.weight[:, 3] = (
                    self.feature_extractor.conv_stem.weight.mean(
                        dim=1
                    )
                )

            self.feature_extractor.conv_stem = new_conv_stem

        self.fc = nn.Sequential(
            nn.Linear(1280, num_classes), nn.ReLU(), nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
