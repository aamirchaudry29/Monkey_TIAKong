import timm
import torch
import torch.nn as nn


class EfficientNet_B0(nn.Module):
    def __init__(
        self, input_channels=3, num_classes=2, pretrained=True
    ):
        super(EfficientNet_B0, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if input_channels != 3:
            # Replace first conv layer
            new_conv_stem = nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.model.conv_stem.out_channels,
                kernel_size=self.model.conv_stem.kernel_size,
                stride=self.model.conv_stem.stride,
                padding=self.model.conv_stem.padding,
                bias=self.model.conv_stem.bias is not None,
            )
            with torch.no_grad():
                new_conv_stem.weight[
                    :, :3
                ] = self.model.conv_stem.weight
                new_conv_stem.weight[
                    :, 3
                ] = self.model.conv_stem.weight.mean(dim=1)

            self.model.conv_stem = new_conv_stem

    def forward(self, x):
        x = self.model(x)
        return x
