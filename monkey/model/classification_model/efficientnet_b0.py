import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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


if __name__ == "__main__":
    model = EfficientNet_B0(num_classes=1)
    model.eval()
    summary(model)
    test = torch.ones(size=(1, 3, 32, 32), dtype=torch.float32)
    gt = torch.tensor([[1]], dtype=torch.float32)

    loss_fn = nn.BCELoss()

    with torch.no_grad():
        out = model(test)
        out = torch.sigmoid(out)

        loss = loss_fn(out, gt)

    print(out)
    print(loss)
