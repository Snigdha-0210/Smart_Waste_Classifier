import torch
import torch.nn as nn
from torchvision import models


class WasteResNet(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Get number of inputs to final layer
        num_features = self.model.fc.in_features

        # Replace final classifier
        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("Creating ResNet18 model...")

    model = WasteResNet(num_classes=6)

    print(model)

    # Test input
    x = torch.randn(4, 3, 224, 224)

    print("\nInput shape:")
    print(x.shape)

    output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    print("\nModel test successful!")