import torch
import torch.nn as nn


class WasteCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        # ==========================================
        # CONVOLUTIONAL FEATURE EXTRACTOR
        # ==========================================

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)


            ,

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),


            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )


        # ==========================================
        # CLASSIFIER
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==============================================
# TEST THE MODEL
# ==============================================

if __name__ == "__main__":

    print("Creating model...")

    model = WasteCNN(num_classes=6)

    print(model)

    # Create fake batch of images
    test_images = torch.randn(
        32,
        3,
        128,
        128
    )

    print("\nInput shape:")
    print(test_images.shape)

    # Forward pass
    output = model(test_images)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput:")
    print(output)

    print("\nModel test successful!")