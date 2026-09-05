import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import get_dataloaders, CLASS_NAMES
from model_resnet import WasteResNet


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = "waste_resnet18_best.pth"

MAX_ERRORS_TO_DISPLAY = 30


# ImageNet normalization values
MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading data...")

_, _, test_loader = get_dataloaders(
    batch_size=32
)

print("Test data loaded!")


# ============================================================
# 4. CREATE MODEL
# ============================================================

print("\nCreating ResNet18 model...")

model = WasteResNet(
    num_classes=6
)

model = model.to(device)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded!")


# ============================================================
# 6. FIND INCORRECT PREDICTIONS
# ============================================================

print("\nAnalyzing test set...")
print("=" * 70)

errors = []

total = 0
correct = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        for i in range(len(labels)):

            true_label = labels[i].item()
            predicted_label = predictions[i].item()
            confidence = confidences[i].item()

            total += 1

            if true_label == predicted_label:

                correct += 1

            else:

                # Store the image on CPU
                image_cpu = images[i].cpu()

                errors.append({
                    "image": image_cpu,
                    "true": true_label,
                    "predicted": predicted_label,
                    "confidence": confidence
                })


# ============================================================
# 7. SUMMARY
# ============================================================

accuracy = 100 * correct / total

print("\n" + "=" * 70)
print("ERROR ANALYSIS SUMMARY")
print("=" * 70)

print("Total test images :", total)
print("Correct           :", correct)
print("Incorrect         :", len(errors))
print("Accuracy          :", f"{accuracy:.2f}%")

print("=" * 70)


# ============================================================
# 8. SORT ERRORS BY CONFIDENCE
# ============================================================

# Highest-confidence mistakes first.
#
# These are especially interesting because the model
# was very confident but still wrong.

errors.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# ============================================================
# 9. PRINT TOP ERRORS
# ============================================================

print("\nTOP INCORRECT PREDICTIONS")
print("=" * 70)

for i, error in enumerate(errors):

    true_name = CLASS_NAMES[
        error["true"]
    ]

    predicted_name = CLASS_NAMES[
        error["predicted"]
    ]

    confidence = error["confidence"] * 100

    print(
        f"{i + 1:02d}. "
        f"True: {true_name:<18} "
        f"Predicted: {predicted_name:<18} "
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# 10. COUNT ERROR TYPES
# ============================================================

error_pairs = {}

for error in errors:

    true_name = CLASS_NAMES[
        error["true"]
    ]

    predicted_name = CLASS_NAMES[
        error["predicted"]
    ]

    pair = (
        true_name,
        predicted_name
    )

    error_pairs[pair] = (
        error_pairs.get(pair, 0) + 1
    )


print("\n" + "=" * 70)
print("ERROR TYPES")
print("=" * 70)

sorted_pairs = sorted(
    error_pairs.items(),
    key=lambda x: x[1],
    reverse=True
)

for (true_name, predicted_name), count in sorted_pairs:

    print(
        f"{true_name:<20} → "
        f"{predicted_name:<20} : "
        f"{count}"
    )


# ============================================================
# 11. DISPLAY INCORRECT IMAGES
# ============================================================

num_images = min(
    MAX_ERRORS_TO_DISPLAY,
    len(errors)
)

if num_images == 0:

    print("\nNo incorrect predictions!")

else:

    print(
        f"\nDisplaying top {num_images} "
        "highest-confidence mistakes..."
    )

    cols = 5
    rows = int(
        np.ceil(num_images / cols)
    )

    plt.figure(
        figsize=(18, 3.5 * rows)
    )

    for i in range(num_images):

        error = errors[i]

        image = error["image"]

        # ----------------------------------------------------
        # Undo ImageNet normalization
        # ----------------------------------------------------

        image = (
            image * STD
        ) + MEAN

        image = torch.clamp(
            image,
            0,
            1
        )

        image = image.permute(
            1,
            2,
            0
        ).numpy()

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        ax = plt.subplot(
            rows,
            cols,
            i + 1
        )

        ax.imshow(image)

        true_name = CLASS_NAMES[
            error["true"]
        ]

        predicted_name = CLASS_NAMES[
            error["predicted"]
        ]

        confidence = (
            error["confidence"] * 100
        )

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}\n"
            f"Confidence: {confidence:.1f}%"
        )

        ax.axis("off")

    plt.suptitle(
        "ResNet18 - Highest Confidence Incorrect Predictions",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        "resnet_error_analysis.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nError analysis image saved as:"
    )

    print(
        "resnet_error_analysis.png"
    )


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE!")
print("=" * 70)