import torch
import torch.nn as nn

from prepare_data import get_dataloaders
from model_resnet import WasteResNet


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading data...")

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=32
)

print("Data loaded!")


# ============================================================
# 3. CREATE MODEL
# ============================================================

print("\nCreating ResNet18 model...")

model = WasteResNet(
    num_classes=6
)

model = model.to(device)


# ============================================================
# 4. LOAD BEST TRAINED MODEL
# ============================================================

print("\nLoading best trained model...")

model.load_state_dict(
    torch.load(
        "waste_resnet18_best.pth",
        map_location=device
    )
)

model.eval()

print("Best model loaded!")


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. EVALUATE ON TEST SET
# ============================================================

print("\nEvaluating on test set...")
print("=" * 70)

test_loss = 0.0
correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        # Predictions
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ============================================================
# 7. CALCULATE RESULTS
# ============================================================

test_loss = (
    test_loss /
    len(test_loader)
)

test_accuracy = (
    100.0 * correct / total
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RESNET18 TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.2f}%"
)

print(
    f"Correct       : {correct}/{total}"
)

print("=" * 70)


# ============================================================
# 9. COMPARISON
# ============================================================

old_cnn_accuracy = 59.77

improvement = (
    test_accuracy -
    old_cnn_accuracy
)

print("\nMODEL COMPARISON")
print("=" * 70)

print(
    f"Old CNN       : {old_cnn_accuracy:.2f}%"
)

print(
    f"ResNet18      : {test_accuracy:.2f}%"
)

print(
    f"Improvement   : {improvement:+.2f} percentage points"
)

print("=" * 70)


print("\nEvaluation complete!")