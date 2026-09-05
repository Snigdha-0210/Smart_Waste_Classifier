import torch
import torch.nn as nn

from prepare_data import get_dataloaders, CLASS_NAMES
from model import WasteCNN


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

print("\nCreating model...")

model = WasteCNN(num_classes=6)

# Load our trained weights
model.load_state_dict(
    torch.load(
        "waste_classifier.pth",
        map_location=device
    )
)

model = model.to(device)

print("Trained model loaded!")


# ============================================================
# 4. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 5. EVALUATE ON TEST SET
# ============================================================

print("\nEvaluating on test set...")
print("=" * 60)

model.eval()

test_loss = 0.0
correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        # Get predictions
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


# ============================================================
# 6. CALCULATE RESULTS
# ============================================================

test_loss = test_loss / len(test_loader)

test_accuracy = 100 * correct / total


print("\nTEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")
print(f"Correct       : {correct}/{total}")


print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)