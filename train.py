import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import get_dataloaders
from model import WasteCNN


# ==============================
# 1. DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# 2. LOAD DATA
# ==============================

print("\nLoading data...")

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=32
)

print("Data loaded!")


# ==============================
# 3. CREATE MODEL
# ==============================

print("\nCreating model...")

model = WasteCNN(num_classes=6)

model = model.to(device)

print("Model moved to:", device)


# ==============================
# 4. LOSS FUNCTION
# ==============================

criterion = nn.CrossEntropyLoss()


# ==============================
# 5. OPTIMIZER
# ==============================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==============================
# 6. TRAINING
# ==============================

epochs = 10

print("\nStarting training...")
print("=" * 60)


for epoch in range(epochs):

    # --------------------------
    # TRAIN
    # --------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # --------------------------
    # VALIDATION
    # --------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()


    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # --------------------------
    # PRINT RESULTS
    # --------------------------

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# ==============================
# 7. SAVE MODEL
# ==============================

torch.save(
    model.state_dict(),
    "waste_classifier.pth"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("Model saved as: waste_classifier.pth")
print("=" * 60)