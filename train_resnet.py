import torch
import torch.nn as nn
import torch.optim as optim

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
# 3. CREATE RESNET18 MODEL
# ============================================================

print("\nCreating ResNet18 model...")

model = WasteResNet(
    num_classes=6
)

model = model.to(device)

print("Model moved to:", device)


# ============================================================
# 4. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 5. OPTIMIZER
# ============================================================

# Small learning rate because we are using a pretrained
# ResNet18.

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)


# ============================================================
# 6. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 7. TRAINING SETTINGS
# ============================================================

epochs = 15

best_val_accuracy = 0.0


# ============================================================
# 8. TRAINING LOOP
# ============================================================

print("\nStarting ResNet18 training...")
print("=" * 70)


for epoch in range(epochs):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss = (
        val_loss /
        len(val_loader)
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "waste_resnet18_best.pth"
        )

        print(
            f"  ✓ Best model saved! "
            f"Validation accuracy: "
            f"{val_accuracy:.2f}%"
        )


# ============================================================
# 9. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("RESNET18 TRAINING COMPLETE!")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved as:"
)

print(
    "waste_resnet18_best.pth"
)

print("=" * 70)