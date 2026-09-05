import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from prepare_data import get_dataloaders
from model_resnet import WasteResNet


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading data...")

_, _, test_loader = get_dataloaders(batch_size=32)

print("Test data loaded!")


# ============================================================
# 4. CREATE MODEL
# ============================================================

print("\nCreating ResNet18 model...")

model = WasteResNet(num_classes=6)

model = model.to(device)


# ============================================================
# 5. LOAD TRAINED MODEL
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
# 6. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")
print("=" * 70)

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ============================================================
# 9. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ResNet18 Waste Classification Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix_resnet.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix_resnet.png")