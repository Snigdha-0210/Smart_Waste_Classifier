import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from prepare_data import get_dataloaders
from model import WasteCNN


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

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=32
)

print("Data loaded!")


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = WasteCNN(num_classes=6)

model.load_state_dict(
    torch.load(
        "waste_classifier.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded!")


# ============================================================
# 5. GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")
print("=" * 60)

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 7. DISPLAY MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    xticks_rotation=45
)

plt.title("Waste Classifier - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")