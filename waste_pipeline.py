import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from model_resnet import WasteResNet
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = r"C:\Users\misty\Downloads\waste7.jpg"

RESNET_MODEL_PATH = "waste_resnet18_best.pth"

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SMART WASTE DETECTION + CLASSIFICATION PIPELINE")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

classifier = WasteResNet(num_classes=6)

classifier.load_state_dict(
    torch.load(
        RESNET_MODEL_PATH,
        map_location=device
    )
)

classifier = classifier.to(device)
classifier.eval()

print("ResNet18 loaded successfully!")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

detector = YOLO("yolo11n.pt")

print("YOLO loaded successfully!")


# ============================================================
# RUN OBJECT DETECTION
# ============================================================

print("\nRunning object detection...")
print("-" * 70)

results = detector(
    IMAGE_PATH,
    conf=0.25
)

result = results[0]


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("\nObjects detected:", len(result.boxes))


# ============================================================
# PROCESS EACH DETECTED OBJECT
# ============================================================

detections = []


for i, box in enumerate(result.boxes):

    class_id = int(box.cls[0])

    detector_confidence = float(box.conf[0])

    detector_class = detector.names[class_id]

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.width, int(x2))
    y2 = min(image.height, int(y2))

    # Crop detected object
    crop = image.crop((x1, y1, x2, y2))

    # Prepare for ResNet
    tensor = transform(crop)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    # ========================================================
    # RESNET CLASSIFICATION
    # ========================================================

    with torch.no_grad():

        output = classifier(tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    classifier_confidence = confidence.item()


    # Store result
    detections.append({
        "detector_class": detector_class,
        "detector_confidence": detector_confidence,
        "predicted_class": predicted_class,
        "classifier_confidence": classifier_confidence,
        "box": (x1, y1, x2, y2),
        "crop": crop
    })


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL WASTE CLASSIFICATION RESULTS")
print("=" * 70)


if len(detections) == 0:

    print("\nNo objects were detected.")

else:

    for i, detection in enumerate(detections):

        print(
            f"\nObject {i + 1}"
        )

        print(
            f"YOLO detected : "
            f"{detection['detector_class']}"
        )

        print(
            f"Detection confidence : "
            f"{detection['detector_confidence'] * 100:.2f}%"
        )

        print(
            f"ResNet prediction : "
            f"{detection['predicted_class']}"
        )

        print(
            f"Classification confidence : "
            f"{detection['classifier_confidence'] * 100:.2f}%"
        )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

print("\nCreating visualization...")


plt.figure(figsize=(14, 10))

plt.imshow(image)

for detection in detections:

    x1, y1, x2, y2 = detection["box"]

    predicted_class = detection["predicted_class"]

    confidence = (
        detection["classifier_confidence"] * 100
    )

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=3
    )

    plt.gca().add_patch(rectangle)

    plt.text(
        x1,
        y1,
        f"{predicted_class} {confidence:.1f}%",
        fontsize=12,
        backgroundcolor="white"
    )


plt.axis("off")

plt.title(
    "Smart Waste Detection + ResNet18 Classification"
)

plt.tight_layout()

output_file = "waste_pipeline_result.jpg"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nResult saved as:")
print(output_file)

print("\n")
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)