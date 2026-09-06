from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# YOLO OBJECT DETECTION TEST
# ============================================================

# Change this to the image you want to test
IMAGE_PATH = r"C:\Users\misty\Downloads\waste7.jpg"


print("=" * 70)
print("SMART WASTE - OBJECT DETECTION TEST")
print("=" * 70)

print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded!")

print("\nRunning object detection...")
print("-" * 70)


# Run detection
results = model(
    IMAGE_PATH,
    conf=0.25
)


# ============================================================
# DISPLAY DETECTIONS
# ============================================================

result = results[0]

print("\nDETECTED OBJECTS")
print("=" * 70)

if result.boxes is None or len(result.boxes) == 0:

    print("No objects detected.")

else:

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        coordinates = box.xyxy[0].tolist()

        print(
            f"{i + 1:02d}. "
            f"{class_name:<20} "
            f"Confidence: {confidence * 100:.2f}%"
        )

        print(
            f"    Box: "
            f"{[round(x, 1) for x in coordinates]}"
        )


# ============================================================
# SAVE DETECTION IMAGE
# ============================================================

output_path = "detected_waste.jpg"

annotated_image = result.plot()

Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

print("\n" + "=" * 70)
print("Detection image saved as:")
print(output_path)
print("=" * 70)