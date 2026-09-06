from ultralytics import YOLO
from pathlib import Path


# ============================================================
# SMART WASTE DETECTION
# TEST TRAINED YOLO MODEL
# ============================================================

print("=" * 70)
print("SMART WASTE DETECTION - MODEL TEST")
print("=" * 70)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model_path = Path(
    r"C:\Users\misty\OneDrive\Documents\Smart_Waste_Classifier\runs\detect\detection\runs\waste_yolo-5\weights\best.pt"
)

if not model_path.exists():
    print("\nERROR: best.pt not found!")
    print(model_path)
    exit()

print("\nModel:")
print(model_path)


# ------------------------------------------------------------
# TEST IMAGES
# ------------------------------------------------------------

test_images = Path(
    r"C:\Users\misty\OneDrive\Documents\Smart_Waste_Classifier\detection\dataset\images\test"
)

if not test_images.exists():
    print("\nERROR: Test image directory not found!")
    print(test_images)
    exit()

print("\nTest directory:")
print(test_images)


# ------------------------------------------------------------
# FIND IMAGES
# ------------------------------------------------------------

# Use rglob and a set so Windows cannot add
# the same image twice.

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

images = sorted(
    {
        image.resolve()
        for image in test_images.iterdir()
        if image.is_file()
        and image.suffix in image_extensions
    }
)


print("\nUnique test images found:", len(images))


if len(images) == 0:
    print("\nERROR: No images found!")
    exit()


# ------------------------------------------------------------
# SELECT IMAGES
# ------------------------------------------------------------

NUMBER_OF_IMAGES = 20

images_to_test = images[:NUMBER_OF_IMAGES]

print("Images selected:", len(images_to_test))

print("\nSelected images:")

for image in images_to_test:
    print(" ", image.name)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = YOLO(str(model_path))

print("Model loaded successfully.")


# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

print("\nClasses:")

for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")


# ------------------------------------------------------------
# RUN DETECTION
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("RUNNING DETECTION")
print("=" * 70)


results = model.predict(
    source=[str(image) for image in images_to_test],

    # Start with 0.25 confidence
    conf=0.25,

    imgsz=640,

    # RTX 4060
    device=0,

    # Save images with boxes
    save=True,

    # Save YOLO text predictions
    save_txt=True,

    save_conf=True,

    project="detection/test_results",

    name="waste_test",

    exist_ok=True,

    verbose=True
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("DETECTION RESULTS")
print("=" * 70)


total_detections = 0


for image_path, result in zip(images_to_test, results):

    print("\nImage:", image_path.name)

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("  No waste detected.")

        continue


    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        total_detections += 1

        print(
            f"  {class_name:20s}"
            f" confidence={confidence:.2f}"
        )


# ------------------------------------------------------------
# FINISHED
# ------------------------------------------------------------

output_directory = (
    Path("detection/test_results/waste_test").resolve()
)


print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nUnique images tested:", len(images_to_test))

print("Total detections:", total_detections)

print("\nAnnotated images:")

print(output_directory)

print("\nOpen this folder and inspect the images.")

print("=" * 70)