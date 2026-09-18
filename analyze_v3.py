from pathlib import Path
from collections import Counter

BASE = Path(r"detection\dataset_v3")

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

print("=" * 70)
print("SMART WASTE CLASSIFIER - V3 DATASET ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# SPLIT COUNTS
# ------------------------------------------------------------

print()
print("DATASET SPLITS")
print("-" * 50)

for split in ["train", "val", "test"]:

    image_dir = BASE / "images" / split
    label_dir = BASE / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    labels = list(label_dir.glob("*.txt"))

    print(
        f"{split.upper():5} | "
        f"Images: {len(images):5} | "
        f"Labels: {len(labels):5}"
    )


# ------------------------------------------------------------
# TRAINING OBJECT COUNTS
# ------------------------------------------------------------

print()
print("TRAIN OBJECT COUNTS")
print("-" * 50)

train_labels = list((BASE / "labels" / "train").glob("*.txt"))

object_counts = Counter()
image_counts = Counter()
classes_per_image = Counter()

for label_file in train_labels:

    classes_in_image = set()

    for line in label_file.read_text().splitlines():

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) < 1:
            continue

        class_id = int(parts[0])

        if 0 <= class_id < 6:
            object_counts[class_id] += 1
            classes_in_image.add(class_id)

    for class_id in classes_in_image:
        image_counts[class_id] += 1

    classes_per_image[len(classes_in_image)] += 1


for class_id in range(6):

    print(
        f"{CLASS_NAMES[class_id]:18} : "
        f"{object_counts[class_id]:6} objects | "
        f"{image_counts[class_id]:5} images"
    )


# ------------------------------------------------------------
# CLASS PERCENTAGES
# ------------------------------------------------------------

total_objects = sum(object_counts.values())

print()
print("TRAIN OBJECT DISTRIBUTION")
print("-" * 50)

for class_id in range(6):

    percentage = (
        object_counts[class_id] / total_objects * 100
        if total_objects > 0
        else 0
    )

    print(
        f"{CLASS_NAMES[class_id]:18} : "
        f"{percentage:6.2f}%"
    )

print(f"{'TOTAL':18} : {total_objects:6} objects")


# ------------------------------------------------------------
# CLASSES PER IMAGE
# ------------------------------------------------------------

print()
print("CLASSES PER TRAIN IMAGE")
print("-" * 50)

for number_of_classes in sorted(classes_per_image):

    print(
        f"{number_of_classes} class(es): "
        f"{classes_per_image[number_of_classes]} images"
    )


# ------------------------------------------------------------
# MULTI-CLASS PERCENTAGE
# ------------------------------------------------------------

total_train_images = len(train_labels)

multiclass_images = sum(
    count
    for class_count, count in classes_per_image.items()
    if class_count >= 2
)

multiclass_percentage = (
    multiclass_images / total_train_images * 100
    if total_train_images > 0
    else 0
)

print()
print("MIXED-SCENE INFORMATION")
print("-" * 50)

print(f"Training images       : {total_train_images}")
print(f"Multi-class images    : {multiclass_images}")
print(f"Multi-class percentage: {multiclass_percentage:.2f}%")

print()
print("=" * 70)
print("V3 ANALYSIS COMPLETE")
print("=" * 70)