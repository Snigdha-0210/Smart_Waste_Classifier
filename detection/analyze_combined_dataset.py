from pathlib import Path
from collections import Counter

ROOT = Path("detection/dataset_combined")
LABEL_DIR = ROOT / "labels" / "train"

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
]

counts = Counter()
images_with_class = Counter()
objects_per_image = Counter()
class_combinations = Counter()

label_files = list(LABEL_DIR.glob("*.txt"))

print("=" * 70)
print("COMBINED YOLO TRAINING DATASET ANALYSIS")
print("=" * 70)

for label_file in label_files:
    image_classes = []
    object_count = 0

    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        continue

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if 0 <= class_id < len(CLASS_NAMES):
            counts[class_id] += 1
            object_count += 1

            if class_id not in image_classes:
                image_classes.append(class_id)

    for class_id in image_classes:
        images_with_class[class_id] += 1

    if object_count > 0:
        objects_per_image[object_count] += 1

    combination = tuple(sorted(image_classes))
    if combination:
        class_combinations[combination] += 1


print()
print("1. OBJECT COUNTS")
print("-" * 70)

total_objects = sum(counts.values())

for class_id, name in enumerate(CLASS_NAMES):
    count = counts[class_id]
    percentage = (count / total_objects * 100) if total_objects else 0

    print(
        f"{name:<18} : "
        f"{count:>6} objects | "
        f"{images_with_class[class_id]:>5} images | "
        f"{percentage:>6.2f}%"
    )

print("-" * 70)
print(f"{'TOTAL':<18} : {total_objects:>6} objects")
print(f"{'TRAIN LABEL FILES':<18} : {len(label_files):>6}")


print()
print("2. OBJECTS PER IMAGE")
print("-" * 70)

for count, image_count in sorted(objects_per_image.items()):
    print(f"{count:>4} objects/image : {image_count:>5} images")


print()
print("3. IMAGES CONTAINING MULTIPLE WASTE CLASSES")
print("-" * 70)

multi_class_images = 0

for combination, image_count in sorted(
    class_combinations.items(),
    key=lambda x: (-x[1], x[0])
):
    if len(combination) >= 2:
        multi_class_images += image_count

        names = " + ".join(CLASS_NAMES[i] for i in combination)

        print(f"{image_count:>5} images : {names}")

print("-" * 70)
print(f"Images containing 2+ classes: {multi_class_images}")


print()
print("4. FOOD ORGANICS ANALYSIS")
print("-" * 70)

food_objects = counts[1]
food_images = images_with_class[1]

print(f"Food Organics objects : {food_objects}")
print(f"Food Organics images  : {food_images}")

if food_images:
    print(
        f"Average Food Organics objects/image: "
        f"{food_objects / food_images:.2f}"
    )

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
