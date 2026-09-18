from pathlib import Path
from collections import Counter

ROOT = Path(
    r"detection\raw_supplemental\Trash Detection Dataset\CUSTOM_DATASET"
)

names = [
    "Food Organics",
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

print("=" * 75)
print("SUPPLEMENTAL DATASET - TRAINING ANALYSIS")
print("=" * 75)

label_dir = ROOT / "train" / "labels"

if not label_dir.exists():
    print("ERROR: Training labels folder not found:")
    print(label_dir)
    raise SystemExit

total_files = 0
total_objects = 0

object_counts = Counter()
image_counts = Counter()
class_combinations = Counter()

for file in label_dir.glob("*.txt"):

    classes = set()

    try:
        for line in file.read_text().splitlines():

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 1:
                continue

            class_id = int(parts[0])

            if 0 <= class_id < 6:
                object_counts[class_id] += 1
                total_objects += 1
                classes.add(class_id)

    except Exception as e:
        print("Error reading:", file.name, e)
        continue

    if not classes:
        continue

    total_files += 1

    for class_id in classes:
        image_counts[class_id] += 1

    combination = tuple(sorted(classes))
    class_combinations[combination] += 1


print()
print("TRAINING DATA")
print("-" * 50)

print(f"Label files with objects : {total_files}")
print(f"Total objects            : {total_objects}")

print()
print("OBJECT COUNTS")
print("-" * 50)

for i, name in enumerate(names):
    print(f"{name:18}: {object_counts[i]}")

print()
print("IMAGES CONTAINING EACH CLASS")
print("-" * 50)

for i, name in enumerate(names):
    print(f"{name:18}: {image_counts[i]}")

print()
print("NUMBER OF CLASSES PER IMAGE")
print("-" * 50)

distribution = Counter()

for combination, count in class_combinations.items():

    number = len(combination)

    if number == 1:
        distribution["1 class"] += count
    elif number == 2:
        distribution["2 classes"] += count
    elif number == 3:
        distribution["3 classes"] += count
    else:
        distribution["4+ classes"] += count

for key in [
    "1 class",
    "2 classes",
    "3 classes",
    "4+ classes"
]:
    print(f"{key:15}: {distribution[key]}")

print()
print("CLASS COMBINATIONS")
print("-" * 50)

for combination, count in class_combinations.most_common():

    labels = [names[i] for i in combination]

    print(f"{' + '.join(labels):55}: {count}")

print()
print("=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)