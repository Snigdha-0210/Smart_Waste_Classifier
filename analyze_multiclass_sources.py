from pathlib import Path
from collections import Counter

print("=" * 75)
print("SMART WASTE DETECTION - MULTI-CLASS SOURCE ANALYSIS")
print("=" * 75)

sources = {
    "TACO": Path(r"detection\dataset_taco_original"),
    "Combined": Path(r"detection\dataset_combined"),
    "Balanced V2": Path(r"detection\dataset_v2"),
}

names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

for source_name, root in sources.items():

    print(f"\n{source_name}")
    print("-" * 50)

    label_root = root / "labels" / "train"

    if not label_root.exists():
        print("Training labels not found:", label_root)
        continue

    distribution = Counter()
    class_presence = Counter()

    total_images = 0

    for file in label_root.glob("*.txt"):

        classes = set()

        try:
            for line in file.read_text().splitlines():

                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) >= 1:
                    class_id = int(parts[0])

                    if 0 <= class_id < len(names):
                        classes.add(class_id)

        except Exception:
            continue

        if not classes:
            continue

        total_images += 1

        number_of_classes = len(classes)

        if number_of_classes == 1:
            distribution["1 class"] += 1
        elif number_of_classes == 2:
            distribution["2 classes"] += 1
        elif number_of_classes == 3:
            distribution["3 classes"] += 1
        else:
            distribution["4+ classes"] += 1

        for class_id in classes:
            class_presence[class_id] += 1

    print(f"Training images: {total_images}")

    print("\nImages by number of classes:")
    for key in ["1 class", "2 classes", "3 classes", "4+ classes"]:
        print(f"{key:15}: {distribution[key]}")

    print("\nImages containing each class:")
    for i, name in enumerate(names):
        print(f"{name:18}: {class_presence[i]}")

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)