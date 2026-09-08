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

food_counts = []

for label_file in LABEL_DIR.glob("*.txt"):
    food_count = 0

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

        if class_id == 1:
            food_count += 1

    if food_count > 0:
        food_counts.append((food_count, label_file.name))


print("=" * 70)
print("FOOD ORGANICS IMAGE DISTRIBUTION")
print("=" * 70)

print()
print(f"Images containing Food Organics: {len(food_counts)}")
print(f"Total Food Organics objects: {sum(x[0] for x in food_counts)}")

print()
print("OBJECT COUNT RANGES PER IMAGE")
print("-" * 70)

ranges = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201+", 201, 10**9),
]

for name, low, high in ranges:
    matching = [
        (count, filename)
        for count, filename in food_counts
        if low <= count <= high
    ]

    objects = sum(count for count, _ in matching)

    print(
        f"{name:<10} : "
        f"{len(matching):>4} images | "
        f"{objects:>6} objects"
    )

print()
print("TOP 30 IMAGES BY FOOD ORGANICS COUNT")
print("-" * 70)

for count, filename in sorted(
    food_counts,
    reverse=True
)[:30]:

    print(f"{count:>4} Food Organics : {filename}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)