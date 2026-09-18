from pathlib import Path
from collections import Counter

BASE = Path(r"detection\dataset_v3")
NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

print("=" * 70)
print("SMART WASTE CLASSIFIER - INSPECTING V3 LABELS")
print("=" * 70)

labels = list((BASE / "labels" / "train").glob("*.txt"))

counts = Counter()
source_like = Counter()

for label in labels:
    for line in label.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) >= 1:
            class_id = int(parts[0])

            if 0 <= class_id < 6:
                counts[class_id] += 1

print()
print("V3 TRAINING LABEL COUNTS")
print("-" * 50)

for i in range(6):
    print(f"{i} = {NAMES[i]:18} : {counts[i]:6} objects")

print()
print("TOTAL:", sum(counts.values()))

print()
print("=" * 70)
print("SAMPLE V3 LABEL FILES")
print("=" * 70)

for label in labels[:20]:
    print()
    print(label.name)

    lines = [
        line.strip()
        for line in label.read_text().splitlines()
        if line.strip()
    ]

    print("First labels:", lines[:5])

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)