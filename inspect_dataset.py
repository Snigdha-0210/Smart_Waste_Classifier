from datasets import load_dataset
from collections import Counter

print("Loading dataset...\n")

dataset = load_dataset("ddompe/waste-segregation-dataset")

print("Dataset loaded successfully!\n")

# --------------------------------------------------
# 1. Dataset split sizes
# --------------------------------------------------

print("=" * 50)
print("DATASET SPLITS")
print("=" * 50)

for split_name, split in dataset.items():
    print(f"{split_name:12} : {len(split)} images")

# --------------------------------------------------
# 2. Class distribution
# --------------------------------------------------

print("\n" + "=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)

for split_name, split in dataset.items():

    labels = split["label_name"]
    counts = Counter(labels)

    print(f"\n{split_name.upper()}")

    for class_name, count in sorted(counts.items()):
        print(f"{class_name:25} : {count}")

# --------------------------------------------------
# 3. All classes
# --------------------------------------------------

print("\n" + "=" * 50)
print("ALL CLASSES")
print("=" * 50)

all_classes = sorted(set(dataset["train"]["label_name"]))

for i, class_name in enumerate(all_classes):
    print(f"{i}: {class_name}")

print("\nInspection complete.")