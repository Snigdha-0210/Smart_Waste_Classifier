from pathlib import Path
from collections import Counter

ROOT = Path(r"detection")

folders = {
    "TACO original": ROOT / "dataset_taco_original",
    "Supplemental": ROOT / "raw_supplemental" / "Trash Detection Dataset" / "CUSTOM_DATASET",
    "Combined": ROOT / "dataset_combined",
    "Balanced V2": ROOT / "dataset_v2",
}

names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

print("=" * 75)
print("SMART WASTE DETECTION - DATASET SOURCE ANALYSIS")
print("=" * 75)

for source_name, root in folders.items():

    print(f"\n{source_name}")
    print("-" * 50)

    if not root.exists():
        print("Folder not found:", root)
        continue

    label_root = root / "labels"

    if not label_root.exists():
        print("No labels folder:", label_root)
        continue

    counts = Counter()
    files = 0
    objects = 0

    for split in ["train", "val", "test"]:

        split_dir = label_root / split

        if not split_dir.exists():
            continue

        for file in split_dir.glob("*.txt"):

            files += 1

            try:
                for line in file.read_text().splitlines():

                    if not line.strip():
                        continue

                    parts = line.split()

                    if len(parts) >= 1:
                        class_id = int(parts[0])

                        if 0 <= class_id < len(names):
                            counts[class_id] += 1
                            objects += 1

            except Exception as e:
                print("Error:", file, e)

    print("Label files :", files)
    print("Objects     :", objects)

    for i, name in enumerate(names):
        print(f"{name:18}: {counts[i]}")

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)