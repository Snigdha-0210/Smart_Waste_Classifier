from pathlib import Path
import shutil
import random

# ============================================================
# SMART WASTE CLASSIFIER
# FINAL V4 DATASET BUILDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V4_ROOT = (
    PROJECT_ROOT
    / "detection"
    / "v4_source"
    / "extracted"
    / "A synthetic outdoor waste image dataset with YOLO-"
    / "SyntheticOutdoorWaste_YOLO"
    / "SynWasteNet"
)

OUTPUT_ROOT = PROJECT_ROOT / "detection" / "dataset_final_v4"

# V4 already uses these final class IDs.
CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Battery",
    "E-Waste",
    "Cloth/Textile",
    "Other Waste",
]

RANDOM_SEED = 42


def copy_split(split_name, source_split):
    source_images = V4_ROOT / "split" / source_split / "images"
    source_labels = V4_ROOT / "split" / source_split / "labels"

    output_images = OUTPUT_ROOT / "images" / split_name
    output_labels = OUTPUT_ROOT / "labels" / split_name

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(source_images.glob("*"))

    copied = 0
    skipped = 0

    for image_path in images:
        if not image_path.is_file():
            continue

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        shutil.copy2(
            label_path,
            output_labels / label_path.name
        )

        copied += 1

    print(
        f"{split_name.upper():5s} | "
        f"copied: {copied:5d} | "
        f"skipped: {skipped:5d}"
    )

    return copied


def create_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}

names:
"""

    for index, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {index}: {name}\n"

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    print("\nCreated:")
    print(yaml_path)


def analyze_labels():
    print("\n=== FINAL DATASET CLASS COUNTS ===")

    counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for split in ["train", "val"]:
        labels_dir = OUTPUT_ROOT / "labels" / split

        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text(
                encoding="utf-8"
            ).splitlines():

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    continue

                if class_id in counts:
                    counts[class_id] += 1

    total = sum(counts.values())

    print()

    for class_id, name in enumerate(CLASS_NAMES):
        print(
            f"{class_id:2d} | "
            f"{name:17s} | "
            f"{counts[class_id]:5d}"
        )

    print("-" * 45)
    print(f"TOTAL OBJECTS: {total}")


def main():

    print("=" * 70)
    print("SMART WASTE CLASSIFIER - FINAL V4 DATASET BUILDER")
    print("=" * 70)

    print("\nSource:")
    print(V4_ROOT)

    print("\nDestination:")
    print(OUTPUT_ROOT)

    if not V4_ROOT.exists():
        print("\nERROR: V4 source folder was not found.")
        print("Check the V4 extraction path.")
        return

    if OUTPUT_ROOT.exists():
        print("\nWARNING:")
        print("The final V4 dataset already exists.")
        print("Nothing will be deleted or overwritten automatically.")
        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== COPYING DATASET ===")

    train_count = copy_split(
        "train",
        "train"
    )

    val_count = copy_split(
        "val",
        "val"
    )

    create_yaml()

    analyze_labels()

    print("\n=== FINAL DATASET SUMMARY ===")
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")
    print(f"Total images: {train_count + val_count}")

    print("\nClasses:")

    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    print("\nFinal dataset:")
    print(OUTPUT_ROOT)

    print("\nIMPORTANT:")
    print("Existing V3/V4 datasets were NOT modified.")
    print("No model was trained yet.")


if __name__ == "__main__":
    main()