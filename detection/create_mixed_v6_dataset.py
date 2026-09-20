from pathlib import Path
from collections import defaultdict, Counter
import random
import shutil

# ============================================================
# MIXED V6 DATASET
# V4 SYNTHETIC + CONTROLLED REAL-WORLD V3
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent

V4_ROOT = PROJECT / "detection" / "dataset_final_v4"
V3_ROOT = PROJECT / "detection" / "dataset_v3"
OUT_ROOT = PROJECT / "detection" / "dataset_mixed_v6"

SEED = 42
MAX_V3_IMAGES = 1500

random.seed(SEED)

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


def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def read_label(label_path):
    rows = []

    if not label_path.exists():
        return rows

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) >= 5:
            rows.append(parts)

    return rows


def copy_pair(image, label, dst_img, dst_lbl, prefix):
    new_image = dst_img / f"{prefix}_{image.name}"
    new_label = dst_lbl / f"{prefix}_{image.stem}.txt"

    shutil.copy2(image, new_image)
    shutil.copy2(label, new_label)


def main():

    if OUT_ROOT.exists():
        raise RuntimeError(
            f"{OUT_ROOT} already exists.\n"
            "Delete it manually only if you are certain it is safe."
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy ALL V4 training and validation data
    # --------------------------------------------------------

    print("Copying V4 data...")

    for split in ["train", "val"]:

        src_images = V4_ROOT / "images" / split
        src_labels = V4_ROOT / "labels" / split

        dst_images = OUT_ROOT / "images" / split
        dst_labels = OUT_ROOT / "labels" / split

        for image in get_images(src_images):

            label = src_labels / f"{image.stem}.txt"

            if not label.exists():
                print("WARNING: missing label:", image.name)
                continue

            shutil.copy2(
                image,
                dst_images / image.name
            )

            shutil.copy2(
                label,
                dst_labels / label.name
            )

    # --------------------------------------------------------
    # Read V3 training dataset
    # --------------------------------------------------------

    print("Reading V3 training dataset...")

    v3_images = V3_ROOT / "images" / "train"
    v3_labels = V3_ROOT / "labels" / "train"

    records = []

    for image in get_images(v3_images):

        label = v3_labels / f"{image.stem}.txt"

        if not label.exists():
            continue

        rows = read_label(label)

        if not rows:
            continue

        counts = Counter(
            int(row[0])
            for row in rows
            if 0 <= int(row[0]) <= 5
        )

        if not counts:
            continue

        records.append(
            {
                "image": image,
                "label": label,
                "counts": counts,
                "total": sum(counts.values()),
            }
        )

    print("Usable V3 images:", len(records))

    # --------------------------------------------------------
    # Target object budget
    #
    # We deliberately keep the six real-world classes within
    # a controlled range instead of trying to make them exact.
    # --------------------------------------------------------

    target_per_class = 350

    selected = []
    selected_set = set()

    current = Counter()

    # Randomize candidates first.
    random.shuffle(records)

    # --------------------------------------------------------
    # First pass:
    # Prefer images that contain currently underrepresented
    # classes and have a small number of Food objects.
    # --------------------------------------------------------

    while len(selected) < MAX_V3_IMAGES:

        candidates = [
            r for r in records
            if r["image"] not in selected_set
        ]

        if not candidates:
            break

        # Calculate current six-class object counts.
        deficits = {
            cls: max(0, target_per_class - current[cls])
            for cls in range(6)
        }

        # Choose the class with the largest deficit.
        target_class = max(
            deficits,
            key=deficits.get
        )

        if deficits[target_class] <= 0:
            break

        # Candidates containing that class.
        class_candidates = [
            r for r in candidates
            if r["counts"].get(target_class, 0) > 0
        ]

        if not class_candidates:
            break

        # Score candidates:
        # - favor target class
        # - strongly discourage excessive Food
        # - favor mixed images
        def score(r):

            target_count = r["counts"].get(
                target_class,
                0
            )

            food_count = r["counts"].get(
                1,
                0
            )

            other_classes = len(
                r["counts"]
            )

            return (
                target_count * 10
                + other_classes * 2
                - food_count * 4
                + random.random()
            )

        class_candidates.sort(
            key=score,
            reverse=True
        )

        chosen = class_candidates[0]

        selected.append(chosen)
        selected_set.add(chosen["image"])

        for cls, count in chosen["counts"].items():
            current[cls] += count

    # --------------------------------------------------------
    # If fewer than 1500 selected, fill remaining slots while
    # keeping Food-heavy images discouraged.
    # --------------------------------------------------------

    if len(selected) < MAX_V3_IMAGES:

        remaining = [
            r for r in records
            if r["image"] not in selected_set
        ]

        remaining.sort(
            key=lambda r: (
                r["counts"].get(1, 0),
                -len(r["counts"]),
                r["total"]
            )
        )

        for r in remaining:

            if len(selected) >= MAX_V3_IMAGES:
                break

            selected.append(r)
            selected_set.add(r["image"])

            for cls, count in r["counts"].items():
                current[cls] += count

    # --------------------------------------------------------
    # Shuffle selected set
    # --------------------------------------------------------

    random.shuffle(selected)

    print()
    print("Selected V3 images:", len(selected))
    print()
    print("Selected V3 object counts:")

    for cls in range(6):
        print(
            f"{cls}: {CLASS_NAMES[cls]:18} "
            f"{current[cls]}"
        )

    # --------------------------------------------------------
    # Copy selected V3 images
    # --------------------------------------------------------

    print()
    print("Copying selected V3 images...")

    dst_images = OUT_ROOT / "images" / "train"
    dst_labels = OUT_ROOT / "labels" / "train"

    for record in selected:

        copy_pair(
            record["image"],
            record["label"],
            dst_images,
            dst_labels,
            "v3real"
        )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    yaml_text = f"""path: {OUT_ROOT.as_posix()}

train: images/train
val: images/val

nc: 10

names:
"""

    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"

    (OUT_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    train_images = len(
        get_images(OUT_ROOT / "images" / "train")
    )

    val_images = len(
        get_images(OUT_ROOT / "images" / "val")
    )

    train_labels = len(
        list((OUT_ROOT / "labels" / "train").glob("*.txt"))
    )

    val_labels = len(
        list((OUT_ROOT / "labels" / "val").glob("*.txt"))
    )

    print()
    print("=" * 60)
    print("MIXED V6 DATASET CREATED")
    print("=" * 60)

    print("Output:", OUT_ROOT)
    print("Train images:", train_images)
    print("Train labels:", train_labels)
    print("Validation images:", val_images)
    print("Validation labels:", val_labels)
    print("Classes:", len(CLASS_NAMES))

    print("=" * 60)


if __name__ == "__main__":
    main()