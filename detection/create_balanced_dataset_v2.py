from pathlib import Path
import shutil
import random
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path("detection/dataset_combined")
OUTPUT = Path("detection/dataset_v2")

SEED = 42
random.seed(SEED)

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
]

# Maximum Food Organics objects allowed in a single training image.
# Very dense organic images are the main source of imbalance.
MAX_FOOD_OBJECTS_PER_IMAGE = 20

# We will retain all non-organic training images.
# Food Organics images above the threshold are sampled down.
#
# Food images <= 20 objects:
#   KEEP ALL
#
# Food images > 20 objects:
#   KEEP a controlled fraction.
DENSE_FOOD_KEEP_FRACTION = 0.25


# ============================================================
# PATHS
# ============================================================

SOURCE_IMAGES = SOURCE / "images"
SOURCE_LABELS = SOURCE / "labels"

OUTPUT_IMAGES = OUTPUT / "images"
OUTPUT_LABELS = OUTPUT / "labels"


# ============================================================
# CLEAN OUTPUT
# ============================================================

if OUTPUT.exists():
    print("Removing previous dataset_v2...")
    shutil.rmtree(OUTPUT)


for split in ["train", "val", "test"]:
    (OUTPUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_LABELS / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# ANALYZE A LABEL FILE
# ============================================================

def read_labels(label_path):
    """
    Returns:
        list of label lines
        Counter of class IDs
    """

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], Counter()

    valid_lines = []
    counts = Counter()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if 0 <= class_id < len(CLASS_NAMES):
            valid_lines.append(line)
            counts[class_id] += 1

    return valid_lines, counts


# ============================================================
# COPY ONE IMAGE + LABEL
# ============================================================

def copy_pair(split, image_path, label_path, new_name=None):

    if new_name is None:
        new_name = image_path.name

    destination_image = OUTPUT_IMAGES / split / new_name
    destination_label = OUTPUT_LABELS / split / (
        Path(new_name).stem + ".txt"
    )

    shutil.copy2(image_path, destination_image)
    shutil.copy2(label_path, destination_label)


# ============================================================
# FIND IMAGE FOR LABEL
# ============================================================

def find_image(image_dir, label_path):

    stem = label_path.stem

    extensions = [
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
        ".webp",
        ".WEBP",
    ]

    for ext in extensions:
        candidate = image_dir / (stem + ext)

        if candidate.exists():
            return candidate

    return None


# ============================================================
# PROCESS TRAINING SET
# ============================================================

print("=" * 70)
print("CREATING BALANCED YOLO DATASET V2")
print("=" * 70)

train_label_dir = SOURCE_LABELS / "train"
train_image_dir = SOURCE_IMAGES / "train"

train_labels = sorted(train_label_dir.glob("*.txt"))

kept = []
skipped = []

for label_path in train_labels:

    image_path = find_image(train_image_dir, label_path)

    if image_path is None:
        skipped.append(label_path.name)
        continue

    lines, counts = read_labels(label_path)

    if not lines:
        skipped.append(label_path.name)
        continue

    food_count = counts[1]

    # --------------------------------------------------------
    # NON-FOOD IMAGES
    # --------------------------------------------------------

    if food_count == 0:
        kept.append(
            (
                image_path,
                label_path,
                "non_food"
            )
        )
        continue

    # --------------------------------------------------------
    # MODERATE FOOD IMAGES
    # --------------------------------------------------------

    if food_count <= MAX_FOOD_OBJECTS_PER_IMAGE:
        kept.append(
            (
                image_path,
                label_path,
                "food_moderate"
            )
        )
        continue

    # --------------------------------------------------------
    # DENSE FOOD IMAGES
    # --------------------------------------------------------

    # Randomly retain only a fraction of these.
    if random.random() < DENSE_FOOD_KEEP_FRACTION:
        kept.append(
            (
                image_path,
                label_path,
                "food_dense_kept"
            )
        )
    else:
        skipped.append(label_path.name)


# ============================================================
# COPY SELECTED TRAINING IMAGES
# ============================================================

for image_path, label_path, reason in kept:
    copy_pair(
        "train",
        image_path,
        label_path
    )


# ============================================================
# COPY VALIDATION + TEST UNCHANGED
# ============================================================

for split in ["val", "test"]:

    image_dir = SOURCE_IMAGES / split
    label_dir = SOURCE_LABELS / split

    label_files = sorted(label_dir.glob("*.txt"))

    for label_path in label_files:

        image_path = find_image(image_dir, label_path)

        if image_path is None:
            continue

        copy_pair(
            split,
            image_path,
            label_path
        )


# ============================================================
# STATISTICS
# ============================================================

def analyze_split(split):

    label_dir = OUTPUT_LABELS / split

    object_counts = Counter()
    image_counts = Counter()

    label_files = list(label_dir.glob("*.txt"))

    for label_path in label_files:

        lines, counts = read_labels(label_path)

        for class_id, count in counts.items():
            object_counts[class_id] += count

        for class_id in counts:
            image_counts[class_id] += 1

    return label_files, object_counts, image_counts


print()
print("=" * 70)
print("V2 DATASET STATISTICS")
print("=" * 70)

for split in ["train", "val", "test"]:

    label_files, object_counts, image_counts = analyze_split(split)

    total_objects = sum(object_counts.values())

    print()
    print(split.upper())
    print("-" * 70)
    print(f"Images: {len(label_files)}")
    print(f"Objects: {total_objects}")

    for class_id, class_name in enumerate(CLASS_NAMES):

        objects = object_counts[class_id]
        images = image_counts[class_id]

        percentage = (
            objects / total_objects * 100
            if total_objects
            else 0
        )

        print(
            f"{class_name:<18} : "
            f"{objects:>6} objects | "
            f"{images:>5} images | "
            f"{percentage:>6.2f}%"
        )


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

validation_passed = True

for split in ["train", "val", "test"]:

    image_dir = OUTPUT_IMAGES / split
    label_dir = OUTPUT_LABELS / split

    images = []

    for ext in [
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.png",
        "*.PNG",
        "*.webp",
        "*.WEBP",
    ]:
        images.extend(image_dir.glob(ext))

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem.lower() for p in images}
    label_stems = {p.stem.lower() for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print()
    print(
        f"{split.upper():5} | "
        f"images={len(images)} | "
        f"labels={len(labels)} | "
        f"missing_labels={len(missing_labels)} | "
        f"missing_images={len(missing_images)}"
    )

    if missing_labels or missing_images:
        validation_passed = False


# ============================================================
# CREATE YAML
# ============================================================

yaml_content = """path: .

train: images/train
val: images/val
test: images/test

nc: 6

names:
  0: Cardboard
  1: Food Organics
  2: Glass
  3: Metal
  4: Paper
  5: Plastic
"""

yaml_path = OUTPUT / "data.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")


# ============================================================
# FINAL REPORT
# ============================================================

print()

if validation_passed:
    print("=" * 70)
    print("DATASET V2 VALIDATION PASSED")
    print("=" * 70)
else:
    print("=" * 70)
    print("WARNING: DATASET V2 VALIDATION FAILED")
    print("=" * 70)

print()
print(f"Output dataset:")
print(OUTPUT.resolve())

print()
print(f"Training images kept : {len(kept)}")
print(f"Training images removed: {len(skipped)}")

print()
print("Food Organics policy:")
print(f"  Images <= {MAX_FOOD_OBJECTS_PER_IMAGE} objects: KEEP")
print(
    f"  Images > {MAX_FOOD_OBJECTS_PER_IMAGE} objects: "
    f"KEEP {DENSE_FOOD_KEEP_FRACTION * 100:.0f}% randomly"
)

print()
print("Original dataset was NOT modified.")
print("V1 remains available for comparison.")

print()
print("=" * 70)
print("V2 DATASET CREATION COMPLETE")
print("=" * 70)