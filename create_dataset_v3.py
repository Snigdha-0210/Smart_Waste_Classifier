from pathlib import Path
import shutil
import random
from collections import Counter


# ============================================================
# SMART WASTE CLASSIFIER
# V3 DATASET BUILDER
# ============================================================

random.seed(42)

BASE = Path(r"detection")

TACO = BASE / "dataset_taco_original"

SUPP = (
    BASE
    / "raw_supplemental"
    / "Trash Detection Dataset"
    / "CUSTOM_DATASET"
)

OUTPUT = BASE / "dataset_v3"


CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]


# ============================================================
# SUPPLEMENTAL CLASS MAPPING
# ============================================================
#
# Supplemental:
#   0 = BIODEGRADABLE
#   1 = CARDBOARD
#   2 = GLASS
#   3 = METAL
#   4 = PAPER
#   5 = PLASTIC
#
# Our dataset:
#   0 = Cardboard
#   1 = Food Organics
#   2 = Glass
#   3 = Metal
#   4 = Paper
#   5 = Plastic
#
# Therefore:
#   0 -> 1
#   1 -> 0
#   2 -> 2
#   3 -> 3
#   4 -> 4
#   5 -> 5
# ============================================================

SUPPLEMENTAL_CLASS_MAP = {
    0: 1,
    1: 0,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}


# ============================================================
# SETTINGS
# ============================================================

TARGET_SINGLE_CLASS_IMAGES = {
    0: 500,   # Cardboard
    1: 450,   # Food Organics
    2: 500,   # Glass
    3: 650,   # Metal
    4: 500,   # Paper
    5: 550    # Plastic
}

FOOD_OBJECT_LIMIT = 20
DENSE_FOOD_KEEP_RATIO = 0.35


# ============================================================
# HELPERS
# ============================================================

def get_image_for_label(label_path, image_dir):

    stem = label_path.stem

    candidates = list(image_dir.glob(stem + ".*"))

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    for candidate in candidates:

        if candidate.is_file() and candidate.suffix.lower() in valid_extensions:
            return candidate

    return None


def read_original_classes(label_path):

    classes = set()

    try:

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 1:
                continue

            class_id = int(parts[0])

            if 0 <= class_id < 6:
                classes.add(class_id)

    except Exception:

        return set()

    return classes


def count_food_objects(label_path):

    count = 0

    try:

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 1:
                continue

            class_id = int(parts[0])

            # Supplemental class 0 = BIODEGRADABLE
            if class_id == 0:
                count += 1

    except Exception:

        return 0

    return count


def copy_taco_pair(
    image_path,
    label_path,
    out_image_dir,
    out_label_dir
):

    out_image = out_image_dir / image_path.name
    out_label = out_label_dir / label_path.name

    if out_image.exists() or out_label.exists():
        return False

    shutil.copy2(image_path, out_image)

    # TACO labels are already in our six-class format.
    shutil.copy2(label_path, out_label)

    return True


def copy_supplemental_pair(
    image_path,
    label_path,
    out_image_dir,
    out_label_dir
):

    out_image = out_image_dir / image_path.name
    out_label = out_label_dir / label_path.name

    if out_image.exists() or out_label.exists():
        return False

    # --------------------------------------------------------
    # Read and REMAP supplemental labels
    # --------------------------------------------------------

    converted_lines = []

    try:

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            original_class = int(parts[0])

            if original_class not in SUPPLEMENTAL_CLASS_MAP:
                continue

            new_class = SUPPLEMENTAL_CLASS_MAP[original_class]

            converted_line = " ".join(
                [str(new_class)] + parts[1:]
            )

            converted_lines.append(converted_line)

    except Exception:

        return False

    if not converted_lines:
        return False

    shutil.copy2(image_path, out_image)

    out_label.write_text(
        "\n".join(converted_lines) + "\n"
    )

    return True


def process_split(
    taco_split,
    supp_split,
    output_split,
    training=False
):

    out_images = OUTPUT / "images" / output_split
    out_labels = OUTPUT / "labels" / output_split

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    taco_images = TACO / "images" / taco_split
    taco_labels = TACO / "labels" / taco_split

    if supp_split == "valid":
        supp_root = SUPP / "valid"
    else:
        supp_root = SUPP / supp_split

    supp_images = supp_root / "images"
    supp_labels = supp_root / "labels"

    copied_taco = 0
    copied_supp = 0
    skipped = 0

    # ========================================================
    # TACO
    # ========================================================

    for label in taco_labels.glob("*.txt"):

        image = get_image_for_label(label, taco_images)

        if image is None:
            skipped += 1
            continue

        if copy_taco_pair(
            image,
            label,
            out_images,
            out_labels
        ):
            copied_taco += 1

    # ========================================================
    # VALIDATION / TEST
    # ========================================================

    if not training:

        for label in supp_labels.glob("*.txt"):

            image = get_image_for_label(
                label,
                supp_images
            )

            if image is None:
                skipped += 1
                continue

            if copy_supplemental_pair(
                image,
                label,
                out_images,
                out_labels
            ):
                copied_supp += 1

        return copied_taco, copied_supp, skipped

    # ========================================================
    # TRAINING
    # ========================================================

    single_class = {
        i: []
        for i in range(6)
    }

    multi_class = []

    # --------------------------------------------------------
    # Analyze supplemental training images
    # --------------------------------------------------------

    for label in supp_labels.glob("*.txt"):

        image = get_image_for_label(
            label,
            supp_images
        )

        if image is None:
            skipped += 1
            continue

        classes = read_original_classes(label)

        if not classes:
            skipped += 1
            continue

        # Multi-class image
        if len(classes) >= 2:

            multi_class.append(
                (image, label, classes)
            )

        # Single-class image
        else:

            class_id = next(iter(classes))

            # Convert supplemental class ID to our class ID
            class_id = SUPPLEMENTAL_CLASS_MAP[class_id]

            single_class[class_id].append(
                (image, label)
            )

    # ========================================================
    # KEEP ALL MULTI-CLASS SUPPLEMENTAL IMAGES
    # ========================================================

    for image, label, classes in multi_class:

        if copy_supplemental_pair(
            image,
            label,
            out_images,
            out_labels
        ):
            copied_supp += 1

    # ========================================================
    # SAMPLE SINGLE-CLASS SUPPLEMENTAL IMAGES
    # ========================================================

    for class_id in range(6):

        candidates = single_class[class_id]

        random.shuffle(candidates)

        target = min(
            TARGET_SINGLE_CLASS_IMAGES[class_id],
            len(candidates)
        )

        selected = candidates[:target]

        for image, label in selected:

            if copy_supplemental_pair(
                image,
                label,
                out_images,
                out_labels
            ):
                copied_supp += 1

    # ========================================================
    # DENSE FOOD ORGANICS
    # ========================================================

    dense_food = []

    for label in supp_labels.glob("*.txt"):

        image = get_image_for_label(
            label,
            supp_images
        )

        if image is None:
            continue

        classes = read_original_classes(label)

        # Supplemental {0} = BIODEGRADABLE
        if classes != {0}:
            continue

        food_count = count_food_objects(label)

        if food_count > FOOD_OBJECT_LIMIT:

            dense_food.append(
                (image, label, food_count)
            )

    random.shuffle(dense_food)

    keep_count = int(
        len(dense_food) * DENSE_FOOD_KEEP_RATIO
    )

    for image, label, food_count in dense_food[:keep_count]:

        if copy_supplemental_pair(
            image,
            label,
            out_images,
            out_labels
        ):
            copied_supp += 1

    return copied_taco, copied_supp, skipped


# ============================================================
# START
# ============================================================

print("=" * 75)
print("SMART WASTE CLASSIFIER - BUILDING DATASET V3")
print("=" * 75)

print()
print("Output:")
print(OUTPUT)


# ============================================================
# TRAIN
# ============================================================

print()
print("TRAIN")
print("-" * 50)

taco, supp, skipped = process_split(
    "train",
    "train",
    "train",
    training=True
)

print("TACO copied       :", taco)
print("Supplemental      :", supp)
print("Skipped           :", skipped)


# ============================================================
# VALIDATION
# ============================================================

print()
print("VALIDATION")
print("-" * 50)

taco, supp, skipped = process_split(
    "val",
    "valid",
    "val",
    training=False
)

print("TACO copied       :", taco)
print("Supplemental      :", supp)
print("Skipped           :", skipped)


# ============================================================
# TEST
# ============================================================

print()
print("TEST")
print("-" * 50)

taco, supp, skipped = process_split(
    "test",
    "test",
    "test",
    training=False
)

print("TACO copied       :", taco)
print("Supplemental      :", supp)
print("Skipped           :", skipped)


# ============================================================
# YAML
# ============================================================

yaml_path = OUTPUT / "data.yaml"

yaml_text = """path: C:/Users/misty/OneDrive/Documents/Smart_Waste_Classifier/detection/dataset_v3

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

yaml_path.write_text(yaml_text)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("V3 DATASET VALIDATION")
print("=" * 75)

total_images = 0
total_labels = 0

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}

for split in ["train", "val", "test"]:

    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split

    images = [
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in valid_extensions
    ]

    labels = list(
        label_dir.glob("*.txt")
    )

    print()
    print(split.upper())
    print("Images:", len(images))
    print("Labels:", len(labels))

    if len(images) != len(labels):

        print(
            "WARNING: image/label count mismatch"
        )

    else:

        print("MATCHED")

    total_images += len(images)
    total_labels += len(labels)


print()
print("TOTAL IMAGES:", total_images)
print("TOTAL LABELS:", total_labels)


if total_images == total_labels:

    print()
    print("DATASET V3 VALIDATION PASSED.")

else:

    print()
    print("DATASET V3 VALIDATION FAILED.")


print()
print("YAML:", yaml_path)

print()
print("=" * 75)
print("V3 DATASET CREATION COMPLETE")
print("=" * 75)