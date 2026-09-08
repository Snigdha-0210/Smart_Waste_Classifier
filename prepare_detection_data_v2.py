import json
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image


# ============================================================
# SMART WASTE DETECTION DATA PREPARATION V2
# TACO -> YOLO
# Stratified split for rare classes
# ============================================================

print("=" * 70)
print("SMART WASTE DETECTION DATA PREPARATION V2")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DETECTION_DIR = PROJECT_DIR / "detection"

RAW_DIR = DETECTION_DIR / "raw_taco"
ANNOTATION_FILE = RAW_DIR / "annotations.json"
IMAGE_DIR = RAW_DIR / "images"

DATASET_DIR = DETECTION_DIR / "dataset"
OUTPUT_IMAGES = DATASET_DIR / "images"
OUTPUT_LABELS = DATASET_DIR / "labels"

DATA_YAML = DETECTION_DIR / "data.yaml"


# ============================================================
# 2. CLASSES
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


print("\nTarget classes:")

for index, name in enumerate(CLASS_NAMES):
    print(f"  {index}: {name}")


# ============================================================
# 3. CHECK FILES
# ============================================================

if not ANNOTATION_FILE.exists():
    print("\nERROR: annotations.json not found:")
    print(ANNOTATION_FILE)
    raise SystemExit(1)

if not IMAGE_DIR.exists():
    print("\nERROR: TACO image directory not found:")
    print(IMAGE_DIR)
    raise SystemExit(1)


# ============================================================
# 4. LOAD TACO
# ============================================================

print("\nReading TACO annotations...")

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as f:
    taco = json.load(f)


images = taco["images"]
annotations = taco["annotations"]
categories = taco["categories"]

print(f"Images       : {len(images)}")
print(f"Annotations  : {len(annotations)}")
print(f"Categories   : {len(categories)}")


# ============================================================
# 5. CATEGORY LOOKUP
# ============================================================

category_names = {
    category["id"]: category["name"]
    for category in categories
}


# ============================================================
# 6. CATEGORY MAPPING
# ============================================================

def map_category(name):

    name_lower = name.lower().strip()


    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "aluminium",
        "aluminum",
        "metal",
        "food can",
        "drink can",
        "aerosol",
        "pop tab",
    ]):
        return "Metal"


    # --------------------------------------------------------
    # GLASS
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "glass",
    ]):
        return "Glass"


    # --------------------------------------------------------
    # CARDBOARD
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "carton",
        "cartboard",
        "corrugated",
        "egg carton",
        "pizza box",
        "toilet tube",
    ]):
        return "Cardboard"


    # --------------------------------------------------------
    # FOOD ORGANICS
    # --------------------------------------------------------

    if "food waste" in name_lower:
        return "Food Organics"


    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "paper",
        "tissue",
        "magazine",
    ]):
        return "Paper"


    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "plastic",
        "plastified",
        "polypropylene",
        "crisp packet",
        "garbage bag",
        "carrier bag",
        "spread tub",
        "tupperware",
        "squeezable tube",
        "styrofoam",
        "foam",
        "rope",
        "string",
    ]):
        return "Plastic"


    return None


# Build mapping

category_mapping = {}

print("\nTACO CATEGORY MAPPING")
print("=" * 70)

for category_id, category_name in category_names.items():

    mapped_class = map_category(category_name)

    if mapped_class is not None:

        category_mapping[category_id] = mapped_class

        print(
            f"{category_id:3d} : "
            f"{category_name:<40} -> "
            f"{mapped_class}"
        )


print(
    f"\nMapped TACO categories: "
    f"{len(category_mapping)}"
)


# ============================================================
# 7. ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = defaultdict(list)

for annotation in annotations:

    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    if category_id not in category_mapping:
        continue

    mapped_class = category_mapping[category_id]

    annotations_by_image[image_id].append(
        (
            annotation,
            mapped_class
        )
    )


# ============================================================
# 8. INDEX IMAGE FILES
# ============================================================

print("\nIndexing TACO images...")

image_lookup = {}

for file_path in IMAGE_DIR.rglob("*"):

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]:
        continue

    relative_path = file_path.relative_to(
        IMAGE_DIR
    ).as_posix()

    image_lookup[relative_path] = file_path


print(
    f"Actual image files found: "
    f"{len(image_lookup)}"
)


# ============================================================
# 9. BUILD USABLE IMAGE LIST
# ============================================================

usable_images = []

object_counts = Counter()

image_class_counts = Counter()


for image_info in images:

    image_id = image_info["id"]

    file_name = image_info["file_name"]
    file_name = file_name.replace("\\", "/")

    if image_id not in annotations_by_image:
        continue

    if file_name not in image_lookup:
        continue

    valid_objects = []

    classes_in_image = set()

    for annotation, mapped_class in annotations_by_image[image_id]:

        bbox = annotation.get("bbox")

        if bbox is None:
            continue

        if len(bbox) != 4:
            continue

        x, y, width, height = bbox

        if width <= 1 or height <= 1:
            continue

        valid_objects.append(
            (
                annotation,
                mapped_class
            )
        )

        classes_in_image.add(mapped_class)

        object_counts[mapped_class] += 1


    if not valid_objects:
        continue


    usable_images.append(
        (
            image_info,
            valid_objects,
            image_lookup[file_name],
            classes_in_image,
        )
    )

    for class_name in classes_in_image:
        image_class_counts[class_name] += 1


print("\nUsable images:", len(usable_images))

print("\nTotal mapped objects:")

for class_name in CLASS_NAMES:

    print(
        f"  {class_name:<20}"
        f"{object_counts[class_name]}"
    )


print("\nImages containing each class:")

for class_name in CLASS_NAMES:

    print(
        f"  {class_name:<20}"
        f"{image_class_counts[class_name]}"
    )


# ============================================================
# 10. STRATIFIED SPLIT
# ============================================================

print("\nCreating stratified train/val/test split...")


random.seed(42)


# Separate images according to rare class presence.

food_images = [
    item
    for item in usable_images
    if "Food Organics" in item[3]
]

other_images = [
    item
    for item in usable_images
    if "Food Organics" not in item[3]
]


random.shuffle(food_images)
random.shuffle(other_images)


# We have only a handful of Food Organics images.
# Force at least one into validation and one into test
# if enough distinct images exist.

food_test_count = 1 if len(food_images) >= 3 else 0
food_val_count = 1 if len(food_images) >= 2 else 0

food_test = food_images[:food_test_count]

food_val = food_images[
    food_test_count:
    food_test_count + food_val_count
]

food_train = food_images[
    food_test_count + food_val_count:
]


# Standard 70/15/15 split for the remaining images.

random.shuffle(other_images)

other_total = len(other_images)

other_train_end = int(other_total * 0.70)

other_val_end = int(other_total * 0.85)

other_train = other_images[:other_train_end]

other_val = other_images[
    other_train_end:other_val_end
]

other_test = other_images[
    other_val_end:
]


train_data = food_train + other_train
val_data = food_val + other_val
test_data = food_test + other_test


random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)


print("\nDataset split:")

print(
    f"Training   : {len(train_data)}"
)

print(
    f"Validation : {len(val_data)}"
)

print(
    f"Test       : {len(test_data)}"
)


# ============================================================
# 11. SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    split_name,
    split_data
):

    counts = Counter()

    images_per_class = Counter()

    for _, objects, _, classes in split_data:

        for _, mapped_class in objects:
            counts[mapped_class] += 1

        for class_name in classes:
            images_per_class[class_name] += 1


    print(f"\n{split_name.upper()}")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<20}"
            f"objects={counts[class_name]:4d} "
            f"images={images_per_class[class_name]:3d}"
        )


print_split_statistics(
    "Train",
    train_data
)

print_split_statistics(
    "Validation",
    val_data
)

print_split_statistics(
    "Test",
    test_data
)


# ============================================================
# 12. RESET OUTPUT DIRECTORIES
# ============================================================

print("\nPreparing output directories...")


for split in [
    "train",
    "val",
    "test",
]:

    image_dir = OUTPUT_IMAGES / split
    label_dir = OUTPUT_LABELS / split

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    for file in image_dir.iterdir():

        if file.is_file():
            file.unlink()


    for file in label_dir.iterdir():

        if file.is_file():
            file.unlink()


# ============================================================
# 13. BBOX CONVERSION
# ============================================================

def convert_bbox_to_yolo(
    bbox,
    image_width,
    image_height,
):

    x, y, width, height = bbox

    center_x = x + width / 2
    center_y = y + height / 2

    center_x /= image_width
    center_y /= image_height

    width /= image_width
    height /= image_height

    center_x = max(
        0.0,
        min(1.0, center_x),
    )

    center_y = max(
        0.0,
        min(1.0, center_y),
    )

    width = max(
        0.0,
        min(1.0, width),
    )

    height = max(
        0.0,
        min(1.0, height),
    )

    return (
        center_x,
        center_y,
        width,
        height,
    )


# ============================================================
# 14. PROCESS SPLIT
# ============================================================

def process_split(
    split_data,
    split_name,
):

    print(
        f"\nProcessing {split_name}..."
    )

    image_output_dir = (
        OUTPUT_IMAGES /
        split_name
    )

    label_output_dir = (
        OUTPUT_LABELS /
        split_name
    )

    created = 0
    skipped = 0
    objects = 0

    for index, (
        image_info,
        image_annotations,
        source_path,
        _,
    ) in enumerate(
        split_data,
        start=1,
    ):

        try:

            with Image.open(source_path) as image:

                image = image.convert("RGB")

                image_width, image_height = image.size

                if (
                    image_width <= 1
                    or
                    image_height <= 1
                ):
                    skipped += 1
                    continue


                image_id = image_info["id"]

                output_name = (
                    f"{image_id:06d}.jpg"
                )

                output_image = (
                    image_output_dir /
                    output_name
                )

                output_label = (
                    label_output_dir /
                    f"{image_id:06d}.txt"
                )


                image.save(
                    output_image,
                    "JPEG",
                    quality=95,
                )


                label_lines = []


                for (
                    annotation,
                    mapped_class,
                ) in image_annotations:

                    bbox = annotation["bbox"]

                    (
                        x_center,
                        y_center,
                        width,
                        height,
                    ) = convert_bbox_to_yolo(
                        bbox,
                        image_width,
                        image_height,
                    )


                    class_id = CLASS_TO_ID[
                        mapped_class
                    ]


                    label_lines.append(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}"
                    )

                    objects += 1


                if not label_lines:

                    output_image.unlink(
                        missing_ok=True
                    )

                    skipped += 1
                    continue


                output_label.write_text(
                    "\n".join(label_lines),
                    encoding="utf-8",
                )

                created += 1


        except Exception as e:

            skipped += 1

            print(
                f"\nWARNING: Could not process:"
            )

            print(source_path)

            print("Reason:", e)


        if (
            index % 100 == 0
            or
            index == len(split_data)
        ):

            print(
                f"  Processed "
                f"{index}/{len(split_data)}"
            )


    print(
        f"  Images created : {created}"
    )

    print(
        f"  Images skipped : {skipped}"
    )

    print(
        f"  Objects        : {objects}"
    )

    return created, objects


# ============================================================
# 15. GENERATE DATASET
# ============================================================

train_created, train_objects = process_split(
    train_data,
    "train",
)

val_created, val_objects = process_split(
    val_data,
    "val",
)

test_created, test_objects = process_split(
    test_data,
    "test",
)


# ============================================================
# 16. DATA.YAML
# ============================================================

yaml_text = f"""path: {DATASET_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: Cardboard
  1: Food Organics
  2: Glass
  3: Metal
  4: Paper
  5: Plastic
"""


DATA_YAML.write_text(
    yaml_text,
    encoding="utf-8",
)


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)


validation_failed = False


for split in [
    "train",
    "val",
    "test",
]:

    image_files = list(
        (
            OUTPUT_IMAGES /
            split
        ).glob("*.jpg")
    )

    label_files = list(
        (
            OUTPUT_LABELS /
            split
        ).glob("*.txt")
    )


    image_stems = {
        file.stem
        for file in image_files
    }

    label_stems = {
        file.stem
        for file in label_files
    }


    missing_labels = (
        image_stems -
        label_stems
    )

    missing_images = (
        label_stems -
        image_stems
    )


    print(f"\n{split.upper()}")

    print(
        f"  Images : {len(image_files)}"
    )

    print(
        f"  Labels : {len(label_files)}"
    )


    if missing_labels:

        print(
            f"  Missing labels: "
            f"{len(missing_labels)}"
        )

        validation_failed = True


    if missing_images:

        print(
            f"  Missing images: "
            f"{len(missing_images)}"
        )

        validation_failed = True


if validation_failed:

    print("\nDATASET VALIDATION FAILED.")

else:

    print("\nDATASET VALIDATION PASSED.")

    print(
        "\nDataset:"
    )

    print(DATASET_DIR)

    print(
        "\nYAML:"
    )

    print(DATA_YAML)


print("\n")
print("=" * 70)
print("PREPARATION COMPLETE")
print("=" * 70)