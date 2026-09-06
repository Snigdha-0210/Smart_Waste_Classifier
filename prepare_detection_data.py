import json
import random
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# SMART WASTE DETECTION DATA PREPARATION
# TACO -> YOLO
# ============================================================

print("=" * 70)
print("SMART WASTE DETECTION DATA PREPARATION")
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
# 2. TARGET CLASSES
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
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

    print("\nERROR:")
    print("Annotation file not found:")
    print(ANNOTATION_FILE)

    raise SystemExit(1)


if not IMAGE_DIR.exists():

    print("\nERROR:")
    print("TACO image directory not found:")
    print(IMAGE_DIR)

    raise SystemExit(1)


print("\nAnnotation file found:")
print(ANNOTATION_FILE)

print("\nImage directory found:")
print(IMAGE_DIR)


# ============================================================
# 4. LOAD ANNOTATIONS
# ============================================================

print("\nReading annotations...")

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    taco = json.load(f)


images = taco["images"]
annotations = taco["annotations"]
categories = taco["categories"]


print(f"Images in TACO annotations: {len(images)}")
print(f"Annotations: {len(annotations)}")
print(f"Categories: {len(categories)}")


# ============================================================
# 5. CATEGORY INFORMATION
# ============================================================

category_names = {}

for category in categories:

    category_id = category["id"]

    category_name = category["name"]

    category_names[category_id] = category_name


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
        "pop tab"
    ]):

        return "Metal"


    # --------------------------------------------------------
    # GLASS
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "glass",
        "glass bottle",
        "glass jar",
        "broken glass"
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
        "toilet tube"
    ]):

        return "Cardboard"


    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "food waste"
    ]):

        return "Food Organics"


    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if any(x in name_lower for x in [
        "paper",
        "tissue",
        "magazine",
        "normal paper",
        "wrapping paper",
        "paper bag",
        "paper cup",
        "paper straw"
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
        "string"
    ]):

        return "Plastic"


    return None


# ============================================================
# 7. BUILD CATEGORY MAPPING
# ============================================================

print("\n")
print("=" * 70)
print("TACO CATEGORY MAPPING")
print("=" * 70)


category_mapping = {}

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
# 8. CREATE ANNOTATION LOOKUP
# ============================================================

annotations_by_image = {}

for annotation in annotations:

    image_id = annotation["image_id"]

    category_id = annotation["category_id"]

    if category_id not in category_mapping:

        continue

    mapped_class = category_mapping[category_id]

    if image_id not in annotations_by_image:

        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(
        (
            annotation,
            mapped_class
        )
    )


# ============================================================
# 9. BUILD IMAGE LOOKUP
# ============================================================

print("\n")
print("=" * 70)
print("INDEXING TACO IMAGE FILES")
print("=" * 70)

print("\nSearching recursively inside:")

print(IMAGE_DIR)


# This is the important fix.
# TACO stores images inside batch_1, batch_2, etc.

all_image_files = list(
    IMAGE_DIR.rglob("*")
)


image_lookup = {}

for file_path in all_image_files:

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:
        continue

    relative_path = file_path.relative_to(
        IMAGE_DIR
    ).as_posix()

    image_lookup[relative_path] = file_path


print(
    f"\nActual image files found: "
    f"{len(image_lookup)}"
)


if len(image_lookup) == 0:

    print("\nERROR:")
    print("No image files were found.")

    raise SystemExit(1)


# ============================================================
# 10. FIND USABLE IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("FINDING USABLE ANNOTATED IMAGES")
print("=" * 70)


usable_images = []

object_counts = {
    class_name: 0
    for class_name in CLASS_NAMES
}


for image_info in images:

    image_id = image_info["id"]

    file_name = image_info["file_name"]

    # Normalize Windows/Linux separators
    file_name = file_name.replace("\\", "/")


    if image_id not in annotations_by_image:

        continue


    if file_name not in image_lookup:

        continue


    mapped_annotations = annotations_by_image[
        image_id
    ]


    valid_objects = []

    for annotation, mapped_class in mapped_annotations:

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


    if len(valid_objects) == 0:

        continue


    usable_images.append(
        (
            image_info,
            valid_objects,
            image_lookup[file_name]
        )
    )


    for _, mapped_class in valid_objects:

        object_counts[mapped_class] += 1


# ============================================================
# 11. PRINT DATA STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("USABLE DATA")
print("=" * 70)


print(
    f"\nImages with mapped objects: "
    f"{len(usable_images)}"
)


print("\nMapped object counts:")

for class_name in CLASS_NAMES:

    print(
        f"  {class_name:<20} "
        f"{object_counts[class_name]}"
    )


if len(usable_images) < 100:

    print("\nERROR:")

    print(
        "Very few usable images were found."
    )

    raise SystemExit(1)


# ============================================================
# 12. VERIFY IMAGE ACCESS
# ============================================================

print("\nChecking image files...")


check_count = min(
    100,
    len(usable_images)
)


located = 0


for _, _, image_path in usable_images[:check_count]:

    if image_path.exists():

        located += 1


print(
    f"Images successfully located: "
    f"{located}/{check_count}"
)


if located != check_count:

    print("\nERROR:")

    print(
        "Some TACO images could not be located."
    )

    raise SystemExit(1)


# ============================================================
# 13. RESET DATASET
# ============================================================

print("\nPreparing YOLO dataset directories...")


for split in [
    "train",
    "val",
    "test"
]:

    image_dir = OUTPUT_IMAGES / split

    label_dir = OUTPUT_LABELS / split


    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Remove old generated files

    for file in image_dir.iterdir():

        if file.is_file():

            file.unlink()


    for file in label_dir.iterdir():

        if file.is_file():

            file.unlink()


# ============================================================
# 14. SHUFFLE DATA
# ============================================================

random.seed(42)

random.shuffle(
    usable_images
)


# ============================================================
# 15. SPLIT DATA
# ============================================================

total = len(usable_images)

train_end = int(
    total * 0.70
)

val_end = int(
    total * 0.85
)


train_data = usable_images[
    :train_end
]

val_data = usable_images[
    train_end:val_end
]

test_data = usable_images[
    val_end:
]


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
# 16. CONVERT BBOX TO YOLO
# ============================================================

def convert_bbox_to_yolo(
    bbox,
    image_width,
    image_height
):

    x, y, width, height = bbox


    center_x = x + width / 2

    center_y = y + height / 2


    center_x /= image_width

    center_y /= image_height

    width /= image_width

    height /= image_height


    # Clamp values

    center_x = max(
        0.0,
        min(1.0, center_x)
    )

    center_y = max(
        0.0,
        min(1.0, center_y)
    )

    width = max(
        0.0,
        min(1.0, width)
    )

    height = max(
        0.0,
        min(1.0, height)
    )


    return (
        center_x,
        center_y,
        width,
        height
    )


# ============================================================
# 17. PROCESS SPLIT
# ============================================================

def process_split(
    data,
    split_name
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
        source_path
    ) in enumerate(data, start=1):


        try:

            # ------------------------------------------------
            # Open image
            # ------------------------------------------------

            with Image.open(source_path) as image:

                image = image.convert("RGB")

                image_width, image_height = image.size


                if image_width <= 1 or image_height <= 1:

                    skipped += 1

                    continue


                # ------------------------------------------------
                # New filename
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Save image
                # ------------------------------------------------

                image.save(
                    output_image,
                    "JPEG",
                    quality=95
                )


                # ------------------------------------------------
                # Create YOLO labels
                # ------------------------------------------------

                label_lines = []


                for annotation, mapped_class in image_annotations:

                    bbox = annotation["bbox"]


                    x_center, y_center, w, h = (
                        convert_bbox_to_yolo(
                            bbox,
                            image_width,
                            image_height
                        )
                    )


                    class_id = CLASS_TO_ID[
                        mapped_class
                    ]


                    label_lines.append(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f}"
                    )


                    objects += 1


                if len(label_lines) == 0:

                    output_image.unlink(
                        missing_ok=True
                    )

                    skipped += 1

                    continue


                output_label.write_text(
                    "\n".join(label_lines),
                    encoding="utf-8"
                )


                created += 1


        except Exception as e:

            skipped += 1

            print(
                f"\nWARNING: "
                f"Could not process "
                f"{source_path}"
            )

            print(
                f"Reason: {e}"
            )


        if (
            index % 100 == 0
            or index == len(data)
        ):

            print(
                f"  Processed "
                f"{index}/{len(data)}"
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
# 18. PROCESS DATASET
# ============================================================

train_created, train_objects = process_split(
    train_data,
    "train"
)


val_created, val_objects = process_split(
    val_data,
    "val"
)


test_created, test_objects = process_split(
    test_data,
    "test"
)


# ============================================================
# 19. CREATE DATA.YAML
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
    encoding="utf-8"
)


print("\n")
print("=" * 70)
print("FINAL YOLO DATASET")
print("=" * 70)


print(
    f"Train images : {train_created}"
)

print(
    f"Val images   : {val_created}"
)

print(
    f"Test images  : {test_created}"
)

print(
    f"Train objects: {train_objects}"
)

print(
    f"Val objects  : {val_objects}"
)

print(
    f"Test objects : {test_objects}"
)


print("\ndata.yaml created:")

print(DATA_YAML)


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATING DATASET")
print("=" * 70)


total_images = (
    train_created +
    val_created +
    test_created
)


total_objects = (
    train_objects +
    val_objects +
    test_objects
)


print(
    f"Total images : {total_images}"
)

print(
    f"Total objects: {total_objects}"
)


# Check that images and labels match

validation_failed = False


for split in [
    "train",
    "val",
    "test"
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
        x.stem
        for x in image_files
    }

    label_stems = {
        x.stem
        for x in label_files
    }


    missing_labels = (
        image_stems -
        label_stems
    )


    missing_images = (
        label_stems -
        image_stems
    )


    print(
        f"\n{split.upper()}"
    )

    print(
        f"  Images: {len(image_files)}"
    )

    print(
        f"  Labels: {len(label_files)}"
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


# ============================================================
# 21. FINISH
# ============================================================

if validation_failed:

    print("\n")
    print("=" * 70)
    print("WARNING: DATASET VALIDATION FOUND PROBLEMS")
    print("=" * 70)

else:

    print("\n")
    print("=" * 70)
    print("DETECTION DATA PREPARATION COMPLETE")
    print("=" * 70)


    print("\nDataset:")

    print(DATASET_DIR)


    print("\nConfiguration:")

    print(DATA_YAML)


    print("\nNext step:")

    print(
        "Train YOLO using the generated data.yaml"
    )


print("=" * 70)