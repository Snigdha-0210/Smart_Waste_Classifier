from pathlib import Path
import shutil
import yaml


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETECTION_DIR = PROJECT_ROOT / "detection"

# TACO prepared dataset
TACO_ROOT = DETECTION_DIR / "dataset"

# Supplemental dataset
SUPPLEMENTAL_ROOT = (
    DETECTION_DIR
    / "raw_supplemental"
    / "Trash Detection Dataset"
    / "CUSTOM_DATASET"
)

# Final combined dataset
OUTPUT_ROOT = DETECTION_DIR / "dataset_combined"


# ============================================================
# TARGET CLASSES
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
]

NUM_CLASSES = len(CLASS_NAMES)


# Supplemental dataset mapping:
#
# Supplemental:
# 0 BIODEGRADABLE
# 1 CARDBOARD
# 2 GLASS
# 3 METAL
# 4 PAPER
# 5 PLASTIC
#
# Combined:
# 0 Cardboard
# 1 Food Organics
# 2 Glass
# 3 Metal
# 4 Paper
# 5 Plastic

SUPPLEMENTAL_CLASS_MAP = {
    0: 1,  # BIODEGRADABLE -> Food Organics
    1: 0,  # CARDBOARD -> Cardboard
    2: 2,  # GLASS -> Glass
    3: 3,  # METAL -> Metal
    4: 4,  # PAPER -> Paper
    5: 5,  # PLASTIC -> Plastic
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(image_dir):
    """Return all image files in a directory."""
    if not image_dir.exists():
        return []

    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label_for_image(image_path, label_dir):
    """Find label file corresponding to an image."""
    label_path = label_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        return label_path

    return None


def validate_yolo_label(label_path):
    """
    Validate YOLO annotation file.

    Expected format:
    class_id x_center y_center width height
    """

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        return False, 0

    object_count = 0

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Empty annotation files are allowed.
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            return False, object_count

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            return False, object_count

        # Class must exist.
        if class_id < 0 or class_id >= NUM_CLASSES:
            return False, object_count

        # Coordinates must be normalized.
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            return False, object_count

        object_count += 1

    return True, object_count


def remap_label_file(
    source_label,
    destination_label,
    class_map
):
    """
    Copy a YOLO label while remapping class IDs.
    """

    try:
        lines = source_label.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        return False, 0

    new_lines = []
    object_count = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            return False, object_count

        try:
            old_class = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            return False, object_count

        if old_class not in class_map:
            return False, object_count

        new_class = class_map[old_class]

        if new_class < 0 or new_class >= NUM_CLASSES:
            return False, object_count

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            return False, object_count

        new_lines.append(
            f"{new_class} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        object_count += 1

    destination_label.write_text(
        "\n".join(new_lines) + ("\n" if new_lines else ""),
        encoding="utf-8"
    )

    return True, object_count


def prepare_output_directories():
    """Create clean output dataset directories."""

    if OUTPUT_ROOT.exists():
        print()
        print("Removing previous combined dataset...")
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:

        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS TACO
# ============================================================

def process_taco_split(split):
    """
    Copy TACO data.

    IMPORTANT:
    TACO structure is:

    dataset/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
    """

    source_images = TACO_ROOT / "images" / split
    source_labels = TACO_ROOT / "labels" / split

    destination_images = OUTPUT_ROOT / "images" / split
    destination_labels = OUTPUT_ROOT / "labels" / split

    images = get_image_files(source_images)

    copied = 0
    objects = 0
    skipped = 0

    for image_path in images:

        label_path = find_label_for_image(
            image_path,
            source_labels
        )

        if label_path is None:
            skipped += 1
            continue

        valid, object_count = validate_yolo_label(
            label_path
        )

        if not valid:
            skipped += 1
            continue

        # Prefix prevents filename collisions.
        destination_image = (
            destination_images
            / f"taco_{image_path.name}"
        )

        destination_label = (
            destination_labels
            / f"taco_{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        copied += 1
        objects += object_count

    return copied, objects, skipped


# ============================================================
# PROCESS SUPPLEMENTAL DATASET
# ============================================================

def process_supplemental_split(
    source_split,
    destination_split
):
    """
    Copy supplemental dataset.

    source structure:

    CUSTOM_DATASET/
        train/
        valid/
        test/

    Each split contains:

        images/
        labels/
    """

    source_images = (
        SUPPLEMENTAL_ROOT
        / source_split
        / "images"
    )

    source_labels = (
        SUPPLEMENTAL_ROOT
        / source_split
        / "labels"
    )

    destination_images = (
        OUTPUT_ROOT
        / "images"
        / destination_split
    )

    destination_labels = (
        OUTPUT_ROOT
        / "labels"
        / destination_split
    )

    images = get_image_files(source_images)

    copied = 0
    objects = 0
    skipped = 0

    for image_path in images:

        label_path = find_label_for_image(
            image_path,
            source_labels
        )

        # Supplemental dataset has some unmatched files.
        # We simply skip them.
        if label_path is None:
            skipped += 1
            continue

        destination_image = (
            destination_images
            / f"supp_{image_path.name}"
        )

        destination_label = (
            destination_labels
            / f"supp_{image_path.stem}.txt"
        )

        valid, object_count = remap_label_file(
            label_path,
            destination_label,
            SUPPLEMENTAL_CLASS_MAP
        )

        if not valid:
            if destination_label.exists():
                destination_label.unlink()

            skipped += 1
            continue

        shutil.copy2(
            image_path,
            destination_image
        )

        copied += 1
        objects += object_count

    return copied, objects, skipped


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_final_dataset():

    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    total_images = 0
    total_labels = 0
    total_objects = 0

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    all_valid = True

    for split in ["train", "val", "test"]:

        image_dir = OUTPUT_ROOT / "images" / split
        label_dir = OUTPUT_ROOT / "labels" / split

        images = get_image_files(image_dir)

        labels = sorted(label_dir.glob("*.txt"))

        image_stems = {
            image.stem
            for image in images
        }

        label_stems = {
            label.stem
            for label in labels
        }

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            print(
                f"{split.upper()}: "
                f"{len(missing_labels)} images without labels"
            )
            all_valid = False

        if missing_images:
            print(
                f"{split.upper()}: "
                f"{len(missing_images)} labels without images"
            )
            all_valid = False

        split_objects = 0

        for label in labels:

            valid, object_count = validate_yolo_label(
                label
            )

            if not valid:
                print(
                    f"INVALID LABEL: {label}"
                )
                all_valid = False
                continue

            split_objects += object_count

            try:
                lines = label.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()

                for line in lines:

                    if not line.strip():
                        continue

                    class_id = int(
                        line.split()[0]
                    )

                    class_counts[
                        CLASS_NAMES[class_id]
                    ] += 1

            except Exception:
                all_valid = False

        print(
            f"{split.upper():5s} | "
            f"images: {len(images):4d} | "
            f"labels: {len(labels):4d} | "
            f"objects: {split_objects:5d}"
        )

        total_images += len(images)
        total_labels += len(labels)
        total_objects += split_objects

    print()
    print("CLASS OBJECT COUNTS:")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:16s}: "
            f"{class_counts[class_name]}"
        )

    print()
    print(f"TOTAL IMAGES : {total_images}")
    print(f"TOTAL LABELS : {total_labels}")
    print(f"TOTAL OBJECTS: {total_objects}")

    print()

    if all_valid:
        print("DATASET VALIDATION PASSED.")
    else:
        print("DATASET VALIDATION FAILED.")

    return all_valid


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_yaml():

    yaml_path = OUTPUT_ROOT / "data.yaml"

    data = {
        "path": str(OUTPUT_ROOT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    return yaml_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SMART WASTE DETECTION DATASET MERGER")
    print("=" * 70)

    print()
    print("TACO source:")
    print(TACO_ROOT)

    print()
    print("Supplemental source:")
    print(SUPPLEMENTAL_ROOT)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()
    print("Target classes:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {i}: {class_name}")

    # --------------------------------------------------------
    # Check sources
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING SOURCES")
    print("=" * 70)

    if not TACO_ROOT.exists():
        raise FileNotFoundError(
            f"TACO dataset not found:\n{TACO_ROOT}"
        )

    if not SUPPLEMENTAL_ROOT.exists():
        raise FileNotFoundError(
            f"Supplemental dataset not found:\n"
            f"{SUPPLEMENTAL_ROOT}"
        )

    for split in ["train", "val", "test"]:

        if not (
            TACO_ROOT
            / "images"
            / split
        ).exists():

            raise FileNotFoundError(
                f"TACO images folder missing:\n"
                f"{TACO_ROOT / 'images' / split}"
            )

        if not (
            TACO_ROOT
            / "labels"
            / split
        ).exists():

            raise FileNotFoundError(
                f"TACO labels folder missing:\n"
                f"{TACO_ROOT / 'labels' / split}"
            )

    print("TACO structure: OK")
    print("Supplemental structure: OK")

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    prepare_output_directories()

    # --------------------------------------------------------
    # Process TACO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING TACO")
    print("=" * 70)

    taco_totals = {
        "images": 0,
        "objects": 0,
        "skipped": 0,
    }

    for split in ["train", "val", "test"]:

        copied, objects, skipped = (
            process_taco_split(split)
        )

        taco_totals["images"] += copied
        taco_totals["objects"] += objects
        taco_totals["skipped"] += skipped

        print(
            f"{split.upper():5s}: "
            f"{copied} images | "
            f"{objects} objects | "
            f"{skipped} skipped"
        )

    print()
    print(
        f"TACO TOTAL: "
        f"{taco_totals['images']} images | "
        f"{taco_totals['objects']} objects | "
        f"{taco_totals['skipped']} skipped"
    )

    # --------------------------------------------------------
    # Process Supplemental
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING SUPPLEMENTAL DATASET")
    print("=" * 70)

    supplemental_totals = {
        "images": 0,
        "objects": 0,
        "skipped": 0,
    }

    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    for source_split, destination_split in split_mapping.items():

        copied, objects, skipped = (
            process_supplemental_split(
                source_split,
                destination_split
            )
        )

        supplemental_totals["images"] += copied
        supplemental_totals["objects"] += objects
        supplemental_totals["skipped"] += skipped

        print(
            f"{source_split.upper():5s} -> "
            f"{destination_split.upper():5s}: "
            f"{copied} images | "
            f"{objects} objects | "
            f"{skipped} skipped"
        )

    print()
    print(
        f"SUPPLEMENTAL TOTAL: "
        f"{supplemental_totals['images']} images | "
        f"{supplemental_totals['objects']} objects | "
        f"{supplemental_totals['skipped']} skipped"
    )

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATA.YAML")
    print("=" * 70)

    yaml_path = create_yaml()

    print()
    print(f"Created:")
    print(yaml_path)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    valid = validate_final_dataset()

    print()
    print("=" * 70)

    if valid:

        print("MERGE COMPLETE")
        print("=" * 70)

        print()
        print("Combined dataset:")
        print(OUTPUT_ROOT)

        print()
        print("YAML:")
        print(yaml_path)

        print()
        print("The original datasets were NOT modified.")

    else:

        print("MERGE FAILED VALIDATION")
        print("=" * 70)

        raise SystemExit(1)


if __name__ == "__main__":
    main()