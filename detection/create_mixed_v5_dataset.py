from pathlib import Path
from collections import defaultdict
import random
import shutil

# ============================================================
# MIXED V5 DATASET CREATOR
# V4 SYNTHETIC DATA + CONTROLLED V3 REAL-WORLD DATA
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent

V4_ROOT = PROJECT / "detection" / "dataset_final_v4"
V3_ROOT = PROJECT / "detection" / "dataset_v3"
OUT_ROOT = PROJECT / "detection" / "dataset_mixed_v5"

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


def get_image_files(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def read_classes(label_path):
    classes = set()

    if not label_path.exists():
        return classes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if parts:
            classes.add(int(parts[0]))

    return classes


def copy_pair(src_image, src_label, dst_image_dir, dst_label_dir, prefix):
    dst_image = dst_image_dir / f"{prefix}_{src_image.name}"
    dst_label = dst_label_dir / f"{prefix}_{src_image.stem}.txt"

    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_label, dst_label)


def main():
    if OUT_ROOT.exists():
        raise RuntimeError(
            f"Output already exists:\n{OUT_ROOT}\n"
            "Delete it manually only if you are certain it is safe."
        )

    # Create output directories
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Copy V4 train and validation data unchanged
    # --------------------------------------------------------
    print("Copying V4 training data...")

    for split in ["train", "val"]:
        src_img_dir = V4_ROOT / "images" / split
        src_lbl_dir = V4_ROOT / "labels" / split
        dst_img_dir = OUT_ROOT / "images" / split
        dst_lbl_dir = OUT_ROOT / "labels" / split

        for image_path in get_image_files(src_img_dir):
            label_path = src_lbl_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print("Skipping missing V4 label:", image_path.name)
                continue

            shutil.copy2(image_path, dst_img_dir / image_path.name)
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)

    # --------------------------------------------------------
    # 2. Group V3 training images by contained classes
    # --------------------------------------------------------
    print("Reading V3 training labels...")

    v3_img_dir = V3_ROOT / "images" / "train"
    v3_lbl_dir = V3_ROOT / "labels" / "train"

    class_to_images = defaultdict(list)
    all_v3_pairs = []

    for image_path in get_image_files(v3_img_dir):
        label_path = v3_lbl_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        classes = read_classes(label_path)

        if not classes:
            continue

        pair = (image_path, label_path)
        all_v3_pairs.append(pair)

        for class_id in classes:
            class_to_images[class_id].append(pair)

    # --------------------------------------------------------
    # 3. Balanced selection of V3 images
    # --------------------------------------------------------
    print("Selecting controlled V3 subset...")

    selected = set()
    target_per_class = MAX_V3_IMAGES // 6

    for class_id in range(6):
        candidates = class_to_images[class_id].copy()
        random.shuffle(candidates)

        for pair in candidates:
            if len(selected) >= MAX_V3_IMAGES:
                break

            selected.add(pair)

            if sum(
                1 for p in selected
                if class_id in read_classes(p[1])
            ) >= target_per_class:
                break

    # Fill remaining slots randomly if needed
    remaining = [
        pair for pair in all_v3_pairs
        if pair not in selected
    ]
    random.shuffle(remaining)

    for pair in remaining:
        if len(selected) >= MAX_V3_IMAGES:
            break
        selected.add(pair)

    selected = list(selected)
    random.shuffle(selected)

    print("Selected V3 images:", len(selected))

    # --------------------------------------------------------
    # 4. Copy selected V3 data with unique prefixes
    # --------------------------------------------------------
    print("Copying selected V3 data...")

    dst_img_dir = OUT_ROOT / "images" / "train"
    dst_lbl_dir = OUT_ROOT / "labels" / "train"

    for image_path, label_path in selected:
        copy_pair(
            image_path,
            label_path,
            dst_img_dir,
            dst_lbl_dir,
            "v3real"
        )

    # --------------------------------------------------------
    # 5. Write YAML
    # --------------------------------------------------------
    yaml_text = f"""path: {OUT_ROOT.as_posix()}

train: images/train
val: images/val

nc: {len(CLASS_NAMES)}

names:
"""

    for index, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {index}: {name}\n"

    (OUT_ROOT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------
    train_images = len(get_image_files(OUT_ROOT / "images" / "train"))
    val_images = len(get_image_files(OUT_ROOT / "images" / "val"))

    train_labels = len(list((OUT_ROOT / "labels" / "train").glob("*.txt")))
    val_labels = len(list((OUT_ROOT / "labels" / "val").glob("*.txt")))

    print("\n" + "=" * 60)
    print("MIXED V5 DATASET CREATED")
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