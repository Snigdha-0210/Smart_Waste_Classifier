import os
import sys
import subprocess
from pathlib import Path


# ============================================================
# SMART WASTE CLASSIFIER — PROJECT STATUS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# HELPERS
# ============================================================

def exists(filename):
    return (BASE_DIR / filename).exists()


def folder_exists(folder):
    return (BASE_DIR / folder).is_dir()


def get_size(filename):
    path = BASE_DIR / filename

    if not path.exists():
        return "Not found"

    size_mb = path.stat().st_size / (1024 * 1024)

    return f"{size_mb:.2f} MB"


def check_package(package_name):

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                package_name
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            for line in result.stdout.splitlines():

                if line.startswith("Version:"):
                    return line.replace(
                        "Version:",
                        ""
                    ).strip()

            return "Installed"

        return "Not installed"

    except Exception:

        return "Could not check"


def print_status(name, status, details=""):

    if status:
        symbol = "✓"
    else:
        symbol = "✗"

    print(
        f"{symbol} {name:<35} "
        f"{details}"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("SMART WASTE CLASSIFIER — PROJECT STATUS")
print("=" * 75)

print()
print("Project directory:")
print(BASE_DIR)

print()


# ============================================================
# 1. ENVIRONMENT
# ============================================================

print("=" * 75)
print("1. PYTHON ENVIRONMENT")
print("=" * 75)

print_status(
    "Python",
    True,
    sys.version.split()[0]
)

print_status(
    "Virtual environment",
    (
        hasattr(sys, "prefix")
        and sys.prefix != sys.base_prefix
    )
)

print()


# ============================================================
# 2. PYTORCH / CUDA
# ============================================================

print("=" * 75)
print("2. PYTORCH / GPU")
print("=" * 75)

try:

    import torch

    print_status(
        "PyTorch",
        True,
        torch.__version__
    )

    cuda_available = torch.cuda.is_available()

    print_status(
        "CUDA available",
        cuda_available
    )

    if cuda_available:

        print_status(
            "GPU",
            True,
            torch.cuda.get_device_name(0)
        )

        print(
            f"  CUDA version: {torch.version.cuda}"
        )

    else:

        print(
            "  GPU acceleration is not available."
        )

except Exception as e:

    print_status(
        "PyTorch",
        False,
        str(e)
    )

print()


# ============================================================
# 3. CORE PROJECT FILES
# ============================================================

print("=" * 75)
print("3. CORE PROJECT FILES")
print("=" * 75)

core_files = [

    "prepare_data.py",
    "model.py",
    "model_resnet.py",
    "train.py",
    "train_resnet.py",

    "evaluate.py",
    "evaluate_resnet.py",

    "confusion_matrix.py",
    "confusion_matrix_resnet.py",

    "error_analysis.py",

    "predict.py",

    "detect_waste.py",

    "waste_pipeline.py",

    "visualize_dataset.py",
    "visualize_dataset_multiple.py",
    "inspect_dataset.py",

    "app.py",
    "app_backup.py",

    "waste_resnet18_best.pth"

]

for filename in core_files:

    path = BASE_DIR / filename

    if path.exists():

        size = get_size(filename)

        print_status(
            filename,
            True,
            size
        )

    else:

        print_status(
            filename,
            False,
            "Missing"
        )

print()


# ============================================================
# 4. RESNET RESULTS
# ============================================================

print("=" * 75)
print("4. RESNET18 RESULTS")
print("=" * 75)

print_status(
    "ResNet18 trained",
    exists("waste_resnet18_best.pth"),
    "Best model exists"
)

print()
print("Known results from your completed evaluation:")
print()
print("  Training images : 2246")
print("  Validation      : 562")
print("  Test images     : 706")
print()
print("  Test accuracy   : 93.34%")
print("  Correct         : 659 / 706")
print("  Incorrect       : 47 / 706")
print()
print("  Original CNN    : 59.77%")
print("  ResNet18        : 93.34%")
print("  Improvement     : +33.57 percentage points")

print()


# ============================================================
# 5. CLASSIFICATION CLASSES
# ============================================================

print("=" * 75)
print("5. CURRENT WASTE CLASSES")
print("=" * 75)

classes = [

    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"

]

for i, class_name in enumerate(classes):

    print(
        f"  {i}: {class_name}"
    )

print()


# ============================================================
# 6. ERROR ANALYSIS
# ============================================================

print("=" * 75)
print("6. ERROR ANALYSIS")
print("=" * 75)

print()
print("Total test images : 706")
print("Correct           : 659")
print("Incorrect         : 47")
print("Accuracy          : 93.34%")

print()
print("Largest confusion pairs:")
print()
print("  Plastic   -> Metal       : 12")
print("  Paper     -> Cardboard   : 7")
print("  Metal     -> Plastic     : 5")
print("  Cardboard -> Plastic     : 4")
print("  Plastic   -> Glass       : 4")

print()
print(
    "Interpretation:"
)

print(
    "The classifier works well on individual waste images,"
)

print(
    "but visually similar materials are sometimes confused."
)

print()


# ============================================================
# 7. STREAMLIT APPLICATION
# ============================================================

print("=" * 75)
print("7. STREAMLIT APPLICATION")
print("=" * 75)

print_status(
    "Streamlit app",
    exists("app.py")
)

print_status(
    "App backup",
    exists("app_backup.py")
)

if exists("app.py"):

    print()
    print(
        "Current app:"
    )

    print(
        "  ✓ Image upload"
    )

    print(
        "  ✓ ResNet18 prediction"
    )

    print(
        "  ✓ Confidence display"
    )

    print(
        "  ✓ Class probabilities"
    )

    print(
        "  ✓ Dashboard/history work"
    )

print()


# ============================================================
# 8. YOLO / OBJECT DETECTION
# ============================================================

print("=" * 75)
print("8. OBJECT DETECTION / YOLO")
print("=" * 75)

ultralytics_version = check_package(
    "ultralytics"
)

if ultralytics_version != "Not installed":

    print_status(
        "Ultralytics / YOLO",
        True,
        ultralytics_version
    )

else:

    print_status(
        "Ultralytics / YOLO",
        False,
        "Not installed"
    )


print_status(
    "Detection directory",
    folder_exists("detection")
)

print_status(
    "Detection dataset directory",
    folder_exists(
        os.path.join(
            "detection",
            "datasets"
        )
    )
)

print()


# ============================================================
# 9. CURRENT PROJECT LEVEL
# ============================================================

print("=" * 75)
print("9. CURRENT PROJECT LEVEL")
print("=" * 75)

print()
print("CURRENTLY COMPLETED:")
print()
print("  [✓] Dataset loading")
print("  [✓] Dataset filtering")
print("  [✓] Data preprocessing")
print("  [✓] Original CNN baseline")
print("  [✓] ResNet18 model")
print("  [✓] GPU/CUDA training")
print("  [✓] Model evaluation")
print("  [✓] Confusion matrix")
print("  [✓] Error analysis")
print("  [✓] Prediction script")
print("  [✓] Streamlit application")
print("  [✓] Image upload prediction")

print()
print("CURRENT LIMITATION:")
print()
print(
    "  [!] Current ResNet is an IMAGE CLASSIFIER."
)

print(
    "      It predicts one dominant class per image."
)

print()
print(
    "  [!] It is NOT yet an object detector."
)

print(
    "      It cannot reliably locate multiple waste objects"
)

print(
    "      inside a large garbage heap."
)

print()


# ============================================================
# 10. NEXT DEVELOPMENT STAGE
# ============================================================

print("=" * 75)
print("10. NEXT DEVELOPMENT STAGE")
print("=" * 75)

print()
print("NEXT:")
print()

print(
    "  >>> OBJECT DETECTION"
)

print()

print(
    "Goal:"
)

print(
    "  Input:  Image containing many waste objects"
)

print(
    "  Output: Bounding boxes + class + confidence"
)

print()

print(
    "Example:"
)

print(
    "  Plastic bottle   -> Plastic"
)

print(
    "  Metal can        -> Metal"
)

print(
    "  Cardboard box    -> Cardboard"
)

print(
    "  Paper            -> Paper"
)

print()

print(
    "Then:"
)

print(
    "  Detection -> Counting -> Waste composition -> Dashboard"
)

print()


# ============================================================
# 11. RECOMMENDED ROADMAP
# ============================================================

print("=" * 75)
print("11. RECOMMENDED ROADMAP")
print("=" * 75)

roadmap = [

    "Find suitable multi-object waste detection dataset",

    "Prepare YOLO dataset",

    "Create data.yaml",

    "Train YOLO detector using RTX 4060",

    "Evaluate detection performance",

    "Test on realistic garbage heaps",

    "Add object detection to Streamlit",

    "Count detected waste objects",

    "Calculate waste composition",

    "Add confidence / unknown detection",

    "Add webcam / real-time detection",

    "Optimize final application"

]

for i, item in enumerate(roadmap, 1):

    print(
        f"  {i:02d}. {item}"
    )

print()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print()

print(
    "Your classification system is already working."
)

print(
    "ResNet18 achieved 93.34% test accuracy."
)

print()

print(
    "The next major upgrade is NOT another classifier."
)

print(
    "The next major upgrade is OBJECT DETECTION."
)

print()

print(
    "This will allow the system to detect multiple"
)

print(
    "individual waste objects inside one garbage image."
)

print()

print(
    "================================================"
)

print(
    "STATUS CHECK COMPLETE"
)

print(
    "================================================"
)

print()