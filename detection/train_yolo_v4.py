from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================================
# SMART WASTE CLASSIFIER
# YOLO11n - FINAL V4 10-CLASS TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "detection"
    / "dataset_final_v4"
    / "data.yaml"
)

RUN_NAME = "waste_yolo_v4_10class"

# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

MODEL = "yolo11n.pt"
IMG_SIZE = 640
EPOCHS = 80
BATCH = 16
WORKERS = 0

# RTX 4060 Laptop GPU
DEVICE = 0

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("SMART WASTE CLASSIFIER")
    print("YOLO11n - FINAL V4 10-CLASS TRAINING")
    print("=" * 75)

    print("\nProject:")
    print(PROJECT_ROOT)

    print("\nDataset:")
    print(DATASET)

    print("\nModel:")
    print(MODEL)

    print("\nTraining configuration:")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH}")
    print(f"  Workers    : {WORKERS}")
    print(f"  Device     : CUDA:{DEVICE}")

    # --------------------------------------------------------
    # CHECK GPU
    # --------------------------------------------------------

    print("\n=== GPU CHECK ===")

    if torch.cuda.is_available():

        print("CUDA available: YES")
        print("GPU:", torch.cuda.get_device_name(DEVICE))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(DEVICE).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    else:

        print("WARNING: CUDA is NOT available.")
        print("Training would run on CPU.")

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    print("\n=== DATASET CHECK ===")

    if not DATASET.exists():

        print("ERROR: Dataset YAML was not found:")
        print(DATASET)
        return

    print("Dataset YAML found.")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n=== LOADING YOLO11n ===")

    model = YOLO(MODEL)

    print("Pretrained YOLO11n loaded.")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n=== STARTING TRAINING ===")
    print("This may take several hours.")
    print("Do NOT close the terminal while training is running.")
    print("=" * 75)

    results = model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMG_SIZE,

        batch=BATCH,

        workers=WORKERS,

        device=DEVICE,

        project="runs/detect",

        name=RUN_NAME,

        exist_ok=False,

        pretrained=True,

        optimizer="auto",

        patience=20,

        save=True,

        save_period=10,

        cache=False,

        amp=True,

        plots=True,

        verbose=True

    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)

    run_directory = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / RUN_NAME
    )

    print("\nTraining results:")
    print(run_directory)

    print("\nBest model:")
    print(run_directory / "weights" / "best.pt")

    print("\nLast model:")
    print(run_directory / "weights" / "last.pt")

    print("\nIMPORTANT:")
    print("Do not delete best.pt.")
    print("We will use it for evaluation.")


if __name__ == "__main__":
    main()