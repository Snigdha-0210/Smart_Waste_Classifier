from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    PROJECT_ROOT
    / "detection"
    / "dataset_combined"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "detection"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SMART WASTE DETECTION - COMBINED DATASET TRAINING")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATA_YAML)

    print()
    print("Model:")
    print("YOLO11n")

    print()
    print("Classes:")
    print("  0: Cardboard")
    print("  1: Food Organics")
    print("  2: Glass")
    print("  3: Metal")
    print("  4: Paper")
    print("  5: Plastic")

    print()
    print("Training configuration:")
    print("  Image size : 640")
    print("  Batch size : 16")
    print("  Epochs     : 100")
    print("  Workers    : 0")
    print("  Device     : GPU 0")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    # --------------------------------------------------------
    # Load pretrained YOLO11n
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING YOLO11n")
    print("=" * 70)

    model = YOLO("yolo11n.pt")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    results = model.train(

        data=str(DATA_YAML),

        # Training
        epochs=100,
        imgsz=640,
        batch=16,

        # Hardware
        device=0,
        workers=0,

        # Training behavior
        pretrained=True,
        amp=True,

        # Validation
        val=True,

        # Early stopping
        patience=20,

        # Augmentation
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # Mosaic augmentation
        mosaic=1.0,

        # Close mosaic near the end
        close_mosaic=10,

        # Output
        project=str(RUNS_DIR),
        name="waste_yolo_combined",
        exist_ok=True,

        # Save
        save=True,
        plots=True,

        # Reproducibility
        seed=42,

        # Cache disabled to avoid excessive RAM usage
        cache=False,
    )

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    run_dir = RUNS_DIR / "waste_yolo_combined"

    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"

    print()
    print("Training directory:")
    print(run_dir)

    print()
    print("Best model:")
    print(best_model)

    print()
    print("Last model:")
    print(last_model)

    print()
    print("Next step will be evaluation on the TEST dataset.")


if __name__ == "__main__":
    main()