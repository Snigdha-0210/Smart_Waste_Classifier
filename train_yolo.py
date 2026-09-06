from ultralytics import YOLO
import torch
from pathlib import Path


def main():

    print("=" * 70)
    print("SMART WASTE DETECTION - YOLO TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # CHECK DEVICE
    # ---------------------------------------------------------

    print("\nChecking device...")

    if torch.cuda.is_available():
        print("CUDA available: YES")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        device = 0
    else:
        print("CUDA available: NO")
        print("Training will use CPU.")
        device = "cpu"

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    data_yaml = Path("detection/data.yaml")

    print("\nDataset configuration:")
    print(data_yaml.resolve())

    if not data_yaml.exists():
        print("\nERROR: data.yaml not found.")
        return

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    print("\nLoading pretrained YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded.")

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\nStarting training...")
    print("=" * 70)

    results = model.train(

        # Dataset
        data=str(data_yaml),

        # Model
        pretrained=True,

        # Training
        epochs=80,
        imgsz=640,
        batch=16,

        # GPU
        device=device,

        # IMPORTANT FOR WINDOWS
        # Prevents the multiprocessing crash
        workers=0,

        # Augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # Optimization
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Training behavior
        patience=15,
        amp=True,

        # Output
        project="detection/runs",
        name="waste_yolo",
        exist_ok=False,

        # Save checkpoints
        save=True,
        save_period=10,

        # Validation
        val=True,

        # Reproducibility
        seed=0,

        # Display
        verbose=True,
        plots=True
    )

    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO TRAINING COMPLETE")
    print("=" * 70)

    print("\nTraining results:")
    print(results)

    print("\nYour trained model should be here:")

    print(
        Path("detection/runs/detect/waste_yolo/weights/best.pt").resolve()
    )

    print("\nYou can use best.pt for detection.")
    print("=" * 70)


if __name__ == "__main__":
    main()