from ultralytics import YOLO

# ============================================================
# SMART WASTE DETECTION - V6
# Mixed synthetic + real-world dataset
# ============================================================

MODEL = "yolo11n.pt"

DATA = r"detection\dataset_mixed_v6\data.yaml"

IMG_SIZE = 640
EPOCHS = 80
BATCH = 16
WORKERS = 0
DEVICE = 0

PROJECT = r"runs\detect"
NAME = "waste_yolo_v6"

model = YOLO(MODEL)

model.train(
    data=DATA,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,

    project=PROJECT,
    name=NAME,

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

print()
print("=" * 70)
print("V6 TRAINING COMPLETE")
print("=" * 70)
print("Best model:")
print(
    r"runs\detect\runs\detect\waste_yolo_v6\weights\best.pt"
)
print("=" * 70)