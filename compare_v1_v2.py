from pathlib import Path

names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

roots = {
    "V1": Path(r"runs\detect\runs\detect\detection\combined_test_predictions\labels"),
    "V2": Path(r"runs\detect\runs\detect\runs\detect\detection\v2_test_predictions\labels")
}

print("=" * 70)
print("V1 vs V2 TEST PREDICTION COMPARISON")
print("=" * 70)

for model_name, folder in roots.items():

    if not folder.exists():
        print(f"\n{model_name}: FOLDER NOT FOUND")
        print(folder)
        continue

    files = list(folder.glob("*.txt"))

    counts = [0] * 6
    total = 0

    for file in files:
        try:
            lines = file.read_text().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) >= 1:
                    class_id = int(parts[0])

                    if 0 <= class_id < 6:
                        counts[class_id] += 1
                        total += 1

        except Exception as e:
            print(f"Could not read {file.name}: {e}")

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Images with predictions : {len(files)}")
    print(f"Total detections        : {total}")

    for i, name in enumerate(names):
        print(f"{name:18}: {counts[i]}")


print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)