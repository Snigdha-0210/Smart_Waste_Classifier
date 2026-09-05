from datasets import load_dataset
import matplotlib.pyplot as plt
import random

print("Loading dataset...")

dataset = load_dataset("ddompe/waste-segregation-dataset")

print("Dataset loaded.")

target_classes = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Cardboard",
    "Food Organics"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, class_name in zip(axes.flatten(), target_classes):

    class_indices = [
        i
        for i, label in enumerate(dataset["train"]["label_name"])
        if label == class_name
    ]

    index = random.choice(class_indices)

    image = dataset["train"][index]["image"]

    ax.imshow(image)
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()

# Save instead of displaying
output_path = "dataset_samples.png"
plt.savefig(output_path, dpi=150)

print(f"\nVisualization saved to: {output_path}")
print("Open dataset_samples.png to inspect the images.")