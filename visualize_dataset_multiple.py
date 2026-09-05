from datasets import load_dataset
import matplotlib.pyplot as plt
import random

print("Loading dataset...")

dataset = load_dataset("ddompe/waste-segregation-dataset")

print("Dataset loaded.")

# Our six target classes
target_classes = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Cardboard",
    "Food Organics"
]

# Number of images per class
images_per_class = 5

# Create figure
fig, axes = plt.subplots(
    len(target_classes),
    images_per_class,
    figsize=(15, 18)
)

for row, class_name in enumerate(target_classes):

    # Find all training images belonging to this class
    class_indices = [
        i
        for i, label in enumerate(dataset["train"]["label_name"])
        if label == class_name
    ]

    # Randomly select 5 images
    selected_indices = random.sample(
        class_indices,
        images_per_class
    )

    for col, index in enumerate(selected_indices):

        image = dataset["train"][index]["image"]

        ax = axes[row, col]

        ax.imshow(image)
        ax.axis("off")

        # Put class name on the first image of each row
        if col == 0:
            ax.set_title(class_name, fontsize=14, loc="left")

plt.tight_layout()

output_path = "dataset_samples_multiple.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print()
print(f"Visualization saved to: {output_path}")
print("Open the image and inspect the samples.")