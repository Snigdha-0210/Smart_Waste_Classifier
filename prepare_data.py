from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


# ============================================================
# 1. CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]


# ============================================================
# 2. ORIGINAL LABEL → NEW LABEL
# ============================================================

# Original dataset labels:
#
# 0 = Cardboard
# 1 = Food Organics
# 2 = Glass
# 3 = Metal
# 4 = Miscellaneous Trash
# 5 = Paper
# 6 = Plastic
# 7 = Textile Trash
# 8 = Vegetation
#
# We are keeping only:
#
# 0 = Cardboard
# 1 = Food Organics
# 2 = Glass
# 3 = Metal
# 5 = Paper
# 6 = Plastic
#
# New labels:
#
# 0 → Cardboard
# 1 → Food Organics
# 2 → Glass
# 3 → Metal
# 5 → Paper → NEW 4
# 6 → Plastic → NEW 5

LABEL_MAPPING = {
    0: 0,   # Cardboard
    1: 1,   # Food Organics
    2: 2,   # Glass
    3: 3,   # Metal
    5: 4,   # Paper
    6: 5    # Plastic
}


# ============================================================
# 3. IMAGE SETTINGS
# ============================================================

# ResNet18 pretrained on ImageNet works best with
# ImageNet-style 224x224 images.

IMAGE_SIZE = 224


# ============================================================
# 4. IMAGE NORMALIZATION
# ============================================================

# These are the standard ImageNet mean and standard deviation.
#
# Because we are using a pretrained ResNet18, using the same
# normalization used during ImageNet training is important.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# 5. PYTORCH DATASET
# ============================================================

class WasteDataset(Dataset):

    def __init__(self, hf_dataset, transform=None):

        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        # Get one example
        item = self.dataset[index]

        # Get image
        image = item["image"]

        # Get original dataset label
        original_label = item["label"]

        # Convert original label to our 0-5 label
        label = LABEL_MAPPING[original_label]

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

def get_dataloaders(batch_size=32):

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    dataset = load_dataset(
        "ddompe/waste-segregation-dataset"
    )

    print("Dataset loaded successfully!")


    # --------------------------------------------------------
    # Filter to our six classes
    # --------------------------------------------------------

    def keep_six_classes(example):

        return example["label"] in LABEL_MAPPING


    print("\nFiltering unwanted classes...")

    dataset = dataset.filter(
        keep_six_classes
    )

    print("Filtering complete!")


    # ========================================================
    # 7. TRAINING TRANSFORMS
    # ========================================================

    # Training images receive augmentation.
    #
    # This helps the model learn that waste can appear:
    # - rotated
    # - flipped
    # - slightly zoomed
    # - slightly shifted
    #
    # We keep augmentation moderate because some classes
    # (especially Glass / Metal / Plastic) can be affected
    # by aggressive transformations.

    train_transform = transforms.Compose([

        # Resize while keeping enough visual information
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        # Random horizontal flip
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        # Small rotation
        transforms.RandomRotation(
            degrees=10
        ),

        # Slight changes in size and position
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10)
        ),

        # Convert PIL image → Tensor
        transforms.ToTensor(),

        # ImageNet normalization for pretrained ResNet
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


    # ========================================================
    # 8. VALIDATION / TEST TRANSFORMS
    # ========================================================

    # IMPORTANT:
    #
    # Validation and test images should NOT receive random
    # augmentation.
    #
    # Otherwise our evaluation results would change randomly.

    validation_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


    # ========================================================
    # 9. CREATE PYTORCH DATASETS
    # ========================================================

    train_dataset = WasteDataset(
        dataset["train"],
        transform=train_transform
    )

    validation_dataset = WasteDataset(
        dataset["validation"],
        transform=validation_transform
    )

    test_dataset = WasteDataset(
        dataset["test"],
        transform=validation_transform
    )


    # ========================================================
    # 10. CHECK CUDA
    # ========================================================

    use_cuda = torch.cuda.is_available()


    # ========================================================
    # 11. CREATE DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        # Faster transfer from CPU → GPU
        # when CUDA is available.
        pin_memory=use_cuda
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=use_cuda
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=use_cuda
    )


    # ========================================================
    # 12. PRINT DATASET INFORMATION
    # ========================================================

    print("\n" + "=" * 50)
    print("FINAL DATASET SIZES")
    print("=" * 50)

    print(
        "Training   :",
        len(train_dataset)
    )

    print(
        "Validation :",
        len(validation_dataset)
    )

    print(
        "Test       :",
        len(test_dataset)
    )


    # ========================================================
    # 13. PRINT IMAGE CONFIGURATION
    # ========================================================

    print("\n" + "=" * 50)
    print("IMAGE CONFIGURATION")
    print("=" * 50)

    print(
        "Image size :",
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        "Channels   : 3 (RGB)"
    )

    print(
        "Normalization: ImageNet"
    )

    print(
        "Batch size :",
        batch_size
    )

    print(
        "CUDA       :",
        use_cuda
    )


    # ========================================================
    # 14. RETURN DATALOADERS
    # ========================================================

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# ============================================================
# 15. TEST DATA PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\nTesting data pipeline...")

    train_loader, validation_loader, test_loader = get_dataloaders(
        batch_size=32
    )


    # ========================================================
    # GET ONE TRAINING BATCH
    # ========================================================

    print("\nGetting one training batch...")

    images, labels = next(
        iter(train_loader)
    )


    # ========================================================
    # PRINT BATCH INFORMATION
    # ========================================================

    print("\n" + "=" * 50)
    print("BATCH INFORMATION")
    print("=" * 50)

    print(
        "Images shape :",
        images.shape
    )

    print(
        "Labels shape :",
        labels.shape
    )

    print(
        "Image dtype  :",
        images.dtype
    )

    print(
        "Label dtype  :",
        labels.dtype
    )


    # ========================================================
    # LABEL RANGE
    # ========================================================

    print("\n" + "=" * 50)
    print("LABEL INFORMATION")
    print("=" * 50)

    print(
        "Minimum label:",
        labels.min().item()
    )

    print(
        "Maximum label:",
        labels.max().item()
    )


    # ========================================================
    # IMAGE VALUE RANGE
    # ========================================================

    print("\n" + "=" * 50)
    print("IMAGE VALUE RANGE")
    print("=" * 50)

    print(
        "Minimum pixel value:",
        images.min().item()
    )

    print(
        "Maximum pixel value:",
        images.max().item()
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 50)
    print("DATA PIPELINE TEST COMPLETE!")
    print("=" * 50)