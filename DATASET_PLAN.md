# 📋 Dataset Plan & Curation Strategy

## 1. Dataset Origin & Source

The primary data source for this project is the **Hugging Face Waste Segregation Dataset**:
- **Hugging Face Hub ID**: [`ddompe/waste-segregation-dataset`](https://huggingface.co/datasets/ddompe/waste-segregation-dataset)
- **Modality**: RGB Real-world photographs of municipal and household solid waste.
- **Original Resolution**: Variable dimensions, centered objects on diverse real-world backgrounds.

---

## 2. Class Selection & Filtering Rationale

The original dataset comprises **9 categorical classes**:
0. `Cardboard`
1. `Food Organics`
2. `Glass`
3. `Metal`
4. `Miscellaneous Trash` *(Excluded)*
5. `Paper`
6. `Plastic`
7. `Textile Trash` *(Excluded)*
8. `Vegetation` *(Excluded)*

### Exclusion Criteria

To build a reliable, high-precision industrial/household waste classifier, 3 classes were explicitly filtered out:

1. **`Miscellaneous Trash` (Label 4)**: Highly heterogeneous and ambiguously defined catch-all class. Contains overlapping visual features with all other categories, acting as a label noise source and degrading cross-entropy convergence.
2. **`Textile Trash` (Label 7)**: Follows distinct textile-recycling and donation streams not typically managed in standard municipal solid-waste recycling bins.
3. **`Vegetation` (Label 8)**: Yard waste and garden debris require distinct industrial composting infrastructure separate from domestic food organics.

---

## 3. Class Remapping & Target Taxonomy

The remaining **6 target classes** are remapped to a contiguous zero-indexed tensor representation:

| Original ID | Class Name | Target ID | Waste Stream Category |
|:---:|:---|:---:|:---|
| `0` | **Cardboard** | `0` | Recyclable / Dry Waste |
| `1` | **Food Organics** | `1` | Organic Waste / Compost |
| `2` | **Glass** | `2` | Recyclable Waste |
| `3` | **Metal** | `3` | Recyclable Waste |
| `5` | **Paper** | `4` | Recyclable / Dry Waste |
| `6` | **Plastic** | `5` | Recyclable Waste |

### Code Implementation (`prepare_data.py`)

```python
LABEL_MAPPING = {
    0: 0,   # Cardboard
    1: 1,   # Food Organics
    2: 2,   # Glass
    3: 3,   # Metal
    5: 4,   # Paper
    6: 5    # Plastic
}
```

---

## 4. Dataset Splits & Distribution

The dataset utilizes the official standard partition splits:
- **Train Split**: Used for model weight optimization via backpropagation.
- **Validation Split**: Used during training epochs to tune learning rate schedules (`ReduceLROnPlateau`) and save the optimal model checkpoint (`waste_resnet18_best.pth`).
- **Test Split**: Strictly held out for final unbiased evaluation, confusion matrix generation, and error analysis.

---

## 5. Image Transformation & Augmentation Pipeline

### Training Transforms (Robustness & Invariance)

Augmentation is applied to mimic natural perturbations in recycling camera feeds (angled drops, rotations, scale variations):

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Validation & Test Transforms (Deterministic Evaluation)

```python
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 6. DataLoader Pipeline Configurations

- **Batch Size**: 32
- **Shuffling**: `True` for training; `False` for validation & testing.
- **Pin Memory**: Enabled dynamically when CUDA is available for asynchronous CPU $\rightarrow$ GPU memory transfers.
- **Workers**: 0 (cross-platform Windows/Linux compatibility).
