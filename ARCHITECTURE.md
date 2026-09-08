# 🏗️ Smart Waste Classifier: System Architecture & Technical Specifications

<div align="center">

![Project Banner](thumbnail.jpg)

### Deep Learning-Powered Multi-Stage Waste Segregation, Object Detection & Eco-Disposal Guidance

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-11n_Detector-00FFFF.svg?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ResNet-18](https://img.shields.io/badge/Classifier-ResNet18_93.34%25-success.svg?style=for-the-badge&logo=deepnote&logoColor=white)](https://pytorch.org/vision/main/models/resnet.html)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_Dashboard-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CUDA Accelerated](https://img.shields.io/badge/Hardware-CUDA_Accelerated-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

</div>

---

## 📑 Table of Contents

1. [System Overview & High-Level Architecture](#1-system-overview--high-level-architecture)
2. [End-to-End Hybrid Detection + Classification Pipeline](#2-end-to-end-hybrid-detection--classification-pipeline)
3. [Dataset Engineering & Multi-Dataset Harmonization](#3-dataset-engineering--multi-dataset-harmonization)
4. [Model Architectures & Training Protocols](#4-model-architectures--training-protocols)
5. [OOD Guard V1: Uncertainty & Out-of-Distribution Engine](#5-ood-guard-v1-uncertainty--out-of-distribution-engine)
6. [Quantitative Benchmarks & Evaluation Suite](#6-quantitative-benchmarks--evaluation-suite)
7. [Visual Artifacts & Diagnostics Showcase](#7-visual-artifacts--diagnostics-showcase)
8. [Codebase Map & Module Specifications](#8-codebase-map--module-specifications)
9. [Operational Execution Guide](#9-operational-execution-guide)

---

## 1. System Overview & High-Level Architecture

The **Smart Waste Classifier** is an enterprise-grade, modular computer vision system designed to automate municipal and household solid waste segregation. It solves the critical bottleneck of recycling contamination by combining real-time multi-object localization, fine-grained residual classification, out-of-distribution (OOD) uncertainty filtering, and contextual eco-disposal guidance.

```mermaid
flowchart TD
    %% Global Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef guard fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef ui fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;

    subgraph DataEngine["1. Data Harmonization & Curation Layer"]
        HF[("Hugging Face Hub\n(ddompe/waste-segregation-dataset)")]:::data
        TACO[("TACO Raw Dataset\n(COCO Bounding Boxes)")]:::data
        SUPP[("Supplemental Custom Dataset\n(YOLO Annotations)")]:::data
        
        Merge["merge_detection_datasets.py\n• Label Validation\n• Class Remapping (6 Classes)\n• Prefixing & Collision Handling"]:::data
        Stratify["prepare_detection_data_v2.py\n• Stratified Splitting\n• Rare Class Preservation"]:::data
        
        TACO --> Stratify
        SUPP --> Merge
        TACO --> Merge
        HF --> DataPrep["prepare_data.py\n• 6-Class Remapping\n• Resizing & Augmentation"]:::data
    end

    subgraph CoreModels["2. Deep Learning Core"]
        ResNet["Fine-Tuned ResNet-18\n(waste_resnet18_best.pth)\n93.34% Accuracy"]:::model
        YOLO["YOLO11n Detector\n(yolo11n.pt / waste_yolo_combined)\nReal-time Bounding Boxes"]:::model
        BaseCNN["Custom 3-Block CNN Baseline\n(waste_classifier.pth)\n59.77% Accuracy"]:::model
        
        DataPrep --> ResNet
        DataPrep --> BaseCNN
        Merge --> YOLO
    end

    subgraph InferenceCascade["3. Hybrid Cascaded Inference Pipeline"]
        InputImg["Input Image / Camera Stream"]:::ui --> YOLO
        YOLO --> BBoxes["Localized Bounding Boxes\n+ Object Confidence"]:::model
        BBoxes --> Cropper["Dynamic Crop & Aspect Pad\n(Resize to 224x224 RGB)"]:::data
        Cropper --> ResNet
        ResNet --> Softmax["Softmax Probabilities\nAcross 6 Classes"]:::model
    end

    subgraph UncertaintyLayer["4. Robustness & OOD Guard V1"]
        Softmax --> Entropy["Normalized Shannon Entropy\nH(p) / ln(K)"]:::guard
        Softmax --> Margin["Top-1 vs Top-2 Margin Delta\nΔ = p₁ - p₂"]:::guard
        Cropper --> TTA["Test-Time Augmentations (TTA)\nFlips, Jitter, Scaled Passes"]:::guard
        TTA --> TTAVar["TTA Prediction Agreement\n& Variance Estimator"]:::guard
        
        Entropy --> Gate{"OOD Guard Decision Gating\n• Min Confidence >= 0.70\n• Min Margin >= 0.18\n• Max Entropy <= 0.72\n• Min TTA Agreement >= 0.60"}:::guard
        Margin --> Gate
        TTAVar --> Gate
    end

    subgraph Delivery["5. User Interfaces & Actionable Guidance"]
        Gate -- "Pass" --> Verified["Confident Prediction\n+ Bin Allocation\n+ Eco-Action Guidelines"]:::ui
        Gate -- "Flag" --> OODWarning["OOD / Ambiguous Alert\n+ Uncertainty Diagnostics\n+ Manual Inspection Prompt"]:::guard
        
        Verified --> Dashboard["Streamlit Web App (app.py)"]:::ui
        Verified --> CLI["CLI Predictor (predict.py)"]:::ui
        Verified --> MultiPipeline["End-to-End Pipeline (waste_pipeline.py)"]:::ui
    end
```

---

## 2. End-to-End Hybrid Detection + Classification Pipeline

In complex real-world municipal environments, scenes rarely contain a single centered object against a clean background. The system introduces a **Cascaded Dual-Stage Inference Architecture** (`waste_pipeline.py`):

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Camera Stream
    participant YOLO as YOLOv11 Detector
    participant Pre as Preprocessor & Aspect Padder
    participant ResNet as ResNet-18 Classifier
    participant OOD as OOD Guard V1
    participant UI as Visualizer / Dashboard

    User->>YOLO: Provide High-Resolution Raw Image
    Note over YOLO: Multi-object feature extraction & NMS
    YOLO-->>Pre: Return Bounding Boxes [x₁, y₁, x₂, y₂] & Detection Confidence
    
    loop For Each Detected Object
        Pre->>Pre: Crop Region of Interest (ROI) with Context Margin
        Pre->>Pre: Aspect-Ratio Preserving Resize to 224x224
        Pre->>Pre: Apply ImageNet Normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
        Pre->>ResNet: Forward Pass Tensor (Batch 1, 3, 224, 224)
        ResNet-->>OOD: Logits & Softmax Probabilities [p₀, p₁, ..., p₅]
        OOD->>OOD: Evaluate Shannon Entropy, Margin Delta & TTA Agreement
        alt High Confidence & Low Entropy
            OOD-->>UI: Verified Waste Category + Confidence + Recycling Guidelines
        else OOD or Low Confidence
            OOD-->>UI: Flagged as Ambiguous / OOD Item
        end
    end

    UI->>User: Render Visual Annotated Canvas with Bounding Boxes & Eco Tags
```

### Key Technical Advantages of Cascade Design:
1. **Separation of Concerns**: YOLO specializes in multi-object localization and clutter separation, while ResNet-18 specializes in high-fidelity material classification (e.g., discerning translucent plastic vs clear glass).
2. **Context Retention**: Cropping individual items eliminates background interference and lighting bias from surrounding conveyor belts or floors.
3. **Fail-Safe Operation**: If YOLO detects an unclassified item, ResNet-18 together with OOD Guard evaluates whether the item is valid municipal waste or non-waste noise.

---

## 3. Dataset Engineering & Multi-Dataset Harmonization

### 3.1. Unified 6-Class Waste Taxonomy

To create an industrial-grade taxonomy, ambiguous catch-all classes (*Miscellaneous Trash*, *Textile Trash*, and *Vegetation*) were pruned to eliminate label noise. All downstream classification and detection modules adhere strictly to the contiguous 6-class representation:

| Class ID | Waste Category | Target Stream | Primary Material Signatures | Eco-Action Protocol |
|:---:|:---|:---|:---|:---|
| `0` | **Cardboard** | Recyclable / Dry Waste | Corrugated boxes, packaging cartons, paperboard | Flatten boxes, keep dry, remove adhesive tapes |
| `1` | **Food Organics** | Organic / Compost | Fruit rinds, vegetable peels, leftovers, coffee grounds | Segregate to green compost bin; keep free of plastics |
| `2` | **Glass** | Recyclable Waste | Beverage bottles, food jars, clear/colored glassware | Rinse cleanly, remove metallic/plastic caps, avoid breakage |
| `3` | **Metal** | Recyclable Waste | Aluminum cans, tin containers, foil trays, aerosol cans | Empty contents, rinse residues, compress cans |
| `4` | **Paper** | Recyclable / Dry Waste | Office documents, newspapers, magazines, paper bags | Keep dry, ensure free of food/grease contamination |
| `5` | **Plastic** | Recyclable Waste | PET bottles, HDPE jugs, PP containers, clean wrappers | Check resin identification code, rinse thoroughly |

---

### 3.2. Multi-Dataset Merger & Harmonization Pipeline (`merge_detection_datasets.py`)

The detection subsystem fuses multiple heterogeneous datasets into a single standardized YOLO-format repository (`dataset_combined`):

```mermaid
flowchart LR
    subgraph RawSources["Raw Ingestion"]
        T["TACO Dataset\n(COCO Annotations)"]
        S["Supplemental Custom Dataset\n(6-Class YOLO Annotations)"]
    end

    subgraph Harmonization["Harmonization Engine (merge_detection_datasets.py)"]
        V1["Validate YOLO Labels\n• Normalized BBox Coordinates [0, 1]\n• Positive Non-Zero Width/Height\n• Bounded Class IDs"]
        R1["Remap Supplemental Classes:\n• 0 (Bio) → 1 (Food Organics)\n• 1 (Cardboard) → 0 (Cardboard)\n• 2 (Glass) → 2 (Glass)\n• 3 (Metal) → 3 (Metal)\n• 4 (Paper) → 4 (Paper)\n• 5 (Plastic) → 5 (Plastic)"]
        Pref["Prefix Filenames to Prevent Collision\n• taco_*.jpg / taco_*.txt\n• supp_*.jpg / supp_*.txt"]
    end

    subgraph OutputRepo["Standardized Combined Dataset (dataset_combined)"]
        YAML["data.yaml\n(Path, Splits, 6 Classes)"]
        Imgs["images/\n├── train/\n├── val/\n└── test/"]
        Lbls["labels/\n├── train/\n├── val/\n└── test/"]
    end

    T --> V1 --> Pref --> OutputRepo
    S --> R1 --> V1 --> Pref --> OutputRepo
```

---

## 4. Model Architectures & Training Protocols

### 4.1. Fine-Tuned ResNet-18 Classifier (`model_resnet.py` & `train_resnet.py`)

- **Base Architecture**: 18-layer Residual Network with skip connections to mitigate vanishing gradients.
- **Pretrained Initialization**: `ResNet18_Weights.DEFAULT` (ImageNet-1K).
- **Head Adaptation**: Fully Connected linear head replaced with `nn.Linear(in_features=512, out_features=6)`.
- **Loss Function**: Multi-Class Cross-Entropy Loss with Softmax:
  $$\mathcal{L}_{CE} = -\sum_{i=1}^K y_i \log(\hat{y}_i)$$
- **Optimization Strategy**:
  - Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$)
  - Base Learning Rate: $\eta = 10^{-4}$
  - Learning Rate Scheduler: `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)`
  - Batch Size: 32
  - Augmentation: Stochastic horizontal flips ($p=0.5$), random rotations ($\pm 10^\circ$), random affine scaling ($[0.90, 1.10]$) and translations ($\pm 5\%$).

---

### 4.2. Custom 3-Block CNN Baseline (`model.py` & `train.py`)

As a rigorous empirical baseline, a custom Convolutional Neural Network was designed from scratch:
- **Block 1**: `Conv2d(3, 32, kernel=3, pad=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Block 2**: `Conv2d(32, 64, kernel=3, pad=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Block 3**: `Conv2d(64, 128, kernel=3, pad=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Classifier Head**: `Flatten` $\rightarrow$ `Linear(128 * 16 * 16, 256)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 6)`

---

### 4.3. YOLOv11n Object Detector (`detection/train_yolo.py`)

- **Architecture**: Ultralytics YOLO11 Nano (`yolo11n.pt`) with C3k2 modules and SPPF (Spatial Pyramid Pooling - Fast).
- **Resolution**: $640 \times 640$ pixels.
- **Hyperparameters**:
  - Epochs: 100 with Early Stopping (`patience=20`)
  - Batch Size: 16
  - Mixed Precision: Automated Mixed Precision (`amp=True`)
  - Augmentations: Mosaic augmentation (`mosaic=1.0`), closed during the final 10 epochs (`close_mosaic=10`), rotation ($\pm 10^\circ$), scale jitter ($0.5$).

---

## 5. OOD Guard V1: Uncertainty & Out-of-Distribution Engine

To prevent silent misclassification when arbitrary or out-of-domain images are submitted, **OOD Guard V1** executes a multi-signal uncertainty quantification pipeline:

```mermaid
flowchart TD
    In[Input Image Crop] --> BasePred[Standard Forward Pass]
    In --> TTAPass1[TTA: Horizontal Flip]
    In --> TTAPass2[TTA: Slight Rotation]
    In --> TTAPass3[TTA: Contrast Scale]
    
    BasePred --> ProbCalc["Compute Softmax: p = [p₀, ..., p₅]"]
    ProbCalc --> C1["1. Top-1 Confidence: max(p)"]
    ProbCalc --> C2["2. Top-2 Margin Delta: Δ = p₁ - p₂"]
    ProbCalc --> C3["3. Normalized Shannon Entropy:\nH_norm = -Σ(pᵢ ln pᵢ) / ln(6)"]
    
    TTAPass1 --> TTAVote[TTA Prediction Agreement %]
    TTAPass2 --> TTAVote
    TTAPass3 --> TTAVote
    
    C1 --> Evaluator{"OOD Evaluation Engine"}
    C2 --> Evaluator
    C3 --> Evaluator
    TTAVote --> Evaluator
    
    Evaluator --> |"max(p) >= 0.70<br/>Δ >= 0.18<br/>H_norm <= 0.72<br/>TTA >= 60%"| Approved["VERIFIED IN-DISTRIBUTION<br/>(Render Waste Category & Eco Advice)"]
    Evaluator --> |"Condition Failed"| Flagged["OOD / UNCERTAIN OBJECT<br/>(Display Alert & Request Clarification)"]
```

---

## 6. Quantitative Benchmarks & Evaluation Suite

### 6.1. Empirical Performance Summary

| Architecture / Model | Input Resolution | Pretrained Backbone | Test Accuracy | Test Loss | Correct / Total | Inference Speed (GPU) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 3-Block CNN** | $128 \times 128$ | None (Scratch) | **59.77%** | 1.1400 | 422 / 706 | ~3.2 ms |
| **Fine-Tuned ResNet-18** | $224 \times 224$ | ImageNet-1K | **93.34%** | **0.2114** | **659 / 706** | ~5.8 ms |
| **YOLO11n Detector** | $640 \times 640$ | COCO Pretrained | Real-Time Localization | Fast NMS | Multi-Object | ~8.4 ms |

> 🚀 **Transfer Learning Gain**: ResNet-18 delivers a **+33.57% net accuracy improvement** and reduces misclassifications from 284 down to 47 items on the held-out test split.

---

### 6.2. Per-Class Quantitative Metrics (ResNet-18)

| Class ID | Class Name | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| `0` | **Cardboard** | 0.94 | 0.93 | 0.94 | 118 |
| `1` | **Food Organics** | 0.97 | 0.96 | 0.96 | 120 |
| `2` | **Glass** | 0.91 | 0.92 | 0.91 | 115 |
| `3` | **Metal** | 0.92 | 0.90 | 0.91 | 112 |
| `4` | **Paper** | 0.91 | 0.93 | 0.92 | 119 |
| `5` | **Plastic** | 0.95 | 0.96 | 0.95 | 122 |
| **Macro Avg** | — | **0.93** | **0.93** | **0.93** | **706** |
| **Weighted Avg** | — | **0.93** | **0.93** | **0.93** | **706** |

---

## 7. Visual Artifacts & Diagnostics Showcase

### 7.1. Dataset Visual Distribution

![Dataset Samples Preview](dataset_samples.png)
![Multi-Class Dataset Samples](dataset_samples_multiple.png)

---

### 7.2. Model Confusion Matrices

![ResNet-18 Confusion Matrix](confusion_matrix_resnet.png)
![Baseline CNN Confusion Matrix](confusion_matrix.png)

---

### 7.3. Error Analysis & Edge Cases

The error analysis module (`error_analysis.py`) identifies optical ambiguities (e.g. reflective foil vs polished metals):

![ResNet-18 Error Analysis](resnet_error_analysis.png)

---

### 7.4. Real-Time Detection & Multi-Stage Cascaded Results

![Detected Waste](detected_waste.jpg)
![Waste Pipeline Result](waste_pipeline_result.jpg)

---

## 8. Codebase Map & Module Specifications

```text
Smart_Waste_Classifier/
│
├── .gitignore                         # Comprehensive exclusion rules (datasets, logs, weights, temp files)
├── LICENSE                            # Open-source MIT License
├── README.md                          # Repository landing documentation & overview
├── ARCHITECTURE.md                    # In-depth architectural specifications & Mermaid diagrams
├── DATASET_PLAN.md                    # Data curation strategy & class filtering rationale
├── requirements.txt                   # Production Python dependencies
│
├── app.py                             # Interactive Streamlit Web App with OOD Guard V1
├── predict.py                         # Standalone CLI & direct image URL predictor
├── waste_pipeline.py                  # Dual-stage cascaded YOLO detection + ResNet classification
├── detect_waste.py                    # Standalone YOLO real-time detection visualizer
├── test_yolo.py                       # YOLO benchmarking and test inference runner
│
├── model_resnet.py                    # ResNet-18 transfer learning architecture
├── model.py                           # Baseline 3-block CNN architecture
├── prepare_data.py                    # Hugging Face dataset downloader, remapper & DataLoaders
├── train_resnet.py                    # ResNet-18 trainer with LR plateau scheduler & checkpointing
├── train.py                           # Baseline CNN trainer
├── evaluate_resnet.py                 # Comprehensive ResNet-18 evaluation metrics
├── evaluate.py                        # Baseline CNN evaluation metrics
├── confusion_matrix_resnet.py         # ResNet-18 confusion matrix & classification report generator
├── confusion_matrix.py                # Baseline CNN confusion matrix generator
├── error_analysis.py                  # High-confidence misclassification visual inspection
├── inspect_dataset.py                 # Dataset split and label distribution verification
├── visualize_dataset.py               # Single sample inspector
├── visualize_dataset_multiple.py      # Multi-sample grid visualizer
├── project_status.py                  # System health, hardware & checkpoint diagnostic suite
│
├── detection/                         # Object Detection Subsystem
│   ├── merge_detection_datasets.py    # Merges TACO + supplemental datasets with validation
│   ├── train_yolo.py                  # YOLOv11 training script on combined 6-class dataset
│   ├── download_taco_images.py        # Automated TACO image downloader
│   └── data.yaml                      # YOLO dataset configuration
│
├── prepare_detection_data.py          # TACO COCO-to-YOLO dataset converter V1
├── prepare_detection_data_v2.py       # TACO-to-YOLO converter V2 with stratified split protection
│
├── waste_resnet18_best.pth            # Trained ResNet-18 model weights (93.34% accuracy)
├── waste_classifier.pth               # Trained Baseline CNN model weights (59.77% accuracy)
├── yolo11n.pt                         # Pretrained YOLOv11 neural network weights
│
├── dataset_samples.png                # Dataset class preview visualization
├── dataset_samples_multiple.png       # Comprehensive multi-class grid visualization
├── confusion_matrix_resnet.png        # ResNet-18 confusion matrix heatmap
├── confusion_matrix.png               # Baseline CNN confusion matrix heatmap
├── resnet_error_analysis.png          # ResNet-18 misclassification diagnostic chart
├── detected_waste.jpg                 # YOLO object detection visual output
├── waste_pipeline_result.jpg          # Dual-stage detection + classification output
└── thumbnail.jpg                      # Project header & social banner
```

---

## 9. Operational Execution Guide

### 9.1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Snigdha-0210/Smart_Waste_Classifier.git
cd Smart_Waste_Classifier

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate     # On Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

### 9.2. Launching the Interactive Web Dashboard

```bash
streamlit run app.py
```

### 9.3. Running CLI Inference

```bash
# Predict from local image
python predict.py "path/to/image.jpg"

# Predict directly from a web URL
python predict.py "https://example.com/waste_sample.jpg"
```

### 9.4. Running the End-to-End Cascaded Pipeline

```bash
python waste_pipeline.py
```

### 9.5. Dataset Harmonization & YOLO Detector Training

```bash
# Harmonize and merge detection datasets
python detection/merge_detection_datasets.py

# Train YOLO11n on the combined 6-class dataset
python detection/train_yolo.py
```

### 9.6. System Diagnostics & Health Check

```bash
python project_status.py
```

---

<div align="center">

**Smart Waste Classifier** • Built with ❤️ by [Snigdha](https://github.com/Snigdha-0210)

</div>
