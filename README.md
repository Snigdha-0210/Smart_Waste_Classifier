<div align="center">

# ♻️ Smart Waste Classifier
### Deep Learning-Powered Multi-Stage Waste Segregation, Object Detection & Eco-Disposal Guidance System

![Project Banner](thumbnail.jpg)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-11n_Detector-00FFFF.svg?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ddompe/waste-segregation-dataset)
[![Model](https://img.shields.io/badge/Model-ResNet18-success.svg?style=for-the-badge&logo=deepnote&logoColor=white)](https://pytorch.org/vision/main/models/resnet.html)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-93.34%25-brightgreen.svg?style=for-the-badge&logo=checkmarx&logoColor=white)](#-model-benchmarks--performance)
[![CUDA](https://img.shields.io/badge/Hardware-CUDA_Accelerated-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>An intelligent, end-to-end computer vision platform that localizes and classifies municipal and household waste into 6 recyclable and organic streams with 93.34% accuracy, equipped with OOD Guard V1 uncertainty filtering and real-time eco-guidance.</strong>
</p>

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Dual-Stage Detection Pipeline](#-dual-stage-cascaded-detection-pipeline) •
[Dataset Strategy](#-dataset-curation--harmonization) •
[Model Benchmarks](#-model-benchmarks--performance) •
[Streamlit App](#-interactive-streamlit-web-app) •
[CLI & URL Inference](#-cli-inference-engine) •
[Quickstart](#-quickstart-guide) •
[Architecture Specs](ARCHITECTURE.md) •
[Project Structure](#-project-structure)

---

</div>

## 📌 Problem Statement & Motivation

Improper waste segregation is one of the leading drivers of municipal landfill overflow, ocean contamination, and recycling stream failure. When recyclable items (such as plastics, metals, or paper) are contaminated with food organics or sorted incorrectly, entire batches of recyclable materials become unprocessable and end up in incinerators or landfills.

**Smart Waste Classifier** solves this challenge by leveraging modern transfer learning, convolutional neural networks, and real-time object detection to:
1. **Automate multi-object detection & classification** across 6 primary waste streams in real time.
2. **Eliminate human sorting error** using high-confidence deep learning predictions.
3. **Filter out-of-distribution & ambiguous noise** via **OOD Guard V1** (Shannon entropy, margin delta, test-time augmentations).
4. **Harmonize diverse waste datasets** (TACO COCO datasets + Supplemental custom datasets) into an industrial unified standard.
5. **Educate end users** with contextual eco-guidelines, bin allocation rules, and preparation tips (e.g., flattening cardboard, rinsing containers, separating organic compost).

---

## 🌟 Key Features

- 🧠 **High-Performance ResNet-18 Backbone**: Fine-tuned on a filtered, curated multi-class dataset achieving **93.34% test accuracy** (659 / 706 correct test predictions).
- 🔬 **Transfer Learning vs. Baseline Comparison**: Includes a custom 3-block 2D-CNN baseline (59.77%) and demonstrates a **+33.57% accuracy improvement** through transfer learning.
- 🎯 **Real-Time YOLOv11 Multi-Object Detection**: Integrated YOLO11n detector for multi-item localization and bounding box segmentation.
- 🔄 **Dual-Stage Cascaded Pipeline (`waste_pipeline.py`)**: Seamlessly connects YOLO object localization with fine-grained ResNet-18 material classification.
- 🛡️ **OOD Guard V1 Uncertainty Engine**: Filters non-waste/ambiguous objects using Shannon entropy, top-2 margin analysis, and Test-Time Augmentation (TTA) consistency.
- 🗂️ **Automated Dataset Fusion Engine (`detection/merge_detection_datasets.py`)**: Validates, remaps, and fuses heterogeneous datasets into a clean 6-class YOLO structure.
- 🌐 **Interactive Streamlit Web Dashboard**: Upload waste images or use a live webcam feed for instant classification, top-3 ranked probabilities, and disposal guidance.
- ⚡ **Dual-Mode CLI Inference Engine (`predict.py`)**: Predict waste categories directly from local image files or direct image URLs from the web.
- 📊 **Robust MLOps & Diagnostics**: Built-in scripts for learning rate scheduling (`ReduceLROnPlateau`), error analysis, confusion matrix plotting, and dataset inspection.
- 🚀 **Hardware Acceleration**: Automatic GPU detection (CUDA) with fallback to CPU execution.

---

## 🏗️ System Architecture

For complete in-depth technical documentation and sequence diagrams, refer to [**`ARCHITECTURE.md`**](ARCHITECTURE.md).

```mermaid
flowchart TD
    %% Global Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef guard fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef ui fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;

    subgraph DataPipeline["1. Data Ingestion & Harmonization"]
        HF[("Hugging Face Dataset\n(ddompe/waste-segregation-dataset)")]:::data --> Filter["Class Filter & Remap\n(9 Classes → 6 Target Classes)"]:::data
        Filter --> Augment["Augmentations & Transforms\n(Resize 224x224, Flip, Rotation, Affine, ImageNet Norm)"]:::data
        Augment --> Loaders["PyTorch DataLoaders\n(Train / Val / Test)"]:::data
        
        TACO[("TACO Dataset\n(COCO BBoxes)")]:::data --> Merger["merge_detection_datasets.py\n• Label Validator\n• Coordinate Normalizer\n• 6-Class Remapper"]:::data
        SUPP[("Supplemental Dataset\n(Custom Annotations)")]:::data --> Merger
        Merger --> YOLOData[("Unified YOLO Dataset\n(dataset_combined)")]:::data
    end

    subgraph Modeling["2. Model Training & Optimization"]
        Loaders --> ResNet["Pretrained ResNet-18\n(waste_resnet18_best.pth)\n93.34% Accuracy"]:::model
        Loaders --> BaseCNN["Custom 3-Block CNN Baseline\n(waste_classifier.pth)\n59.77% Accuracy"]:::model
        YOLOData --> YOLOTrain["YOLO11n Training Loop\n(detection/train_yolo.py)"]:::model
        YOLOTrain --> YOLOBest[("YOLO11 Detector Weights\n(yolo11n.pt)")]:::model
    end

    subgraph Evaluation["3. Evaluation & Diagnostics"]
        ResNet --> EvalSuite["Evaluation Suite\n(evaluate_resnet.py)"]:::model
        ResNet --> Confusion["Confusion Matrix\n(confusion_matrix_resnet.png)"]:::model
        ResNet --> ErrorAnalysis["Error Diagnostics\n(resnet_error_analysis.png)"]:::model
    end

    subgraph Deployment["4. Inference, Uncertainty Guard & UI"]
        InputImg["Input Image / Stream"]:::ui --> YOLOBest
        YOLOBest --> Crop["BBox Cropping & Aspect Padding"]:::data
        Crop --> ResNet
        ResNet --> OOD["OOD Guard V1\n• Shannon Entropy\n• Top-2 Margin Delta\n• TTA Consistency"]:::guard
        OOD --> StreamlitApp["Streamlit Web App (app.py)"]:::ui
        OOD --> CLIPredict["CLI & URL Predictor (predict.py)"]:::ui
        OOD --> CascadePipe["Multi-Object Pipeline (waste_pipeline.py)"]:::ui
    end
```

---

## 🎯 Dual-Stage Cascaded Detection Pipeline

When processing complex scenes with multiple objects or cluttered backgrounds, the **Cascaded Dual-Stage Inference Pipeline** (`waste_pipeline.py`) performs localized detection followed by fine-grained classification:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Camera Input
    participant YOLO as YOLOv11 Detector
    participant Crop as ROI Cropper & Aspect Padder
    participant ResNet as ResNet-18 Classifier
    participant OOD as OOD Guard V1
    participant UI as Visual Output

    User->>YOLO: Submit Scene Image
    YOLO-->>Crop: Detect Object Bounding Boxes [x₁, y₁, x₂, y₂]
    loop For Every Detected Bounding Box
        Crop->>Crop: Crop ROI & Pad to 224x224 RGB
        Crop->>ResNet: Feed Tensor (Batch 1, 3, 224, 224)
        ResNet-->>OOD: Compute Class Logits & Probabilities
        OOD->>OOD: Verify Entropy, Top-2 Margin & TTA Agreement
        alt Verified In-Distribution
            OOD-->>UI: Tag BBox with Waste Class + Confidence %
        else Ambiguous / OOD
            OOD-->>UI: Flag BBox as Uncertain / OOD
        end
    end
    UI->>User: Display Annotated Image with Bounding Boxes & Eco Actions
```

### Visual Pipeline Demonstrations

| YOLO Multi-Object Localization | Multi-Stage Cascade Output |
|:---:|:---:|
| ![Detected Waste](detected_waste.jpg) | ![Waste Pipeline Result](waste_pipeline_result.jpg) |

---

## 📊 Dataset Curation & Harmonization

The dataset strategy combines single-object classification datasets with multi-object detection datasets, unified under a standardized 6-class taxonomy.

### Target Waste Classes & Disposal Taxonomy

| Icon | Class Name | Target ID | Category | Disposal & Recycling Action |
|:---:|:---|:---:|:---|:---|
| 📦 | **Cardboard** | `0` | Recyclable / Dry Waste | Flatten boxes, keep clean and dry, place in cardboard recycling. |
| 🍎 | **Food Organics** | `1` | Organic / Compost | Place in organic waste bin or home composter; keep away from recyclables. |
| 🍾 | **Glass** | `2` | Recyclable Waste | Rinse containers, remove caps/lids, place in glass collection bin. |
| 🥫 | **Metal** | `3` | Recyclable Waste | Empty and rinse aluminum and tin cans; place in dry recyclables. |
| 📄 | **Paper** | `4` | Recyclable / Dry Waste | Keep dry and free of grease/oil; place in paper recycling stream. |
| 🧴 | **Plastic** | `5` | Recyclable Waste | Rinse plastic bottles and containers; check local polymer acceptance. |

### Dataset Visual Exploration

![Dataset Samples](dataset_samples.png)
![Multi-Class Dataset Samples](dataset_samples_multiple.png)

---

## 📈 Model Benchmarks & Performance

### Quantitative Model Comparison

| Metric | Custom Baseline CNN (`model.py`) | Fine-Tuned ResNet-18 (`model_resnet.py`) | Improvement |
|:---|:---:|:---:|:---:|
| **Architecture** | 3-Block Conv2D + Dense(256) | Deep Residual Network (18 layers) | Deep residual skips |
| **Pretrained Weights** | None (Trained from scratch) | ImageNet-1K (`ResNet18_Weights.DEFAULT`) | Transfer learning |
| **Input Resolution** | $128 \times 128$ | $224 \times 224$ | +75% resolution |
| **Optimization** | Adam ($lr=0.001$) | Adam ($lr=0.0001$) + `ReduceLROnPlateau` | Dynamic LR decay |
| **Test Accuracy** | **59.77%** | **93.34%** | **+33.57% 🚀** |
| **Test Loss** | ~1.1400 | **0.2114** | **-0.9286 loss** |
| **Correct / Total** | 422 / 706 | **659 / 706** | **+237 correct items** |

### Confusion Matrix (ResNet-18)

![Confusion Matrix](confusion_matrix_resnet.png)

### Error Diagnostics & Edge Cases

The error analysis module (`error_analysis.py`) identifies the highest-confidence misclassifications (e.g., reflective metal cans resembling clear glass containers or coated glossy cardboard resembling plastic):

![Error Analysis](resnet_error_analysis.png)

---

## 🖥️ Interactive Streamlit Web App

The Streamlit web application (`app.py`) provides an intuitive UI for users and municipal sorting operators:

- 📤 **Instant Image Upload**: Supports `.jpg`, `.jpeg`, and `.png` image formats.
- 🎯 **Prediction with Confidence Gauge**: Live confidence percentage with dynamic status indicators (`High`, `Moderate`, `Low`).
- 💡 **Actionable Eco-Advice**: Step-by-step instructions on proper bin placement and item preparation.
- 📊 **Probability Distribution**: Full confidence breakdown across all 6 classes and Top-3 ranked results with visual progress bars.
- 🛡️ **OOD Guard V1**: Real-time out-of-distribution detection to flag non-waste or ambiguous objects.

```bash
streamlit run app.py
```

---

## 💻 CLI Inference Engine

Run standalone predictions from your terminal without opening a web browser:

### 1. Predict from a Local Image

```bash
python predict.py "path/to/waste_image.jpg"
```

### 2. Predict from a Web Image URL

```bash
python predict.py "https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=500"
```

---

## 🚀 Quickstart Guide

### Prerequisites

- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA drivers for accelerated training and inference

### 1. Clone the Repository

```bash
git clone https://github.com/Snigdha-0210/Smart_Waste_Classifier.git
cd Smart_Waste_Classifier
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
Smart_Waste_Classifier/
│
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT Open-Source License
├── README.md                          # Comprehensive project documentation
├── ARCHITECTURE.md                    # Detailed system architecture specifications & diagrams
├── DATASET_PLAN.md                    # Data curation, filtering & augmentation strategy
├── requirements.txt                   # Production Python dependencies
│
├── app.py                             # Interactive Streamlit Web Application (with OOD Guard V1)
├── model_resnet.py                    # ResNet-18 Transfer Learning Model Definition
├── model.py                           # Baseline Custom 3-Block CNN Model Definition
├── prepare_data.py                    # Dataset loading, filtering, and DataLoader pipeline
├── train_resnet.py                    # ResNet-18 training script with LR scheduler & checkpointing
├── train.py                           # Baseline CNN training script
├── evaluate_resnet.py                 # ResNet-18 quantitative test evaluation
├── evaluate.py                        # Baseline CNN test evaluation
├── predict.py                         # Standalone CLI & URL inference engine with OOD Guard
├── detect_waste.py                    # YOLO real-time multi-object detection script
├── waste_pipeline.py                  # End-to-end cascaded YOLO detection + ResNet-18 pipeline
├── prepare_detection_data.py          # TACO-to-YOLO dataset converter V1
├── prepare_detection_data_v2.py       # TACO-to-YOLO converter V2 with stratified split protection
├── train_yolo.py                      # YOLO custom training pipeline
├── test_yolo.py                       # YOLO detection inference and benchmarking script
├── project_status.py                  # Diagnostic system health & environment check script
├── confusion_matrix_resnet.py         # Confusion matrix and classification report generator
├── confusion_matrix.py                # Baseline CNN confusion matrix generator
├── error_analysis.py                  # High-confidence error analysis & visualization script
├── inspect_dataset.py                 # Dataset split and class distribution inspection
├── visualize_dataset.py               # Single sample visualization script
├── visualize_dataset_multiple.py      # Multi-sample grid visualizer
│
├── detection/                         # Detection Subsystem & Harmonization
│   ├── merge_detection_datasets.py    # Multi-dataset fusion, remapping & validation script
│   ├── train_yolo.py                  # YOLOv11 training script on combined 6-class dataset
│   ├── download_taco_images.py        # Automated TACO dataset image downloader
│   └── data.yaml                      # YOLO dataset configuration
│
├── waste_resnet18_best.pth            # Trained ResNet-18 weights (93.34% Test Accuracy)
├── waste_classifier.pth               # Trained Baseline CNN weights (59.77% Test Accuracy)
├── yolo11n.pt                         # YOLOv11 neural network weights
│
├── dataset_samples.png                # Dataset sample preview image
├── dataset_samples_multiple.png       # Comprehensive multi-class sample grid
├── confusion_matrix_resnet.png        # ResNet-18 confusion matrix plot
├── confusion_matrix.png               # Baseline CNN confusion matrix plot
├── resnet_error_analysis.png          # ResNet-18 error analysis visual grid
├── detected_waste.jpg                 # YOLO object detection output preview
├── waste_pipeline_result.jpg          # Multi-object detection + classification visual result
└── thumbnail.jpg                      # Project banner & social preview thumbnail
```

---

## 🔮 Future Roadmap

- [x] **YOLO Object Detection Prototype**: Added YOLO object detection testing pipeline (`detect_waste.py`, `waste_pipeline.py`).
- [x] **OOD Guard System**: Implemented Out-of-Distribution uncertainty handling in Streamlit.
- [x] **Dataset Harmonization Engine**: Multi-dataset fusion script (`detection/merge_detection_datasets.py`) for combined detection training.
- [ ] **Edge Deployment**: Optimize models with TensorRT / ONNX Runtime for deployment on embedded devices (e.g., Raspberry Pi, NVIDIA Jetson) in smart bin hardware.
- [ ] **Mobile Application**: Build a Flutter / React Native camera companion app for on-the-go waste sorting.
- [ ] **Expanded Class Taxonomy**: Include E-waste (electronic waste), hazardous chemicals, and battery categories with dedicated disposal workflows.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model, add features, or refine the web app:

1. Fork the project repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

## 👩‍💻 Author & Acknowledgements

Developed with ❤️ by **[Snigdha](https://github.com/Snigdha-0210)**.

- Dataset provided by [`ddompe/waste-segregation-dataset`](https://huggingface.co/datasets/ddompe/waste-segregation-dataset) on Hugging Face and TACO dataset.
- Built with [PyTorch](https://pytorch.org/), [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and [Streamlit](https://streamlit.io/).
