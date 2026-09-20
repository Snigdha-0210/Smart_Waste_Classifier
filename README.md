<div align="center">

# ♻️ Smart Waste Classifier
### Enterprise-Grade Multi-Stage Deep Learning Platform for Solid Waste Segregation, Multi-Object Detection & Eco-Disposal Guidance

![Project Banner](assets/thumbnail.jpg)

[![CI](https://github.com/Snigdha-0210/Smart_Waste_Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Snigdha-0210/Smart_Waste_Classifier/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-11n_Detector-00FFFF.svg?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ddompe/waste-segregation-dataset)
[![ResNet-18](https://img.shields.io/badge/Model-ResNet18-success.svg?style=for-the-badge&logo=deepnote&logoColor=white)](https://pytorch.org/vision/main/models/resnet.html)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-93.34%25-brightgreen.svg?style=for-the-badge&logo=checkmarx&logoColor=white)](#-model-benchmarks--performance)
[![CUDA](https://img.shields.io/badge/Hardware-CUDA_Accelerated-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>An intelligent, end-to-end computer vision ecosystem that localizes and classifies municipal solid waste across recyclable and organic streams with 93.34% accuracy, equipped with OOD Guard V1 uncertainty filtering, a multi-source dataset evolution engine (V1–V6), and real-time eco-guidance.</strong>
</p>

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Cascaded Dual-Stage Pipeline](#-cascaded-dual-stage-detection-pipeline) •
[Dataset Evolution (V1–V6)](#-dataset-strategy--multi-version-evolution-v1v6) •
[Model Benchmarks](#-model-benchmarks--performance) •
[YOLOv11 Detection Metrics](#-yolov11-object-detection-performance) •
[OOD Guard V1](#️-ood-guard-v1-uncertainty-engine) •
[Streamlit Web App](#-interactive-streamlit-web-dashboard) •
[CLI & URL Inference](#-universal-cli--url-inference-engine) •
[Quickstart Guide](#-quickstart-guide) •
[Architecture Specs](ARCHITECTURE.md) •
[Project Structure](#-project-structure)

---

</div>

## 📌 Problem Statement & Motivation

Improper solid waste segregation is one of the leading global drivers of landfill overflow, ocean pollution, and recycling stream failure. When recyclable items (such as plastics, metals, or paper) are contaminated with food organics or sorted into incorrect bins, entire batches of recyclable material become unprocessable and end up in incinerators or landfills.

Standard single-stage image classifiers fail in real-world environments because:
1. **Scene Clutter & Multi-Object Overlaps**: Real-world waste bins, dumps, and conveyor belts contain multiple overlapping items rather than a single centered object.
2. **Ambiguous Catch-All Classes**: Generic classes (e.g., *Miscellaneous Trash*) act as label noise and degrade gradient convergence.
3. **Silent Overconfidence on Out-of-Domain Inputs**: Deep neural networks assign high softmax probabilities to foreign objects (e.g., electronics, furniture, animals) when forced to pick from a closed set.

> [!IMPORTANT]
> **Smart Waste Classifier** resolves these challenges by uniting single-object classification and multi-object scene detection into a robust, two-stage computer vision platform that:
> - **Automates multi-object localization and classification** across primary recyclable and compostable waste streams.
> - **Eliminates human sorting errors** through residual transfer learning (**93.34% test accuracy**).
> - **Guards against silent out-of-domain failures** using **OOD Guard V1** (Shannon entropy, margin delta, test-time augmentations).
> - **Evolves datasets across 6 generations (V1–V6)** from TACO COCO annotations to synthetic-real fusion datasets.
> - **Provides actionable eco-guidance** with step-by-step preparation advice (e.g., flattening cardboard, rinsing containers, compost segregation).

---

## 🌟 Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 High-Performance ResNet-18 Backbone</h3>
      <p>Fine-tuned residual network with <code>ReduceLROnPlateau</code> dynamic scheduling, achieving <strong>93.34% test accuracy</strong> and a <strong>+33.57% improvement</strong> over a 3-block CNN baseline.</p>
    </td>
    <td width="50%">
      <h3>🎯 Real-Time YOLOv11 Multi-Object Detection</h3>
      <p>Custom-trained YOLO11n object detector capable of localizing multiple overlapping waste items in cluttered scenes with real-time inference latency (~8.4 ms on RTX 4060 GPU).</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔄 Cascaded Dual-Stage Inference Pipeline</h3>
      <p>Seamlessly integrates YOLO localization with fine-grained ResNet-18 classification for high-precision ROI analysis, clutter separation, and visual bounding box overlays.</p>
    </td>
    <td width="50%">
      <h3>🛡️ OOD Guard V1 Uncertainty Engine</h3>
      <p>Tri-metric uncertainty filtering using normalized Shannon entropy, top-2 probability margin delta, and 5-pass Test-Time Augmentation (TTA) consistency gating.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🗂️ Multi-Version Dataset Engine (V1–V6)</h3>
      <p>Automated pipeline supporting TACO conversion (V1/V2), multi-source dataset fusion (V3), 10-class synthetic SynWasteNet (V4), and balanced synthetic-real blends (V5/V6).</p>
    </td>
    <td width="50%">
      <h3>🌐 Interactive Streamlit Web Dashboard</h3>
      <p>Full-featured web application with multi-page navigation, instant image upload, probability distribution charts, OOD diagnostic drawers, and session analytics.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>💻 Universal CLI & URL Inference Engine</h3>
      <p>Zero-configuration prediction script supporting local file paths, web image URLs, terminal ASCII confidence bars, and formatted eco-disposal advice.</p>
    </td>
    <td width="50%">
      <h3>🧪 Production-Grade MLOps & CI/CD</h3>
      <p>Automated GitHub Actions CI pipeline testing across Python 3.10, 3.11, and 3.12, paired with a comprehensive Pytest unit test suite.</p>
    </td>
  </tr>
</table>

---

## 🏗️ System Architecture

The following diagram illustrates the complete architectural topology from raw multi-source data ingestion to user-facing inference interfaces:

```mermaid
flowchart TD
    %% Styling definitions
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef guard fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef ui fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;

    subgraph DataPipeline["1. Data Ingestion & Multi-Source Harmonization"]
        HF[("Hugging Face Hub\n(ddompe/waste-segregation-dataset)")]:::data --> Filter["Class Filtering & Remapping\n(9 Classes → 6 Target Classes)"]:::data
        Filter --> Augment["Augmentations & Transforms\n(Resize 224x224, Flip, Rotation, Affine, Norm)"]:::data
        Augment --> Loaders["PyTorch DataLoaders\n(Train: 2246 / Val: 562 / Test: 706)"]:::data
        
        TACO[("TACO Dataset\n(COCO Annotations)")]:::data --> Merger["merge_detection_datasets.py\n• Label BBox Validator\n• Coordinate Normalizer\n• 6-Class Remapper"]:::data
        SUPP[("Supplemental Dataset\n(Custom Annotations)")]:::data --> Merger
        SYNTH[("SynWasteNet Synthetic V4\n(10-Class YOLO Renders)")]:::data --> MixedGen["create_mixed_v6_dataset.py\n• Multi-Class Stratification\n• Synthetic + Real Fusion"]:::data
        Merger --> BalancedV2["create_balanced_dataset_v2.py\n(Density Cluster Limiter)"]:::data
        BalancedV2 --> MixedGen
    end

    subgraph Modeling["2. Model Training & Optimization Core"]
        Loaders --> ResNet["Fine-Tuned ResNet-18\n(waste_resnet18_best.pth)\n93.34% Accuracy"]:::model
        Loaders --> BaseCNN["Custom 3-Block CNN Baseline\n(waste_classifier.pth)\n59.77% Accuracy"]:::model
        MixedGen --> YOLOTrain["YOLO11n Training Loops\n(train_yolo.py / train_yolo_v4.py / train_yolo_v6.py)\nAMP, Mosaic, RTX 4060 GPU"]:::model
        YOLOTrain --> YOLOBest[("YOLO11 Detector Weights\n(best.pt / yolo11n.pt)")]:::model
    end

    subgraph Diagnostics["3. Diagnostics & MLOps Suite"]
        ResNet --> EvalSuite["Evaluation Suite\n(evaluate_resnet.py)"]:::model
        ResNet --> Confusion["Confusion Matrix\n(confusion_matrix_resnet.png)"]:::model
        ResNet --> ErrorAnalysis["Error Diagnostics\n(resnet_error_analysis.png)"]:::model
        YOLOBest --> YOLOMetrics["YOLO Training Curves\n(assets/yolo_training_results.png)"]:::model
    end

    subgraph InferenceCascade["4. Hybrid Cascaded Inference & OOD Guard"]
        InputImg["Input Scene Image / Feed"]:::ui --> YOLOBest
        YOLOBest --> Cropper["BBox Cropping & Aspect Padding\n(Resize to 224x224 RGB)"]:::data
        Cropper --> ResNet
        ResNet --> OOD["OOD Guard V1 Engine\n• Shannon Entropy H_norm <= 0.72\n• Top-2 Margin Delta Δ >= 0.18\n• 5-Pass TTA Agreement >= 60%"]:::guard
    end

    subgraph Presentation["5. Deployment & User Interfaces"]
        OOD --> StreamlitApp["Streamlit Web App\n(app.py)"]:::ui
        OOD --> CLIPredict["Universal CLI & URL Engine\n(predict.py)"]:::ui
        OOD --> CascadePipe["Multi-Object Pipeline Script\n(waste_pipeline.py)"]:::ui
        StreamlitApp --> EcoOutput["Verified Waste Category + Actionable Eco-Guidance"]:::output
    end
```

---

## 🎯 Cascaded Dual-Stage Detection Pipeline

When processing complex scenes with multiple objects or cluttered backgrounds, the **Cascaded Dual-Stage Inference Pipeline** (`waste_pipeline.py`) performs localized detection followed by fine-grained classification:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Camera Feed
    participant YOLO as Stage 1: YOLOv11 Detector
    participant Crop as ROI Cropper & Aspect Padder
    participant ResNet as Stage 2: ResNet-18 Classifier
    participant OOD as OOD Guard V1 Engine
    participant UI as Visual Canvas & Dashboard

    User->>YOLO: Submit Complex Waste Scene Image
    Note over YOLO: Multi-object feature extraction & Non-Maximum Suppression (NMS)
    YOLO-->>Crop: Extract Bounding Boxes [x₁, y₁, x₂, y₂] + Detection Confidence
    
    loop For Every Detected Bounding Box
        Crop->>Crop: Crop ROI with Safety Margin & Pad to 224x224 RGB
        Crop->>ResNet: Forward Pass Tensor (1, 3, 224, 224)
        ResNet-->>OOD: Compute Logits & Softmax Probabilities [p₀, ..., p₅]
        OOD->>OOD: Evaluate Shannon Entropy, Margin Delta & TTA Agreement
        alt Verified In-Distribution
            OOD-->>UI: Tag BBox with Waste Class + Confidence % + Eco Advice
        else Ambiguous / Out-of-Distribution
            OOD-->>UI: Flag BBox as Uncertain / OOD Object
        end
    end
    
    UI->>User: Display Annotated Image with Bounding Boxes & Disposal Actions
```

### Visual Pipeline Demonstrations

| Stage 1: YOLO Multi-Object Localization | Stage 2: Cascaded Classification Output |
|:---:|:---:|
| ![Detected Waste](assets/detected_waste.jpg) | ![Waste Pipeline Result](assets/waste_pipeline_result.jpg) |

---

## 🗂️ Dataset Strategy & Multi-Version Evolution (V1–V6)

### 1. Standardized 6-Class Taxonomy

The primary classification and initial detection models adhere to a standardized 6-class taxonomy. Ambiguous classes (*Miscellaneous Trash*, *Textile Trash*, *Vegetation*) were pruned to maximize gradient convergence:

| Icon | Class Name | Target ID | Waste Category | Disposal & Preparation Protocol |
|:---:|:---|:---:|:---|:---|
| 📦 | **Cardboard** | `0` | Recyclable / Dry Waste | Flatten boxes, remove tape/staples, keep clean and dry; place in blue recycling bin. |
| 🍎 | **Food Organics** | `1` | Organic / Compost | Collect peels, rinds, and food scraps; place in green compost bin. Keep free of plastics. |
| 🍾 | **Glass** | `2` | Recyclable Waste | Empty and rinse bottles/jars, remove metal caps/corks; place in designated glass bin. |
| 🥫 | **Metal** | `3` | Recyclable Waste | Empty aluminum and tin cans, rinse residues, lightly crush cans; place in dry recyclables. |
| 📄 | **Paper** | `4` | Recyclable / Dry Waste | Keep dry and unsoiled; place newspapers, office paper, and clean bags in paper recycling. |
| 🧴 | **Plastic** | `5` | Recyclable Waste | Empty and rinse PET bottles/containers, check resin ID code; place in plastic recycling. |

### 2. Multi-Version Detection Dataset Evolution

| Version | Script / Generator | Data Sources & Curation Method | Target Classes | Key Innovation |
|:---:|:---|:---|:---:|:---|
| **V1** | [`prepare_detection_data.py`](prepare_detection_data.py) | Raw TACO Dataset (COCO annotations) converted to standard YOLO format. | 6 Classes | Establishes baseline bounding box coordinates and image downloaders. |
| **V2** | [`prepare_detection_data_v2.py`](prepare_detection_data_v2.py)<br>[`create_balanced_dataset_v2.py`](detection/create_balanced_dataset_v2.py) | Stratified split protected TACO + High-density Food Organics balancer. | 6 Classes | Caps max food scraps per image at 20 to prevent anchor bias. |
| **V3** | [`merge_detection_datasets.py`](detection/merge_detection_datasets.py)<br>[`create_dataset_v3.py`](create_dataset_v3.py) | Fusion of TACO COCO dataset + Supplemental real-world custom dataset. | 6 Classes | Bounding box coordinate validation, non-zero area checks, prefixing. |
| **V4** | [`create_final_v4_dataset.py`](detection/create_final_v4_dataset.py)<br>[`train_yolo_v4.py`](detection/train_yolo_v4.py) | Synthetic Outdoor Waste YOLO Dataset (`SynWasteNet`). | **10 Classes** | Expands taxonomy to Battery, E-Waste, Cloth/Textile, Other Waste. |
| **V5** | [`create_mixed_v5_dataset.py`](detection/create_mixed_v5_dataset.py) | V4 Synthetic data combined with a controlled subset ($\le 1500$) of real V3 data. | **10 Classes** | Reduces domain gap between synthetic renders and real photographs. |
| **V6** | [`create_mixed_v6_dataset.py`](detection/create_mixed_v6_dataset.py)<br>[`train_yolo_v6.py`](detection/train_yolo_v6.py) | Balanced Multi-Source Fusion (V4 Synthetic + Stratified Real V3). | **10 Classes** | Minority class equalization, deduplication, production mixed training. |

### 3. Visual Dataset Exploration

| Single Sample Preview | Multi-Class Comprehensive Grid |
|:---:|:---:|
| ![Dataset Samples](assets/dataset_samples.png) | ![Multi-Class Dataset Samples](assets/dataset_samples_multiple.png) |

---

## 📈 Model Benchmarks & Performance

### Quantitative Model Comparison

| Metric | Custom Baseline CNN (`model.py`) | Fine-Tuned ResNet-18 (`model_resnet.py`) | Net Gain / Impact |
|:---|:---:|:---:|:---:|
| **Architecture** | 3-Block Conv2D + Dense(256) | Deep Residual Network (18 layers) | Deep residual skip connections |
| **Pretrained Weights** | None (Trained from scratch) | ImageNet-1K (`ResNet18_Weights.DEFAULT`) | Transfer learning feature reuse |
| **Input Resolution** | $128 \times 128$ | $224 \times 224$ | +75% higher spatial resolution |
| **Optimization** | Adam ($lr=0.001$) | Adam ($lr=0.0001$) + `ReduceLROnPlateau` | Dynamic LR plateau decay |
| **Test Accuracy** | **59.77%** | **93.34%** | **+33.57% 🚀** |
| **Test Loss** | ~1.1400 | **0.2114** | **-0.9286 loss reduction** |
| **Correct / Total** | 422 / 706 | **659 / 706** | **+237 correct test items** |
| **Inference Latency** | ~3.2 ms (GPU) | ~5.8 ms (GPU) | Real-time capable (>170 FPS) |

### Per-Class Performance Breakdown (ResNet-18)

```text
               precision    recall  f1-score   support

    Cardboard       0.94      0.93      0.94       118
Food Organics       0.97      0.96      0.96       120
        Glass       0.91      0.92      0.91       115
        Metal       0.92      0.90      0.91       112
        Paper       0.91      0.93      0.92       119
      Plastic       0.95      0.96      0.95       122

     accuracy                           0.93       706
    macro avg       0.93      0.93      0.93       706
 weighted avg       0.93      0.93      0.93       706
```

### Confusion Matrix & Error Analysis

| ResNet-18 Confusion Matrix | High-Confidence Error Diagnostics |
|:---:|:---:|
| ![ResNet-18 Confusion Matrix](assets/confusion_matrix_resnet.png) | ![ResNet-18 Error Analysis](assets/resnet_error_analysis.png) |

> [!NOTE]
> **Error Diagnostics**: The most frequent misclassifications occur between reflective metal foil vs transparent plastic containers (12 samples) and cardstock paper vs cardboard packaging (7 samples), both sharing high visual similarity in ambient lighting.

---

## 🎯 YOLOv11 Object Detection Performance

The custom YOLO11n detector was trained for **100 epochs** on the combined multi-object dataset with Automated Mixed Precision (AMP) and Mosaic augmentations:

| Metric | Score / Benchmark | Notes |
|:---|:---:|:---|
| **mAP @ 0.50** | **57.4%** | Multi-object localization across waste classes |
| **mAP @ 0.50:0.95** | **39.2%** | Stringent IoU threshold evaluation |
| **Precision** | **65.0%** | Bounding box prediction accuracy |
| **Recall** | **53.7%** | Object detection coverage |
| **Inference Speed** | **~8.4 ms** | RTX 4060 GPU (~119 FPS) |

### YOLO Training Curves & Diagnostics

| Training Metrics & Loss Curves | Detection Validation Predictions |
|:---:|:---:|
| ![YOLO Training Results](assets/yolo_training_results.png) | ![YOLO Validation Predictions](assets/yolo_val_predictions.jpg) |

| YOLO Confusion Matrix | Precision-Recall (PR) Curve |
|:---:|:---:|
| ![YOLO Confusion Matrix](assets/yolo_confusion_matrix.png) | ![YOLO PR Curve](assets/yolo_pr_curve.png) |

---

## 🛡️ OOD Guard V1: Uncertainty Engine

To prevent silent misclassification when users upload non-waste items or ambiguous photographs, **OOD Guard V1** executes a multi-signal uncertainty quantification pipeline:

```mermaid
flowchart TD
    In[Input Image Crop] --> BasePred[Standard Forward Pass]
    In --> TTAPass1["TTA Pass 1: Horizontal Flip"]
    In --> TTAPass2["TTA Pass 2: Center Crop (256→224)"]
    In --> TTAPass3["TTA Pass 3: Color Jitter"]
    In --> TTAPass4["TTA Pass 4: Random Crop + Padding"]
    
    BasePred --> SoftmaxCalc["Compute Softmax: p = [p₀, ..., p₅]"]
    SoftmaxCalc --> Metric1["1. Top-1 Confidence: max(p)"]
    SoftmaxCalc --> Metric2["2. Top-2 Margin Delta: Δ = p₁ - p₂"]
    SoftmaxCalc --> Metric3["3. Normalized Shannon Entropy:\nH_norm = -Σ(pᵢ ln pᵢ) / ln(6)"]
    
    TTAPass1 --> TTAVote[TTA Prediction Agreement %]
    TTAPass2 --> TTAVote
    TTAPass3 --> TTAVote
    TTAPass4 --> TTAVote
    
    Metric1 --> Gate{"OOD Guard Decision Gating"}
    Metric2 --> Gate
    Metric3 --> Gate
    TTAVote --> Gate
    
    Gate --> |"Confidence >= 0.70<br/>Margin Δ >= 0.18<br/>Entropy <= 0.72<br/>TTA Agreement >= 60%"| Approved["🟢 VERIFIED IN-DISTRIBUTION<br/>(Display Waste Class & Eco Actions)"]
    Gate --> |"Condition Failed"| Flagged["🔴 OUT-OF-DISTRIBUTION / UNCERTAIN<br/>(Flag Ambiguity & Request Manual Check)"]
```

### Mathematical Formulations

1. **Normalized Shannon Entropy ($H_{norm}$)**:
   $$H(\mathbf{p}) = -\sum_{i=1}^K p_i \ln(p_i), \quad H_{norm} = \frac{H(\mathbf{p})}{\ln(K)}$$
   *Threshold*: $H_{norm} \le 0.72$ (ensures low prediction uncertainty).

2. **Top-2 Probability Margin Delta ($\Delta$)**:
   $$\Delta = p_{(1)} - p_{(2)}$$
   *Threshold*: $\Delta \ge 0.18$ (ensures decisive separation from competitor classes).

3. **Test-Time Augmentation (TTA) Agreement ($A_{TTA}$)**:
   $$A_{TTA} = \frac{1}{N} \sum_{n=1}^N \mathbb{I}\left(\arg\max(\mathbf{p}^{(n)}) = \arg\max(\mathbf{p}^{(0)})\right)$$
   *Threshold*: $A_{TTA} \ge 60\%$ across 5 stochastic geometric & photometric views.

---

## 🖥️ Interactive Streamlit Web Dashboard

The Streamlit web application ([`app.py`](app.py)) provides a dashboard for sorting operators, recycling managers, and citizens:

- 📤 **Instant Image Upload**: Supports `.jpg`, `.jpeg`, and `.png` image formats.
- 🎯 **Prediction with Confidence Gauge**: Live confidence percentage with dynamic status indicators (`SUPPORTED`, `UNCERTAIN`, `UNSUPPORTED`).
- 💡 **Actionable Eco-Advice**: Step-by-step instructions on proper bin placement and preparation.
- 📊 **Probability Distribution**: Full confidence breakdown across all 6 classes with visual progress bars.
- 🔍 **OOD Analysis Drawer**: Real-time inspection of Top-1 confidence, Top-2 margin, normalized entropy, and augmentation agreement.
- 🕘 **Persistent History & Analytics**: Dashboard metrics tracking total classifications, category distributions, and confidence trends.

```bash
streamlit run app.py
```

---

## 💻 Universal CLI & URL Inference Engine

Run predictions directly from your terminal using [`predict.py`](predict.py):

### 1. Predict from a Local Image File

```bash
python predict.py "path/to/waste_sample.jpg"
```

**Example Output:**
```text
======================================================================
SMART WASTE CLASSIFIER — INFERENCE ENGINE
======================================================================
Image: path/to/waste_sample.jpg
Model: ResNet-18 (CUDA GPU)

PREDICTION RESULTS:
  Category   : Plastic
  Confidence : 96.84%
  OOD Status : SUPPORTED (🟢 In-Distribution)

TOP-3 PREDICTIONS:
  1. Plastic       : 96.84%  [████████████████████]
  2. Metal         :  2.14%  [█                   ]
  3. Glass         :  0.61%  [                    ]

ECO-DISPOSAL GUIDANCE:
  • Bin: Yellow / Plastic Recycling Bin
  • Action: Rinse thoroughly, remove liquid residues, check resin ID code.
======================================================================
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

### 4. Run Applications & Pipelines

```bash
# Launch Streamlit Web App
streamlit run app.py

# Run Cascaded Detection + Classification Pipeline
python waste_pipeline.py

# Run System Health & Diagnostic Suite
python project_status.py

# Run Pytest Unit Tests
pytest tests/ -v
```

---

## 📁 Project Structure

```text
Smart_Waste_Classifier/
│
├── .github/                           # GitHub Configuration & CI/CD
│   ├── workflows/ci.yml               # GitHub Actions CI matrix (Python 3.10, 3.11, 3.12)
│   ├── ISSUE_TEMPLATE/                # Issue templates (bug report, feature request)
│   └── PULL_REQUEST_TEMPLATE.md       # Standardized PR submission template
│
├── .gitignore                         # Comprehensive git exclusion rules
├── LICENSE                            # MIT Open-Source License
├── pyproject.toml                     # Project packaging metadata & test configurations
├── README.md                          # Repository documentation & user guide
├── ARCHITECTURE.md                    # System architecture & mathematical specifications
├── DATASET_PLAN.md                    # Data curation strategy & class filtering rationale
├── CONTRIBUTING.md                    # Open-source contribution guidelines
├── CODE_OF_CONDUCT.md                 # Contributor Covenant Code of Conduct
├── SECURITY.md                        # Security vulnerability reporting policy
├── requirements.txt                   # Production Python dependencies
├── project_status.py                  # System health & checkpoint diagnostic suite
│
├── app.py                             # Interactive Streamlit Web Application (with OOD Guard V1)
├── app_backup.py                      # Backup copy of initial Streamlit application
├── predict.py                         # Standalone CLI & web URL inference engine
├── waste_pipeline.py                  # Dual-stage cascaded YOLO detection + ResNet-18 pipeline
├── detect_waste.py                    # Standalone YOLO detection script
├── test_yolo.py                       # YOLO benchmarking and test inference runner
│
├── model_resnet.py                    # ResNet-18 Transfer Learning model definition
├── model.py                           # Baseline Custom 3-Block CNN model definition
├── prepare_data.py                    # Dataset loading, 6-class filtering, and PyTorch DataLoaders
├── train_resnet.py                    # ResNet-18 trainer with LR plateau scheduler & checkpointing
├── train.py                           # Baseline CNN trainer
├── evaluate_resnet.py                 # ResNet-18 quantitative test evaluation suite
├── evaluate.py                        # Baseline CNN test evaluation
├── confusion_matrix_resnet.py         # ResNet-18 confusion matrix & classification report generator
├── confusion_matrix.py                # Baseline CNN confusion matrix generator
├── error_analysis.py                  # High-confidence misclassification visual inspection
├── inspect_dataset.py                 # Dataset split and label distribution verification
├── visualize_dataset.py               # Single sample inspector
├── visualize_dataset_multiple.py      # Multi-sample grid visualizer
│
├── detection/                         # Object Detection Subsystem (V1–V6 Evolution)
│   ├── merge_detection_datasets.py    # Multi-dataset fusion, coordinate normalization & validation
│   ├── create_balanced_dataset_v2.py  # High-density Food Organics balancer & dataset_v2 builder
│   ├── create_final_v4_dataset.py     # 10-Class Synthetic V4 dataset builder & YAML generator
│   ├── create_mixed_v5_dataset.py     # Mixed V5 dataset builder (Synthetic + Real-world subset)
│   ├── create_mixed_v6_dataset.py     # Balanced Mixed V6 dataset builder (10-class fusion)
│   ├── analyze_combined_dataset.py    # Class balance and object density analyzer
│   ├── analyze_food_images.py         # Food Organics density distribution inspector
│   ├── train_yolo.py                  # YOLOv11 training script on combined 6-class dataset
│   ├── train_yolo_v4.py               # YOLOv11 training script on 10-class V4 dataset
│   ├── train_yolo_v6.py               # YOLOv11 training script on balanced 10-class V6 dataset
│   ├── download_taco_images.py        # Automated TACO image downloader
│   └── data.yaml                      # YOLO dataset configuration
│
├── prepare_detection_data.py          # TACO COCO-to-YOLO dataset converter V1
├── prepare_detection_data_v2.py       # TACO-to-YOLO converter V2 with stratified split protection
│
├── tests/                             # Automated Pytest Test Suite
│   ├── test_model_architectures.py    # CNN & ResNet forward pass, backward pass & output shapes
│   ├── test_ood_metrics.py            # Shannon entropy, margin delta & OOD decision logic
│   └── test_taxonomy_and_config.py    # Taxonomy consistency, YAML format & disposal mapping
│
├── assets/                            # Curated Repository Visual Assets & Diagnostic Plots
│   ├── thumbnail.jpg                  # Project header & social preview banner
│   ├── dataset_samples.png            # Dataset sample preview visualization
│   ├── dataset_samples_multiple.png   # Multi-class dataset sample grid
│   ├── confusion_matrix_resnet.png    # ResNet-18 confusion matrix heatmap
│   ├── confusion_matrix_baseline.png  # Baseline CNN confusion matrix heatmap
│   ├── resnet_error_analysis.png      # ResNet-18 misclassification diagnostic chart
│   ├── detected_waste.jpg             # YOLO multi-object detection output preview
│   ├── waste_pipeline_result.jpg      # Cascaded detection + classification visual result
│   ├── yolo_training_results.png      # YOLO11n 100-epoch loss and mAP curves
│   ├── yolo_confusion_matrix.png      # YOLO11n detection confusion matrix
│   ├── yolo_val_predictions.jpg       # YOLO11n validation batch detection results
│   ├── yolo_f1_curve.png              # YOLO11n F1-Confidence curve
│   └── yolo_pr_curve.png              # YOLO11n Precision-Recall curve
│
├── prediction_sample/                 # Visual predictions with localized bounding boxes
├── real_world_test/                   # Real-world challenging municipal waste benchmark test set
│
├── waste_resnet18_best.pth            # Trained ResNet-18 weights (93.34% Test Accuracy)
├── waste_classifier.pth               # Trained Baseline CNN weights (59.77% Test Accuracy)
└── yolo11n.pt                         # YOLOv11 neural network weights
```

---

## 🔮 Future Roadmap

- [x] **Custom 3-Block CNN Baseline**: Built and evaluated baseline model (59.77%).
- [x] **ResNet-18 Transfer Learning**: Achieved 93.34% accuracy with dynamic LR scheduling.
- [x] **OOD Guard V1**: Implemented entropy, margin, and test-time augmentation gating.
- [x] **YOLOv11 Multi-Object Detection**: Trained 100 epochs on unified multi-object dataset.
- [x] **Cascaded Dual-Stage Pipeline**: Integrated YOLO localization with ResNet classification.
- [x] **Multi-Source Dataset Evolution (V1–V6)**: Automated TACO conversion, balanced V2, fused V3, synthetic V4, and balanced mixed V6 pipelines.
- [x] **CI/CD & Unit Test Suite**: GitHub Actions CI with Pytest covering architectures, OOD metrics, and taxonomy.
- [ ] **Streamlit Detection Tab**: Embed the cascaded YOLO + ResNet detection pipeline directly in the web UI.
- [ ] **Edge Deployment**: Export models with TensorRT / ONNX Runtime for deployment on embedded smart bin hardware (NVIDIA Jetson / Raspberry Pi).
- [ ] **Live Webcam Stream**: Add real-time video stream classification and object counting in Streamlit.
- [ ] **10-Class Unified UI**: Upgrade Streamlit frontend to support the expanded 10-class taxonomy (including E-waste, batteries, and textiles).

---

## 🤝 Contributing

Contributions are welcome! If you would like to improve the model, expand the dataset, or refine the web dashboard:

1. Read our [**Contributing Guidelines**](CONTRIBUTING.md) for environment setup, code style, and test commands.
2. Review our [**Code of Conduct**](CODE_OF_CONDUCT.md) and [**Security Policy**](SECURITY.md).
3. Fork the repository and create your branch (`git checkout -b feature/AmazingFeature`).
4. Run unit tests (`pytest tests/ -v`) to ensure all tests pass.
5. Open a Pull Request using the [PR Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

## 👩‍💻 Author & Acknowledgements

Developed with ❤️ by **[Snigdha](https://github.com/Snigdha-0210)**.

- Datasets provided by [`ddompe/waste-segregation-dataset`](https://huggingface.co/datasets/ddompe/waste-segregation-dataset) on Hugging Face, the [TACO Dataset](http://tacodataset.org/), and [SynWasteNet](https://github.com/).
- Built with [PyTorch](https://pytorch.org/), [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and [Streamlit](https://streamlit.io/).
