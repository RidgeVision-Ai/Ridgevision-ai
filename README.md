---
title: RidgeVision AI
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🩸 RidgeVision AI — Fingerprint-Based Blood Group Prediction

<div align="center">

![RidgeVision AI](https://img.shields.io/badge/RidgeVision-AI-darkgreen?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Deployed-yellow?style=for-the-badge&logo=huggingface)
![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-~89.5--90%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey?style=for-the-badge)

**RidgeAttenFusionNet** — A dual-branch deep learning ensemble for non-invasive ABO/Rh blood group estimation from fingerprint dermatoglyphics, with Grad-CAM explainability.

🔗 **Live Demo:** [Hugging Face Spaces (RidgeVision AI)](https://sravaninanubala-ridgevision-ai.hf.space/)
📂 **Repository:** [https://github.com/NanubalaSravani/Ridgevision-ai](https://github.com/NanubalaSravani/Ridgevision-ai)

> ⚠️ **Research Prototype v1.0** — Not intended for medical diagnosis or clinical use.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Workflow Pipeline](#-workflow-pipeline)
- [Model Details](#-model-details)
- [Performance Results](#-performance-results)
- [Confusion Matrix](#-confusion-matrix)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Demo Screenshots](#-demo-screenshots)
- [Future Work](#-future-work)
- [Team](#-team)
- [References](#-references)

---

## 🔬 Overview

**RidgeVision AI** is an academic research prototype that explores the correlation between fingerprint ridge patterns (dermatoglyphics) and ABO/Rh blood group classification. The system implements **RidgeAttenFusionNet**, a custom dual-branch ensemble neural network that fuses:

- **Branch 1:** Deep CNN features via transfer learning (EfficientNetB0 / MobileNetV2 with CBAM Attention)
- **Branch 2:** Handcrafted texture features — **LBP** (Local Binary Pattern) + **GLCM** (Gray-Level Co-occurrence Matrix) + **Ridge Density**

Predictions are accompanied by **Grad-CAM** heatmaps for visual explainability, enabling researchers to understand which fingerprint regions drive each prediction.

The ensemble achieves **~89.5%–90% test accuracy** (Kaggle test set evaluation) across all 8 ABO/Rh blood group classes (A+, A−, B+, B−, AB+, AB−, O+, O−).

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Non-Invasive Prediction** | Blood group inference from fingerprint images — no blood sample needed |
| 🧠 **Dual-Branch Ensemble** | Combines deep CNN features with handcrafted texture descriptors |
| 🎯 **8-Class Classification** | Supports all ABO/Rh blood groups: A+, A−, B+, B−, AB+, AB−, O+, O− |
| 🔥 **Grad-CAM Explainability** | Visual heatmap overlay showing which regions influence predictions |
| 📊 **Confidence Matrix** | Class-wise probability distribution for all 8 groups |
| 🧹 **Preprocessing Pipeline** | CLAHE + Gabor filtering + denoising + ridge enhancement |
| 🌐 **Web Interface** | Clean HemaPulse AI-style frontend deployed on Hugging Face Spaces |
| 📈 **~89.5%–90% Ensemble Accuracy** | Macro-average precision, recall, and F1 score all at 0.90 (Kaggle test set evaluation) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RidgeAttenFusionNet                          │
│                    Dual-Branch Ensemble Network                      │
├─────────────────────────────┬───────────────────────────────────────┤
│        Branch 1 (CNN)       │        Branch 2 (Texture)             │
│                             │                                       │
│  Input Fingerprint Image    │  Input Fingerprint Image              │
│          ↓                  │          ↓                            │
│  CLAHE Preprocessing        │  CLAHE + Denoising                    │
│          ↓                  │          ↓                            │
│  Gabor Filter Enhancement   │  Ridge Enhancement (Gabor)            │
│          ↓                  │          ↓                            │
│  EfficientNetB0 Backbone    │  LBP Feature Extraction               │
│  (Pretrained ImageNet)      │  GLCM Texture Descriptors             │
│          ↓                  │  Ridge Density Map                    │
│  CBAM Attention Module      │          ↓                            │
│  (Channel + Spatial)        │  MLP Feature Fusion                   │
│          ↓                  │          ↓                            │
│  Global Average Pooling     │  Dense Layer (256 → 128)              │
│          ↓                  │          ↓                            │
│  Dense (512 → 256)          │  BatchNorm + Dropout (0.3)            │
│          ↓                  │          ↓                            │
│  Dropout (0.4)              │  Softmax Output (8 classes)           │
│          ↓                  │                                       │
│  Softmax Output (8 classes) │                                       │
├─────────────────────────────┴───────────────────────────────────────┤
│                      Ensemble Averaging                              │
│          (Weighted Average of Branch 1 + Branch 2 probabilities)    │
│                             ↓                                        │
│            Final Predicted Blood Group (ABO/Rh)                     │
│                     + Confidence Score                               │
│                     + Grad-CAM Heatmap                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Pipeline

The diagram below illustrates the full end-to-end prediction pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ User Submits │───▶│   Image is   │───▶│   Key Features are   │
│ Fingerprint  │    │  Processed   │    │      Identified       │
│              │    │              │    │                       │
│ Upload PNG/  │    │ Base64 decode│    │ LBP, GLCM, Ridge      │
│ JPG/BMP to   │    │ + preparation│    │ Density & Texture     │
│ web interface│    │ for pipeline │    │ features extracted    │
└──────────────┘    └──────────────┘    └──────────────────────┘
                                                   │
                                                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ Visualizes   │    │   System     │    │    Image is          │
│  Important   │◀───│  Outputs     │◀───│    Enhanced          │
│   Regions    │    │ Blood Group  │    │                      │
│              │    │              │    │ CLAHE + Denoising    │
│ Grad-CAM     │    │ A+, A−, B+,  │    │ + Ridge Enhancement  │
│ heatmap      │    │ B−, AB+, AB−,│    │ applied to input     │
│ displayed    │    │ O+, or O−    │    │ fingerprint image    │
└──────────────┘    └──────────────┘    └──────────────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Results are │    │ Confidence   │    │   Model Predicts     │
│ Presented to │◀───│  Quantified  │◀───│   Blood Group        │
│    User      │    │              │    │                      │
│              │    │ Probability  │    │ Pre-trained Keras     │
│ Results +    │    │ matrix +     │    │ ensemble model        │
│ Confidence + │    │ class-wise   │    │ runs inference        │
│ Heatmap shown│    │ scores shown │    │ on processed image   │
└──────────────┘    └──────────────┘    └──────────────────────┘
```

---

## 🧠 Model Details

### RidgeAttenFusionNet — Two-Model Ensemble

> **RidgeAttenFusionNet** is a custom-designed experimental ensemble architecture developed for this research.

The system trains **two independent CNN models** (Model 88-style and Model 91-style) and averages their softmax outputs:

| Component | Specification |
|-----------|--------------|
| **Backbone** | EfficientNetB0 (pretrained on ImageNet) |
| **Attention** | CBAM — Channel Attention + Spatial Attention |
| **Texture Features** | LBP (radius=1, 8 neighbors) + GLCM (contrast, correlation, energy, homogeneity) |
| **Ridge Features** | Gabor filter bank (8 orientations) for ridge density |
| **Image Preprocessing** | CLAHE (clip limit 2.0, tile grid 8×8) + Gaussian denoising |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output Classes** | 8 (A+, A−, AB+, AB−, B+, B−, O+, O−) |
| **Loss Function** | Categorical Cross-Entropy |
| **Optimizer** | Adam (lr=1e-4, with ReduceLROnPlateau) |
| **Regularization** | Dropout (0.3–0.4) + BatchNormalization |
| **Training Platform** | Kaggle (GPU P100) |

### CBAM Attention Module

```
Input Feature Map
       ↓
┌──────────────────────┐
│  Channel Attention   │  → Global Avg Pool + Global Max Pool → MLP → Sigmoid → Scale
└──────────────────────┘
       ↓
┌──────────────────────┐
│  Spatial Attention   │  → Channel-wise Avg + Max → Conv2D(7×7) → Sigmoid → Scale
└──────────────────────┘
       ↓
Attended Feature Map
```

### Preprocessing Pipeline

```
Raw Fingerprint Image
        ↓
   Grayscale Convert
        ↓
   CLAHE Equalization  ──── (Contrast Limited Adaptive Histogram Equalization)
        ↓
   Gaussian Denoising
        ↓
   Gabor Ridge Enhancement  ──── (8 orientations × 4 frequencies)
        ↓
   Resize to 224×224
        ↓
   Normalize [0, 1]
        ↓
   Ready for Model Inference
```

## 📌 Experimental Note

This system is designed for research purposes to explore potential biometric correlations using deep learning. The results are statistical predictions and should be interpreted as experimental findings rather than deterministic biological conclusions.

---

## 📊 Performance Results

### Ensemble Accuracy: **~89.5%–90%** (Kaggle test set evaluation)

| Model | Test Accuracy |
|-------|--------------|
| Model 88-style | 86.0% |
| Model 91-style | 89.4% |
| **Ensemble (Avg)** | **89.5–90.0%** |

### Per-Class Classification Report (1200 test samples, 150 per class)

| Blood Group | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **A+** | 0.88 | 0.89 | 0.88 | 150 |
| **A−** | 0.92 | 0.89 | 0.91 | 150 |
| **AB+** | 0.88 | 0.86 | 0.87 | 150 |
| **AB−** | 0.92 | 0.92 | 0.92 | 150 |
| **B+** | 0.89 | 0.92 | 0.90 | 150 |
| **B−** | 0.92 | 0.92 | 0.92 | 150 |
| **O+** | 0.93 | 0.85 | 0.89 | 150 |
| **O−** | 0.83 | 0.91 | 0.87 | 150 |
| **Accuracy** | — | — | **0.90** | **1200** |
| **Macro Avg** | 0.90 | 0.90 | 0.90 | 1200 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 1200 |

---

## 🗂️ Confusion Matrix

The confusion matrix below shows the RidgeVision Retrained Two-Model Ensemble performance across all 8 blood group classes on the test set:

```
Actual \ Predicted   A+    A−   AB+   AB−    B+    B−    O+    O−
─────────────────────────────────────────────────────────────────────
A+                  134     0     5     0     0     0     3     8
A−                    0   134     2     2     1     6     3     2
AB+                   6     0   129     0     8     0     3     4
AB−                   0     2     0   138     2     4     0     4
B+                    0     2     5     3   138     2     0     0
B−                    0     3     0     4     5   138     0     0
O+                    7     4     1     1     0     0   127    10
O−                    6     1     4     2     1     0     0   136
```

**Key Observations:**
- **AB−** and **B−** achieve the highest per-class accuracy (138/150 correct)
- **O+** shows occasional confusion with A+ and O−
- **A+** has minor confusion with O− (8 misclassifications), which is a known challenge in dermatoglyphic classification
- Overall strong diagonal dominance confirms excellent model performance

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|----------|------------|---------|
| Backend Framework | FastAPI | API endpoints, inference handling, and backend logic |
| Frontend | HTML5, CSS3, JavaScript | User interface for fingerprint upload and results display |
| Static Hosting | FastAPI StaticFiles | Serves frontend assets |
| Deep Learning | TensorFlow / Keras | Model training and inference |
| Computer Vision | OpenCV, scikit-image | Image preprocessing and feature extraction |
| Explainability | Grad-CAM | Visual explanation of model predictions |
| Training Environment | Kaggle (GPU P100) | Model training and experimentation |
| Deployment | Hugging Face Spaces | Cloud hosting and live demo |

---

## 📁 Project Structure

```
Ridgevision-ai/
│
├── app.py                          # Main application entry point (FastAPI)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── model_88/                   # Style-88 trained weights (.h5 / .keras)
│   ├── model_91/                   # Style-91 trained weights (.h5 / .keras)
│   └── ensemble_config.json        # Ensemble weight configuration
│
├── preprocessing/
│   ├── clahe_pipeline.py           # CLAHE + Gaussian denoising
│   ├── gabor_enhance.py            # Gabor filter ridge enhancement
│   └── feature_extraction.py       # LBP + GLCM + Ridge density
│
├── model_architecture/
│   ├── ridgeatten_fusion.py        # RidgeAttenFusionNet definition
│   ├── cbam_attention.py           # CBAM Channel + Spatial Attention
│   └── ensemble_predict.py         # Ensemble averaging + inference
│
├── explainability/
│   └── gradcam.py                  # Grad-CAM heatmap generation
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Main web interface (HemaPulse AI UI)
│   └── static/
│       ├── style.css               # Styling
│       └── script.js               # Frontend logic (drag-drop, fetch)
│
├── notebooks/
│   └── 90_accuracy_training.ipynb  # Kaggle training notebook
│
├── assets/
│   ├── confusion_matrix.png        # Ensemble confusion matrix
│   ├── workflow_diagram.png        # Pipeline flowchart
│   └── classification_report.txt  # Full test metrics
│
└── report_images/
    └── ridgevision_train_both_ensemble_report.txt
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8–3.11
- pip
- (Recommended) NVIDIA GPU with CUDA 11.x for fast inference

### 1. Clone the Repository

```bash
git clone https://github.com/NanubalaSravani/Ridgevision-ai.git
cd Ridgevision-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core requirements:**
```
tensorflow>=2.10
keras
numpy
pandas
scikit-learn
scikit-image
opencv-python
matplotlib
seaborn
fastapi
uvicorn
Pillow
```

### 4. Download Pre-trained Models

Place model weights in the `models/` directory. If using Kaggle-trained weights:

```bash
# From Kaggle (requires kaggle API token)
kaggle kernels output sravaninanubala/90-accuracy -p ./models/
```

Or download directly from the Hugging Face repo and place `.h5` / `.keras` files in `models/`.

---

## 🚀 Running the App

### Local (FastAPI)

```bash
uvicorn app:app --reload
```
Open your browser at: `http://localhost:8000`

### Using the Web Interface

1. **Upload** a fingerprint image (PNG, JPG, or BMP)
2. The system automatically **validates image quality** (clarity + completeness)
3. Click **Analyze** — the AI inference engine runs:
   - CLAHE + Gabor preprocessing
   - LBP / GLCM feature extraction
   - Dual-branch ensemble inference
   - Grad-CAM heatmap generation
4. **Results** displayed:
   - Predicted blood group (e.g., **AB−**)
   - Confidence percentage
   - Class likelihood matrix (all 8 probabilities)
   - Grad-CAM attention overlay

---

## 🖥️ Demo Screenshots

### Web Interface — RidgeVision AI (HemaPulse Frontend)

The deployed interface at [sravaninanubala-ridgevision-ai.hf.space](https://sravaninanubala-ridgevision-ai.hf.space/) features:

- **Fingerprint Scanner Panel** — Drag & drop or browse for fingerprint image
- **AI Inference Engine Panel** — Live pipeline status (CLAHE → LBP/GLCM Fusion → MLP → Grad-CAM)
- **Diagnostic Report Panel** — Top prediction, confidence score, class likelihood matrix, and Grad-CAM heatmap overlay

---

## 🔮 Future Work

| Enhancement | Description |
|------------|-------------|
| 🗃️ **Larger Dataset** | Train on a more diverse, larger fingerprint-blood group dataset for better generalization |
| 🔍 **Multi-finger Fusion** | Use all 10 fingerprints to improve prediction reliability |
| 📱 **Mobile App** | Deploy as a lightweight mobile application (TFLite) |
| 🩺 **Clinical Validation** | Partner with medical institutions to validate predictions on clinical samples |
| ⚡ **Model Optimization** | Quantization + pruning for edge device deployment |
| 🌐 **Multilingual UI** | Add Telugu and Hindi language support |
| 📚 **Expanded Classes** | Extend to include MN, P, Lewis blood group systems |
| 🔒 **Privacy-Preserving ML** | Federated learning for sensitive biometric data |

---

## 👥 Team

| Name |
|------|
| **N. Sravani** |
| **N. B. Bhuvana Deepthi** |
| **P. Likhitha** |
| **R. Kishore** |

---

## ⚠️ Disclaimer

> This project is an **academic research prototype** developed as a B.Tech mini-project. It demonstrates experimental correlations between fingerprint ridge patterns and blood group classification using machine learning. It is **NOT clinically validated** and must **not be used for medical diagnosis**. Blood group determination must always be performed by a certified medical laboratory.

---

## 📚 References

1. Maltoni, D., Maio, D., Jain, A. K., & Prabhakar, S. (2009). *Handbook of Fingerprint Recognition*. Springer.
2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
4. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI*.
5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
6. Cummins, H., & Midlo, C. (1943). *Finger Prints, Palms and Soles*. Philadelphia: Blakiston.
7. Prasanth Vaidya S., Tyagi A., Ashok S., Naga Satish G., & Meenakshi R. (2024). Advancements in hemotype identification: Fingerprint analysis for blood group determination. *Department of CSE, BVRIT HYDERABAD College of Engineering for Women; Alliance University; Chennai Institute of Technology*.

---

## 📄 License

This project is released for **academic research and educational purposes only**.
© 2025–2026 N. Sravani & Team — The Apollo University.

---

<div align="center">

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge)](https://sravaninanubala-ridgevision-ai.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/NanubalaSravani/Ridgevision-ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/sravaninanubala/90-accuracy)

</div>

