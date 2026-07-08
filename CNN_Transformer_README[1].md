# 🫀 ECG Arrhythmia Classification — CNN + Transformer

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20GPU-yellow)
![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH-red)
![Model](https://img.shields.io/badge/Model-CNN%20%2B%20Transformer-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project implements a **hybrid CNN + Transformer deep learning model** for automated **ECG Arrhythmia Classification** using the **MIT-BIH Arrhythmia Dataset**.

The architecture combines:
- **CNN** (Conv1D blocks) to extract **local morphological features** (QRS complex, P-wave, T-wave)
- **Transformer Encoder** with **Multi-Head Self-Attention** to capture **global temporal dependencies** across the heartbeat

> Developed as an internship project in Deep Learning for Biomedical Signal Processing.

---

## 🏗️ Model Architecture

```
Raw ECG Beat Input  (187 timesteps × 1 channel)
         │
 ┌───────▼────────────────────────────────┐
 │   CNN Block 1: Conv1D(64)  + BN + Pool │  Local feature extraction
 │   CNN Block 2: Conv1D(128) + BN + Pool │  Hierarchical patterns
 │   CNN Block 3: Conv1D(256) + BN + Pool │  High-level features
 └───────┬────────────────────────────────┘
         │  Dense projection → d_model=128
         │
 ┌───────▼────────────────────────────────┐
 │   Positional Encoding                  │  Inject time position info
 └───────┬────────────────────────────────┘
         │
 ┌───────▼────────────────────────────────┐
 │   Transformer Encoder Block 1          │  8-head self-attention
 │     Multi-Head Attention (8 heads)     │  Global temporal dependencies
 │     Feed-Forward (256 units)           │
 │     LayerNorm + Residual connections   │
 ├───────┬────────────────────────────────┤
 │   Transformer Encoder Block 2          │  Deeper attention
 └───────┬────────────────────────────────┘
         │
 ┌───────▼────────────────────────────────┐
 │   Global Average Pooling               │  Aggregate sequence
 │   Dense(128, ReLU) + Dropout           │
 │   Dense(5, Softmax)                    │  5-class output
 └────────────────────────────────────────┘
```

---

## 🎯 Arrhythmia Classes (AAMI Standard)

| Label | Class | Description | Train Samples |
|-------|-------|-------------|---------------|
| 0 | **N** | Normal Beat | 72,471 |
| 1 | **S** | Supraventricular Ectopic Beat | 2,223 |
| 2 | **V** | Ventricular Ectopic Beat | 5,788 |
| 3 | **F** | Fusion of Ventricular and Normal | 641 |
| 4 | **Q** | Unknown / Paced Beat | 6,431 |

---

## 📁 Project Structure

```
ecg_cnn_transformer/
│
├── notebooks/
│   └── ECG_CNN_Transformer.ipynb        ← Main Colab Notebook (run this)
│
├── src/
│   └── model.py                         ← CNN+Transformer model definition
│
├── results/                             ← Generated after running notebook
│   ├── class_distribution.png
│   ├── sample_beats.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── roc_curves.png
│   ├── prediction_confidence.png
│   ├── comparison_chart.png
│   ├── metrics_summary.csv
│   └── comparison_results.csv
│
├── report/
│   └── ECG_CNN_Transformer_Report.pdf   ← Full project report
│
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | MIT-BIH Arrhythmia Dataset |
| Source | [Kaggle — shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) |
| Sampling Rate | 125 Hz |
| Segment Length | 187 samples per beat |
| Training Samples | 87,554 |
| Test Samples | 21,892 |
| Normalization | Amplitude scaled to [0, 1] |

---

## 📈 Expected Results

| Metric | Score |
|--------|-------|
| Accuracy | ~97–99% |
| Precision | ~96–98% |
| Recall | ~96–98% |
| F1-Score | ~96–98% |

---

## 📚 Comparison with Published Research

| Author / Year | Method | Accuracy | F1-Score |
|---------------|--------|----------|----------|
| Acharya et al. (2017) | CNN | 93.47% | 90.64% |
| Kachuee et al. (2018) | ResNet | 93.40% | 91.14% |
| Yildirim et al. (2018) | LSTM | 98.74% | 97.95% |
| Hannun et al. (2019) | Deep CNN | 91.33% | 89.90% |
| Che et al. (2021) | CNN+Transformer | 98.91% | 98.25% |
| **Our Work** | **CNN+Transformer** | **~97–99%** | **~96–98%** |

---

## 🚀 How to Run (3 Steps)

### Step 1 — Get Dataset
Go to [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) and download:
- `mitbih_train.csv`
- `mitbih_test.csv`

### Step 2 — Open in Google Colab
1. Upload `ECG_CNN_Transformer.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. **Enable GPU**: Runtime → Change runtime type → **GPU** (T4)
3. Upload both CSV files when prompted

### Step 3 — Run
Runtime → **Run All**

Training takes ~10–20 minutes on GPU. At the end, `ECG_CNN_Transformer_Results.zip` downloads automatically with all 8 plots + model + CSVs.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| CNN Filters | 64 → 128 → 256 |
| Kernel Size | 7 |
| d_model | 128 |
| Attention Heads | 8 |
| Transformer Blocks | 2 |
| Feed-Forward Dim | 256 |
| Dropout Rate | 0.2 |
| Optimizer | Adam (lr=1e-3) |
| Batch Size | 256 |
| Max Epochs | 50 (EarlyStopping) |
| Class Weights | Balanced (auto) |

---

## 🛠️ Dependencies

```
tensorflow>=2.10
scikit-learn
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

---

## 👤 Author

**Sourav Biswal**  
Internship Project — Deep Learning for ECG Signal Classification  
MIT-BIH Arrhythmia Dataset | CNN + Transformer | Google Colab GPU

---

## 📜 License

MIT License
