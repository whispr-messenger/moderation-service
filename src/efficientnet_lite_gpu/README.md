# 🍽️ Food Classification with EfficientNet Lite GPU

A high-performance food image classification system based on **TensorFlow** and **EfficientNet Lite**, optimized for **real-time GPU inference** with excellent accuracy and efficiency.

---

## 🚀 1. Overview

This project implements a multi-class food image classifier using **EfficientNetB0** and a **two-stage transfer learning strategy**.

### Key Features
- Automatic dataset loading & preprocessing
- Built-in GPU-based data augmentation
- EfficientNet backbone with fine-tuning
- Training pipeline with early stopping & LR scheduling
- Full evaluation on a separate Test dataset
- Automatic generation of:
    - Metrics (accuracy, precision, recall, F1-score)
    - Confusion matrix
    - Training curves
    - Dataset visualizations
    - JSON reports
    - Saved model & logs

---

## 📁 2. Project Structure

```
src/efficientnet_lite_gpu/
├── train.py
├── requirements.txt
├── simple_efficientnet_food.h5
├── .venv-efficientnet/
│
├── training_logs/
│ ├── best_metrics.json
│ ├── training_history.json
│
├── data_exploration/
│ ├── class_distribution.png
│ ├── dataset_statistics.png
│ ├── sample_images.png
│
├── evaluation_results/
│ ├── confusion_matrix.png
│ ├── test_class_report.json
│ ├── test_metrics.json
│
├── test/
│ ├── test_gpu.py
```

---

## ⚙️ 3. Environment & Dependencies

### Virtual Environment

**Windows**

```
..venv-efficientnet\Scripts\activate
```

**macOS / Linux**


```
source .venv-efficientnet/bin/activate
```


### Install Dependencies


### Training

python main.py --action=train  
