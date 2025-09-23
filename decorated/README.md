# Product Defect Detection

This project is a **Product Defect Detection System** using deep learning.
It classifies product images as **good** or **bad**, designed for live conveyor belt inspection or batch testing.

---

## 1. Environment Check

Before starting, you can run `env_check.py` to verify your system setup:

```bash
python env_check.py
```

This script checks:

- Total RAM and Disk
- TensorFlow version
- GPU availability

---

## 2. Dataset Organization

1. Collect images of your product, separated into **good** and **bad** images.
2. Organize them in the following structure (per product):

```
dataset/
│── product_name/
│   ├── train/
│   │   ├── good/
│   │   ├── bad/
│   ├── val/
│   │   ├── good/
│   │   ├── bad/
│   ├── test/
│       ├── good/
│       ├── bad/
```

3. You can use the provided `organize_dataset.py` script to move images into the correct folders by selecting:

   - Product name (e.g., `pill`)
   - Whether the folder contains **good** or **bad** images

This allows training multiple products separately under the same `dataset/` folder.

---

## 3. Training the Model

The `train_model.py` script trains a **binary classifier** using **MobileNetV2** (lightweight and efficient):

```bash
python train_model.py
```

### Features:

- Lightweight model: **MobileNetV2**
- Data augmentation for better generalization
- Grayscale input support
- Saves models in `.keras` format:

  - Best model → `saved_models/MobileNetV2_best_model.keras`
  - Final model → `saved_models/MobileNetV2_final_model.keras`

- Evaluation on test set with:

  - Accuracy and Loss curves
  - Confusion matrix (saved as image)
  - Classification report (Precision, Recall, F1-score, saved as text file)

---

## 4. Results

All evaluation outputs are stored in the `results/` folder:

- `MobileNetV2_confusion_matrix.png`
- `MobileNetV2_training_history.png`
- `MobileNetV2_classification_report.txt`

---

## 5. Live Inference

### Single Image Prediction

Use `predict_image.py` to predict a single image:

```bash
python predict_image.py
Enter path to image: path/to/image.png
Prediction: good
Confidence: 92.00%
```

### Batch Prediction

Use `predict_folder.py` to predict all images in a folder:

```bash
python predict_folder.py
Enter path to folder containing images: path/to/folder
```

- Prints predictions for each image
- Saves results as `predictions.csv` in the same folder

---

## 6. Requirements

- Python 3.7+
- TensorFlow
- numpy
- matplotlib
- seaborn
- scikit-learn

Install required packages using:

```bash
pip install -r requirements.txt
```

---

## 7. Folder Structure

```
project/
│── dataset/
│── results/
│── saved_models/
│── train_model.py
│── predict_image.py
│── predict_folder.py
│── organize_dataset.py
│── env_check.py
│── requirements.txt
```

---

## 8. Notes

- Designed for **binary classification** (good/bad).
- Supports multiple product datasets under `dataset/`.
- Works on **CPU or GPU**. GPU is recommended for faster training.
- Lightweight model (MobileNetV2) makes it suitable for edge devices.
