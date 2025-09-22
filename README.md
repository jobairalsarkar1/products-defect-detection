# Product Defect Detection

This project is a **Product Defect Detection System** using deep learning. It classifies product images as **good** or **bad**, designed for live conveyor belt inspection or batch testing.

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
2. Organize them in the following structure:

```
dataset/
│── train/
│   ├── good/
│   ├── bad/
│── val/
│   ├── good/
│   ├── bad/
│── test/
│   ├── good/
│   ├── bad/
```

3. You can use the provided `organize_dataset.py` script to help move images into the correct folders by selecting whether a folder contains good or bad images.

---

## 3. Training the Model

The `train_model.py` script trains a **binary classifier** using **EfficientNetB0**:

```bash
python train_model.py
```

### Features:

- Transfer learning with EfficientNetB0
- Data augmentation for better generalization
- Saves best model (`best_model.h5`) and final model (`final_model.h5`)
- Evaluation on test set with:

  - Accuracy and Loss curves
  - Confusion matrix
  - Classification report (Precision, Recall, F1-score)

---

## 4. Live Inference

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

## 5. Requirements

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

## 6. Folder Structure

```
project/
│── dataset/
│── train_model.py
│── predict_image.py
│── predict_folder.py
│── organize_dataset.py
│── env_check.py
│── requirements.txt
```

---

## 7. Notes

- Designed for **binary classification** (good/bad).
- Can be extended to any product type or dataset, as long as images are organized properly.
- Works on **CPU or GPU**. GPU is recommended for faster training.
