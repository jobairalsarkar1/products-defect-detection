import csv
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# =====================
# Parameters
# =====================
model_path = "final_model.h5"   # trained model
img_size = (224, 224)           # same size used in training
input_folder = input("Enter path to folder containing images: ").strip()

# Load trained model
model = load_model(model_path)
class_mapping = {0: "bad", 1: "good"}  # depends on ImageDataGenerator order

# =====================
# Function to predict a single image
# =====================


def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = model.predict(img_array)[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0
    return class_mapping[pred_class], pred_prob


# =====================
# Batch Prediction
# =====================
results = []
for fname in os.listdir(input_folder):
    fpath = os.path.join(input_folder, fname)
    if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        label, conf = predict_image(fpath)
        results.append((fname, label, conf))

# =====================
# Print Results
# =====================
print(f"\nPredictions for folder: {input_folder}")
for fname, label, conf in results:
    print(f"{fname} -> {label} ({conf*100:.2f}%)")

# =====================
# Optional: Save to CSV
# =====================
csv_path = os.path.join(input_folder, "predictions.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Prediction", "Confidence"])
    writer.writerows(results)
print(f"\nPredictions saved to {csv_path}")
