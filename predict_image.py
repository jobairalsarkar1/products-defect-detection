import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# =====================
# Parameters
# =====================
model_path = "final_model.h5"   # trained model
img_size = (224, 224)           # same size used in training

# Load trained model
model = load_model(model_path)

# Mapping class index to label
class_mapping = {0: "bad", 1: "good"}  # depends on ImageDataGenerator order

# =====================
# Function to predict a single image
# =====================


def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # rescale
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    pred_prob = model.predict(img_array)[0][0]
    if pred_prob > 0.5:
        pred_class = 1  # good
    else:
        pred_class = 0  # bad

    print(f"Prediction: {class_mapping[pred_class]}")
    print(f"Confidence: {pred_prob*100:.2f}%")


# =====================
# User input
# =====================
img_path = input("Enter path to image: ").strip()
predict_image(img_path)
