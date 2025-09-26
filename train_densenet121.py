import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =====================
# Parameters
# =====================
dataset_dir = "dataset"
results_dir = "results"
models_dir = "saved_models"

# get product name from dataset_dir
product_name = os.path.basename(dataset_dir.rstrip("/"))

model_name = "DenseNet121"
img_size = (224, 224)
batch_size = 16
epochs = 10

# redefine dirs with product_name and model_name subfolders
results_dir = os.path.join(results_dir, product_name, model_name)
models_dir = os.path.join(models_dir, product_name, model_name)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# =====================
# Data Generators (RGB)
# =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=img_size,
    color_mode="rgb",   # <-- RGB input
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "val"),
    target_size=img_size,
    color_mode="rgb",   # <-- RGB input
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=img_size,
    color_mode="rgb",   # <-- RGB input
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# =====================
# Build Model (RGB)
# =====================
input_tensor = Input(shape=(224, 224, 3))  # <-- RGB input

base_model = DenseNet121(
    include_top=False,
    weights=None,  # train from scratch
    input_tensor=input_tensor
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # binary classification

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# =====================
# Callbacks
# =====================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, f"{model_name}_best_model.keras"),
    monitor="val_accuracy", save_best_only=True, verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# =====================
# Train Model
# =====================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# =====================
# Evaluate on Test Set
# =====================
test_gen.reset()
y_pred_prob = model.predict(test_gen, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_gen.classes

# =====================
# Confusion Matrix
# =====================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_gen.class_indices.keys(),
            yticklabels=test_gen.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
plt.close()

# =====================
# Classification Report
# =====================
report = classification_report(
    y_true, y_pred, target_names=test_gen.class_indices.keys())
print("Classification Report:\n")
print(report)

with open(os.path.join(results_dir, f"{model_name}_classification_report.txt"), "w") as f:
    f.write(report)

# =====================
# Plot Training History
# =====================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, f"{model_name}_training_history.png"))
plt.close()

# =====================
# Save Final Model
# =====================
final_model_path = os.path.join(models_dir, f"{model_name}_final_model.keras")
model.save(final_model_path)
print(f"Training complete! Model saved as {final_model_path}")
