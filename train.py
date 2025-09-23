import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =====================
# Parameters
# =====================
dataset_dir = "dataset"
img_size = (224, 224)  # EfficientNetB0 default
batch_size = 16
epochs = 10

# =====================
# Data Generators
# =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load grayscale images
train_gen_gray = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

val_gen_gray = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "val"),
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

test_gen_gray = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# =====================
# Helper: convert grayscale -> RGB
# =====================
def gray_to_rgb(gen):
    for batch_x, batch_y in gen:
        batch_x = np.repeat(batch_x, 3, axis=-1)  # (H,W,1) -> (H,W,3)
        yield batch_x, batch_y

train_gen = gray_to_rgb(train_gen_gray)
val_gen = gray_to_rgb(val_gen_gray)
test_gen = gray_to_rgb(test_gen_gray)

steps_train = len(train_gen_gray)
steps_val = len(val_gen_gray)
steps_test = len(test_gen_gray)

# =====================
# Build Model
# =====================
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",     # pretrained
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # binary classification

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# =====================
# Callbacks
# =====================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# =====================
# Train Model
# =====================
history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# =====================
# Evaluate on Test Set
# =====================
y_pred_prob = model.predict(test_gen, steps=steps_test, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_gen_gray.classes

# =====================
# Confusion Matrix
# =====================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_gen_gray.class_indices.keys(),
            yticklabels=test_gen_gray.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================
# Classification Report
# =====================
report = classification_report(
    y_true, y_pred, target_names=test_gen_gray.class_indices.keys())
print("Classification Report:\n")
print(report)

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
plt.show()

# =====================
# Save Final Model
# =====================
model.save("final_model.keras")
print("Training complete! Model saved as final_model.h5")
