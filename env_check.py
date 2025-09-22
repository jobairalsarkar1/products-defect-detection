import time
import tensorflow as tf
import os
import psutil

print("=== System Info ===")
# RAM
ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
print(f"Total RAM: {ram_gb} GB")

# Disk space
disk_gb = round(psutil.disk_usage("/").total / (1024 ** 3), 2)
print(f"Total Disk: {disk_gb} GB")

print("\n=== TensorFlow Info ===")
print(f"TensorFlow version: {tf.__version__}")

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("No GPU detected. Training will run on CPU (slower).")

# Test TensorFlow GPU
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal((1024, 1024, 10))
        b = tf.random.normal((1024, 1024, 10))
        c = tf.matmul(tf.reshape(a, (-1, 10)),
                      tf.reshape(b, (-1, 10)), transpose_b=True)
    print("✅ GPU is working with TensorFlow.")
except:
    print("⚠ GPU test failed, running on CPU instead.")

# Quick TensorFlow matrix multiply to estimate speed
start = time.time()
x = tf.random.normal((2048, 2048))
y = tf.random.normal((2048, 2048))
z = tf.matmul(x, y)
end = time.time()
print(f"TensorFlow matrix multiply time: {end - start:.3f} sec")
