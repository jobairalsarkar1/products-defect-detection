import os
import shutil
import random
from tqdm import tqdm  # for progress bar

# Define dataset folder structure
dataset_folders = {
    "train": {"good": "dataset/train/good", "bad": "dataset/train/bad"},
    "val": {"good": "dataset/val/good", "bad": "dataset/val/bad"},
    "test": {"good": "dataset/test/good", "bad": "dataset/test/bad"},
}

# Create folders if they don't exist
for split in dataset_folders:
    for cls in dataset_folders[split]:
        os.makedirs(dataset_folders[split][cls], exist_ok=True)

# User input for folder containing images
source_folder = input(
    "Enter the path to the folder containing images: ").strip()

if not os.path.exists(source_folder):
    print("Folder does not exist!")
    exit()

# User input to label folder as good or bad
print("Select the class of images in this folder:")
print("1 - Bad")
print("2 - Good")
class_choice = input("Enter 1 or 2: ").strip()

if class_choice == "1":
    label = "bad"
elif class_choice == "2":
    label = "good"
else:
    print("Invalid choice!")
    exit()

# Collect all image files
images = [f for f in os.listdir(source_folder) if f.lower().endswith(
    ('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

# Shuffle for randomness
random.shuffle(images)

# Split ratio: 70% train, 15% val, 15% test
n = len(images)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

# Function to move images with progress bar


def move_images(img_list, split):
    print(f"\nMoving {len(img_list)} images to {split}/{label} ...")
    for img in tqdm(img_list, desc=f"{split}/{label}", ncols=100):
        src = os.path.join(source_folder, img)
        dst = os.path.join(dataset_folders[split][label], img)
        shutil.copy2(src, dst)  # copy2 keeps metadata


# Move images
move_images(train_imgs, "train")
move_images(val_imgs, "val")
move_images(test_imgs, "test")

print(f"\nDataset organized successfully! Total images: {len(images)}")
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
