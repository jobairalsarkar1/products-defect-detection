import os
import shutil
import random
from tqdm import tqdm

# =====================
# User inputs
# =====================
product_name = input("Enter the product name: ").strip()
source_folder = input(
    "Enter the path to the folder containing images: ").strip()

if not os.path.exists(source_folder):
    print("Folder does not exist!")
    exit()

# Collect all images
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(
    ('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
if len(all_images) == 0:
    print("No images found in the folder!")
    exit()

print(f"There are {len(all_images)} images available.")

# Ask how many to parse
while True:
    try:
        num_images = int(input("How many images would you like to parse? "))
        if 1 <= num_images <= len(all_images):
            break
        else:
            print(f"Enter a number between 1 and {len(all_images)}")
    except ValueError:
        print("Please enter a valid integer.")

# Shuffle and select the requested number
random.shuffle(all_images)
images = all_images[:num_images]

# Ask for class label
print("Select the class of images in this batch:")
print("1 - Bad")
print("2 - Good")
while True:
    class_choice = input("Enter 1 or 2: ").strip()
    if class_choice == "1":
        label = "bad"
        break
    elif class_choice == "2":
        label = "good"
        break
    else:
        print("Invalid choice! Enter 1 or 2.")

# =====================
# Define dataset structure for this product
# =====================
dataset_folders = {}
for split in ["train", "val", "test"]:
    dataset_folders[split] = {}
    path = os.path.join("dataset", product_name, split, label)
    os.makedirs(path, exist_ok=True)
    dataset_folders[split][label] = path

# =====================
# Split into 70/15/15
# =====================
n = len(images)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

# =====================
# Move images
# =====================


def move_images(img_list, split):
    print(f"\nMoving {len(img_list)} images to {split}/{label} ...")
    for img in tqdm(img_list, desc=f"{split}/{label}", ncols=100):
        src = os.path.join(source_folder, img)
        dst = os.path.join(dataset_folders[split][label], img)
        shutil.copy2(src, dst)


move_images(train_imgs, "train")
move_images(val_imgs, "val")
move_images(test_imgs, "test")

print(f"\nDone! {len(images)} {label} images processed.")
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
