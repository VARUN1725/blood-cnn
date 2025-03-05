import os
import shutil
import random

# Set dataset paths
dataset_dir = r"D:\BloodCancerDetection\dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")

# Ensure validation directory exists
os.makedirs(val_dir, exist_ok=True)

# Categories (classes) based on the existing dataset
categories = ["Benign", "Malignant Early Pre-B", "Malignant Pre-B", "Malignant Pro-B"]

# Split ratio (80% train, 20% validation)
split_ratio = 0.2

for category in categories:
    # Define source and destination paths
    source_path = os.path.join(train_dir, category)  # <-- FIXED
    val_path = os.path.join(val_dir, category)

    # Ensure validation directory exists
    os.makedirs(val_path, exist_ok=True)

    # Get all image files in the category
    if not os.path.exists(source_path):
        print(f"❌ Error: Source path {source_path} not found.")
        continue

    images = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle images randomly
    random.shuffle(images)

    # Calculate split index
    split_index = int(len(images) * (1 - split_ratio))

    # Move images to validation folder
    for image in images[split_index:]:
        src = os.path.join(source_path, image)
        dst = os.path.join(val_path, image)
        shutil.move(src, dst)

    print(f"✅ Processed {category}: {split_index} train, {len(images) - split_index} validation")

print("\n🎯 Dataset split completed successfully!")
