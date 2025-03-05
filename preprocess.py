import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
ddataset_path = "D:/BloodCancerDetection/dataset/BloodCellCancerALL"


# Define image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print dataset summary
print("\nDataset Summary:")
print(f"Total training images: {train_data.samples}")
print(f"Total validation images: {val_data.samples}")
print("Class Labels:", train_data.class_indices)

# Check one batch of images
x_batch, y_batch = next(train_data)
print("\nSample Batch Shape:", x_batch.shape)
print("Sample Labels:", y_batch.shape)


# Check class labels
print("Class Labels:", train_data.class_indices)
