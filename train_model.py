import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ✅ Disable XLA JIT Compilation
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0"
tf.config.optimizer.set_jit(False)

# ✅ Force GPU Utilization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]  # Limit to max GPU memory
        )
        print("✅ GPU is configured for full utilization!")
    except RuntimeError as e:
        print(e)

# ✅ Dataset Paths
dataset_path = "D:/BloodCancerDetection/dataset"
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "validation")

# ✅ Image Settings
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32  # Reduced batch size to avoid OOM error

# ✅ Preprocessing & Segmentation
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return img, gray, enhanced, blurred

def segment_image(blurred):
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(thresh, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thresh, edges, morph

def visualize_results(image_path):
    img, gray, enhanced, blurred = preprocess_image(image_path)
    thresh, edges, morph = segment_image(blurred)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 2].imshow(enhanced, cmap="gray")
    axes[0, 2].set_title("Histogram Equalized")
    axes[1, 0].imshow(blurred, cmap="gray")
    axes[1, 0].set_title("Gaussian Blurred")
    axes[1, 1].imshow(thresh, cmap="gray")
    axes[1, 1].set_title("Adaptive Threshold")
    axes[1, 2].imshow(morph, cmap="gray")
    axes[1, 2].set_title("Segmented Output")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ✅ Data Augmentation
datagen_train = ImageDataGenerator(rescale=1.0/255.0, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
datagen_val = ImageDataGenerator(rescale=1.0/255.0)

# ✅ Load Dataset
train_generator = datagen_train.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=BATCH_SIZE, class_mode='categorical')
val_generator = datagen_val.flow_from_directory(val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE, class_mode='categorical')

# ✅ Compute Class Weights
labels_count = Counter(train_generator.classes)
total_samples = sum(labels_count.values())
class_weights = {cls: total_samples / (len(labels_count) * count) for cls, count in labels_count.items()}

# ✅ CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(), MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(), MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(), MaxPooling2D(2,2),

    Flatten(), Dense(256, activation='relu'), Dropout(0.5),
    Dense(4, activation='softmax')  # 4-class classification
])

# ✅ Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Callbacks
callbacks = [EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True),
             ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3)]

# ✅ Train Model
history = model.fit(train_generator, validation_data=val_generator, epochs=30,
                    callbacks=callbacks, class_weight=class_weights)

# ✅ Save Model
model_save_path = "D:/BloodCancerDetection/models/blood_cancer_cnn.h5"
model.save(model_save_path)
print(f"✅ Model Training Complete! Model Saved at {model_save_path}")

# ✅ Test Preprocessing on Sample Image
test_image_path = "D:/BloodCancerDetection/test_image.jpg"
visualize_results(test_image_path)
