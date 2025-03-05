import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ GPU Optimizations
print("🔍 Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

# ✅ Load the Model Once
MODEL_PATH = "D:/BloodCancerDetection/models/blood_cancer_cnn.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found: {MODEL_PATH}")

print("🔄 Loading Model...")
model = load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")

# ✅ Image Preprocessing
def preprocess_image(img_path, img_size=(224, 224)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"🚨 Image file not found: {img_path}")
    
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Grad-CAM Implementation
def get_gradcam(model, img_array):
    # Find the last Conv2D layer dynamically
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        print("🚨 No convolutional layers found. Cannot generate Grad-CAM.")
        return None
    
    target_layer = conv_layers[-3]  # Last Conv2D layer
    print(f"🎯 Using Grad-CAM on layer: {target_layer}")
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(target_layer).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = tf.reduce_mean(predictions[:, class_index])  # Averaging loss
    
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        print("⚠️ Warning: Gradients are None. Skipping Grad-CAM.")
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    pooled_grads = pooled_grads / (tf.reduce_max(pooled_grads) + 1e-10)  # Avoid zero gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = heatmap.numpy()
    if np.max(heatmap) == 0:
        print("⚠️ Warning: Heatmap max value is 0. Skipping normalization.")
        return None
    
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# ✅ Overlay Heatmap on Image
def overlay_heatmap(img_path, heatmap, alpha=0.6):
    if heatmap is None or heatmap.size == 0:
        print("🚨 Skipping Grad-CAM overlay due to invalid heatmap.")
        return None
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    output_path = os.path.join("D:/BloodCancerDetection/static/gradcam/", "test_image_gradcam.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, superimposed_img)
    
    return output_path

# ✅ Main Prediction Function
def model_predict(img_path):
    if not os.path.exists(img_path):
        print("🚨 Test image not found! Upload an image first.")
        return None, None, None
    
    print(f"📸 Predicting Image: {img_path}")
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    # Map class index to labels
    class_labels = {0: "Benign", 1: "Early Pre-B", 2: "Pre-B", 3: "Pro-B"}
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels.get(predicted_class_index, "Unknown")
    confidence = round(np.max(predictions) * 100, 2)
    
    # ✅ Run Grad-CAM only if prediction is successful
    print("🔍 Running Grad-CAM for visualization...")
    heatmap = get_gradcam(model, img_array)
    
    gradcam_path = overlay_heatmap(img_path, heatmap) if heatmap is not None else None
    
    return predicted_label, confidence, gradcam_path

# ✅ Run Prediction (Example)
test_image = "D:/BloodCancerDetection/test_image.jpg"
predicted_label, confidence, gradcam_path = model_predict(test_image)

if predicted_label:
    print(f"🎯 Prediction: {predicted_label} ({confidence}%)")
    if gradcam_path:
        print(f"📸 Grad-CAM Saved at: {gradcam_path}")
    else:
        print("⚠️ Grad-CAM visualization skipped due to invalid heatmap.")
else:
    print("❌ Prediction failed. Please check the image path.")
