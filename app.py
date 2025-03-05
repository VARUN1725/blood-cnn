import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

# ✅ Check and Enable GPU
print("🔍 Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

# ✅ Define Model Path
MODEL_PATH = "D:/BloodCancerDetection/models/blood_cancer_cnn.h5"

# ✅ Load the model once at startup
print("🔄 Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Define Upload Path
UPLOAD_FOLDER = "D:/BloodCancerDetection/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    """Load and preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure correct input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input
    return img_array

def model_predict(img_path):
    """Predict the class and confidence from the given image."""
    print(f"📸 Processing image: {img_path}")
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)

    predicted_class = np.argmax(preds, axis=1)[0]  # Get class index
    confidence = round(np.max(preds) * 100, 2)

    class_names = {0: "Benign", 1: "Early Pre-B", 2: "Pre-B", 3: "Pro-B"}
    predicted_label = class_names.get(predicted_class, "Unknown")

    print(f"🎯 Prediction: {predicted_label}, Confidence: {confidence:.2f}%")
    return predicted_label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    confidence_score = None
    image_path = None  # To store the path of uploaded image

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            print("⚠️ No file uploaded!")
            return render_template("index.html", error="Please upload an image.")

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"✅ File saved at: {file_path}")

        # Predict
        prediction_result, confidence_score = model_predict(file_path)
        image_path = f"/uploads/{file.filename}"  # Web-accessible path

    return render_template(
        "index.html",
        prediction=prediction_result,
        confidence=confidence_score,
        image_path=image_path
    )

# Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
