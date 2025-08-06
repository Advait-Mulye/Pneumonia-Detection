from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam_utils import make_gradcam_heatmap, save_and_overlay_gradcam
import uuid

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = load_model(MODEL_PATH)

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)[0][0]
            label = "Pneumonia" if prediction > 0.5 else "Normal"
            prob = round(float(prediction * 100 if label == "Pneumonia" else (1 - prediction) * 100), 2)

            # Grad-CAM
            heatmap_path = save_and_overlay_gradcam(model, img_array, filepath, label)

            return render_template('result.html', label=label, prob=prob, original=unique_name, heatmap=os.path.basename(heatmap_path))
        else:
            return "Invalid file format. Please upload .png, .jpg or .jpeg."

    return render_template('index.html')

# Run app
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
