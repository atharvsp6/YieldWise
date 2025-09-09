from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model/plant_disease_model.h5")

# Class labels (must match train folders)
class_labels = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight',
                'blast','brown_spot','dead_heart','downy_mildew','hispa','normal','tungro']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    result = class_labels[np.argmax(preds)]
    confidence = float(round(100 * np.max(preds), 2))


    return jsonify({"disease": result, "confidence": confidence, "file": filepath})

if __name__ == "__main__":
    app.run(debug=True)
