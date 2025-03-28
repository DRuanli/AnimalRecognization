# src/AnimalRecognization/components/prediction.py
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import PredictionConfig


class PredictionService:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.class_names = []

    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            logger.info(f"Loading model from {self.config.trained_model_path}")
            self.model = tf.keras.models.load_model(self.config.trained_model_path)
        return self.model

    def load_class_names(self):
        """Load or generate class names"""
        if not os.path.exists(self.config.class_names_file):
            # Find the correct path to animal classes
            base_path = Path(self.config.root_dir).parent / "data_ingestion" / "animal-images"
            nested_paths = [
                os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
                os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            ]

            # Use the first valid path
            class_path = None
            for path in nested_paths:
                if os.path.exists(path) and os.listdir(path):
                    class_path = path
                    break

            if class_path:
                self.class_names = sorted([d for d in os.listdir(class_path)
                                           if os.path.isdir(os.path.join(class_path, d))])

                # Save class names
                os.makedirs(os.path.dirname(self.config.class_names_file), exist_ok=True)
                with open(self.config.class_names_file, 'w') as f:
                    json.dump(self.class_names, f)
                logger.info(f"Saved {len(self.class_names)} class names to {self.config.class_names_file}")
            else:
                logger.warning("Could not find class directories. Using placeholder class names.")
                self.class_names = [f"class_{i}" for i in range(self.config.params_num_classes)]
        else:
            # Load existing class names
            with open(self.config.class_names_file, 'r') as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded {len(self.class_names)} class names from {self.config.class_names_file}")

        return self.class_names

    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = Image.open(image_path)
        img = img.resize(tuple(self.config.params_image_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        """Make prediction for a single image"""
        # Load model if not already loaded
        model = self.load_model()

        # Load class names if not already loaded
        if not self.class_names:
            self.load_class_names()

        # Preprocess the image
        img_array = self.preprocess_image(image_path)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get class name
        if predicted_class_idx < len(self.class_names):
            class_name = self.class_names[predicted_class_idx]
        else:
            class_name = f"class_{predicted_class_idx}"

        result = {
            "class_name": class_name,
            "confidence": confidence,
            "class_index": int(predicted_class_idx)
        }

        return result

    def setup_flask_app(self):
        """Create Flask application files for web interface"""
        # Create app.py in the webapp directory
        os.makedirs(self.config.webapp_dir, exist_ok=True)

        app_py_content = """
from flask import Flask, request, render_template, jsonify
import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.AnimalRecognization.pipeline.stage_05_prediction import PredictionPipeline

app = Flask(__name__)

# Initialize prediction pipeline
predictor = PredictionPipeline()
prediction_service = predictor.get_prediction_service()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded file temporarily
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)

    # Get prediction
    try:
        result = prediction_service.predict(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
"""

        # Write app.py
        with open(os.path.join(self.config.webapp_dir, 'app.py'), 'w') as f:
            f.write(app_py_content)

        # Create templates directory
        templates_dir = os.path.join(self.config.webapp_dir, 'templates')
        os.makedirs(templates_dir, exist_ok=True)

        # Create index.html
        index_html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Animal Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-container {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        #preview {
            max-width: 300px;
            margin: 15px auto;
            display: none;
        }
    </style>
</head>
<body>
    <h1>Animal Recognition</h1>
    <div class="upload-container">
        <h2>Upload an animal image for classification</h2>
        <input type="file" id="imageUpload" accept="image/*">
        <img id="preview" src="#" alt="Preview">
    </div>
    <div id="result"></div>

    <script>
        document.getElementById('imageUpload').addEventListener('change', function(e) {
            // Preview image
            const preview = document.getElementById('preview');
            preview.src = URL.createObjectURL(e.target.files[0]);
            preview.style.display = 'block';

            // Upload and predict
            const formData = new FormData();
            formData.append('file', e.target.files[0]);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                if (data.error) {
                    resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <h3>Prediction Result:</h3>
                        <p><strong>Animal:</strong> ${data.class_name}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    `;
                }
                resultDiv.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = `<p>Error: ${error}</p>`;
                document.getElementById('result').style.display = 'block';
            });
        });
    </script>
</body>
</html>
"""
        # Write index.html
        with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
            f.write(index_html_content)

        logger.info(f"Flask application setup complete in {self.config.webapp_dir}")