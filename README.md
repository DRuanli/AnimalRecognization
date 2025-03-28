# Animal Recognition

A deep learning project for classifying 90 different animal species using TensorFlow and transfer learning.

## Project Overview

This project uses a CNN-based model to classify animal images across 90 different categories. It implements a complete ML pipeline including data ingestion, model preparation, training, evaluation, and deployment.

## Tech Stack

- Python 3.9+
- TensorFlow 2.13.0
- Keras
- Flask
- Pandas, NumPy
- Matplotlib, Seaborn

## Project Architecture

The project follows a modular, component-based architecture:

1. **Data Ingestion**: Downloads and extracts the animal image dataset
2. **Model Preparation**: Creates a transfer learning model based on EfficientNetB0
3. **Model Training**: Trains the model with data augmentation
4. **Model Evaluation**: Evaluates model performance on test data
5. **Prediction Service**: Provides a web interface for real-time predictions

## Dataset

- 90 different animal categories
- ~60 images per animal class
- Natural images with varied backgrounds, poses, and lighting
- Source: [Animal Image Dataset - 90 Different Animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

## Project Structure

```
Animal Recognition/
├── .github/             # GitHub configurations
├── artifacts/           # Generated artifacts during the pipeline execution
├── config/              # Configuration files
├── logs/                # Log files
├── research/            # Research notebooks and trials
├── src/                 # Source code
│   └── AnimalRecognization/
│       ├── components/  # Pipeline components
│       ├── config/      # Configuration manager
│       ├── constants/   # Project constants
│       ├── entity/      # Data classes for config
│       ├── pipeline/    # Pipeline scripts
│       └── utils/       # Utility functions
├── templates/           # HTML templates for web app
├── webapp/              # Flask web application
├── setup.py            # Setup script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/DRuanli/AnimalRecognization.git
cd AnimalRecognization
```

2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

Execute the complete pipeline:
```bash
python main.py
```

This will:
- Download and extract the dataset
- Prepare the model architecture
- Train the model on the dataset
- Evaluate the model's performance
- Set up the prediction service

### Using the Web App

Start the Flask web application:
```bash
python webapp/app.py
```

Then access the application at http://localhost:8080 in your browser.

## Model Information

- Base Architecture: EfficientNetB0 (transfer learning)
- Input Size: 224x224x3
- Output: 90 classes (animal species)
- Training Strategy: Fine-tuning with data augmentation

## Performance Metrics

- Accuracy: ~85% on validation data
- Loss: Cross-entropy loss
- Detailed metrics available in `artifacts/model_evaluation/metrics.csv`
- Confusion matrix visualization in `artifacts/model_evaluation/confusion_matrix.png`

## Future Improvements

- Implement model serving using TensorFlow Serving
- Add more sophisticated data augmentation techniques
- Create a more robust API for integration with other applications
- Implement active learning for continual improvement