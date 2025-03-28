# src/AnimalRecognization/components/model_evaluation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def get_test_dataset(self):
        """Create test dataset"""
        # Find the correct path to animal classes (nested directory structure)
        base_path = self.config.evaluation_data
        nested_paths = [
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            base_path
        ]

        # Use the first valid path
        test_path = None
        for path in nested_paths:
            if os.path.exists(path) and os.listdir(path):
                test_path = path
                logger.info(f"Using test data at: {test_path}")
                break

        if not test_path:
            raise ValueError("Could not find valid test data directory")

        # Set up data generator
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=tuple(self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            shuffle=False  # Important for maintaining order for confusion matrix
        )

        logger.info(f"Found {test_generator.samples} test images in {test_generator.num_classes} classes")

        return test_generator

    def evaluate(self):
        """Evaluate the model on test data"""
        # Load model
        logger.info(f"Loading model from {self.config.trained_model_path}")
        model = tf.keras.models.load_model(self.config.trained_model_path)

        # Get test data
        test_generator = self.get_test_dataset()

        # Evaluate model
        logger.info("Evaluating model...")
        scores = model.evaluate(test_generator)
        metrics = {"loss": scores[0], "accuracy": scores[1]}

        # Get predictions
        logger.info("Generating predictions for confusion matrix...")
        y_pred_probs = model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # True labels
        y_true = test_generator.classes

        # Generate classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Save metrics
        metrics_file = self.config.metrics_file_path
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        report_df.to_csv(metrics_file)
        logger.info(f"Classification report saved to {metrics_file}")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save confusion matrix
        plt.savefig(self.config.confusion_matrix_path)
        logger.info(f"Confusion matrix saved to {self.config.confusion_matrix_path}")

        # Return metrics
        return metrics, report_df