# src/AnimalRecognization/components/prepare_model.py
import tensorflow as tf
from pathlib import Path
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import PrepareModelConfig


class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def get_base_model(self):
        """
        Creates a pre-trained EfficientNetB0 model as the base for transfer learning
        """
        input_shape = self.config.params_image_size + [3]  # [height, width, channels]

        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

        # Freeze base model layers
        base_model.trainable = False

        logger.info("Base model summary:")
        logger.info(f"Input shape: {input_shape}")

        return base_model

    def prepare_full_model(self, base_model):
        """
        Adds classification head to the base model for animal classification
        """
        # Create feature extractor
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Add classification layers
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)

        # Output layer with softmax for multi-class classification
        outputs = tf.keras.layers.Dense(self.config.params_num_classes, activation='softmax')(x)

        # Assemble full model
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Full model prepared")

        return model

    def save_model(self, path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info(f"Model saved at: {path}")

    def prepare_model(self):
        """
        Prepares complete model for animal recognition
        """
        base_model = self.get_base_model()
        full_model = self.prepare_full_model(base_model)

        # Save model
        self.save_model(self.config.model_path, full_model)

        return full_model