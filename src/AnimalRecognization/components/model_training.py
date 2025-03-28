# src/AnimalRecognization/components/model_training.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def get_image_dataset(self):
        """Create training and validation datasets"""
        logger.info("Setting up data generators with augmentation")

        # Find the correct path to animal classes
        # The actual classes are at: animal-images/animal-image-dataset-90-different-animals/animals/animals/
        base_path = self.config.training_data
        nested_paths = [
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            # Add fallback paths if needed
        ]

        # Use the first path that exists and contains class directories
        training_path = None
        for path in nested_paths:
            if os.path.exists(path) and os.listdir(path):
                training_path = path
                logger.info(f"Using training data at: {training_path}")
                break

        if not training_path:
            raise ValueError("Could not find valid training data directory")

        # Rest of your generator code using training_path
        datagen_kwargs = {
            'rescale': 1. / 255,
            'validation_split': self.config.params_validation_split
        }

        # Add augmentation if enabled
        if self.config.params_augmentation:
            datagen_kwargs.update({
                'rotation_range': 20,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'shear_range': 0.2,
                'zoom_range': 0.2,
                'horizontal_flip': True,
                'fill_mode': 'nearest'
            })

        # Create ImageDataGenerators
        train_datagen = ImageDataGenerator(**datagen_kwargs)

        # Flow from directory for training data - with the corrected path
        train_generator = train_datagen.flow_from_directory(
            training_path,
            target_size=tuple(self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Flow for validation data - with the corrected path
        val_generator = train_datagen.flow_from_directory(
            training_path,
            target_size=tuple(self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            subset='validation'
        )

        logger.info(f"Found {train_generator.samples} training images in {train_generator.num_classes} classes")
        logger.info(f"Found {val_generator.samples} validation images")

        return train_generator, val_generator

    # Add this method to the ModelTraining class
    def validate_dataset(self):
        """Validate the dataset structure and files"""
        logger.info(f"Validating dataset at {self.config.training_data}")

        # Check directory structure
        if not os.path.exists(self.config.training_data):
            raise ValueError(f"Training data path does not exist: {self.config.training_data}")

        # Find class directories (should be animal categories)
        class_dirs = [d for d in os.listdir(self.config.training_data)
                      if os.path.isdir(os.path.join(self.config.training_data, d))]

        logger.info(f"Found {len(class_dirs)} classes: {class_dirs[:5]}...")

        # Check for valid images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        problem_files = []

        for class_dir in class_dirs:
            class_path = os.path.join(self.config.training_data, class_dir)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                # Skip directories and non-files
                if not os.path.isfile(file_path):
                    continue

                # Check file extension
                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    problem_files.append((file_path, "Invalid extension"))
                    continue

                # Try to verify it's a valid image
                try:
                    image_type = imghdr.what(file_path)
                    if image_type is None:
                        problem_files.append((file_path, "Not a valid image"))
                except Exception as e:
                    problem_files.append((file_path, f"Error: {str(e)}"))

        # Report problems
        if problem_files:
            logger.warning(f"Found {len(problem_files)} problem files")
            for path, reason in problem_files[:10]:  # Show first 10 problems
                logger.warning(f"  - {path}: {reason}")

            # Remove problem files
            for path, _ in problem_files:
                try:
                    os.remove(path)
                    logger.info(f"Removed problem file: {path}")
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {str(e)}")

        return class_dirs

    def train(self):
        """Train the model"""
        class_dirs = self.validate_dataset()

        # Load the model
        logger.info(f"Loading model from {self.config.updated_base_model_path}")
        model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Get training and validation datasets
        train_generator, val_generator = self.get_image_dataset()

        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.root_dir, "model_ckpt.h5"),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                monitor='val_loss',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.1,
                patience=3,
                monitor='val_loss',
                verbose=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config.root_dir, "logs")
            )
        ]

        # Fine-tune the model
        logger.info("Starting model training")

        # Unfreeze some top layers for fine-tuning
        for layer in model.layers[-20:]:
            layer.trainable = True

        # Recompile the model with lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        steps_per_epoch = train_generator.samples // self.config.params_batch_size
        validation_steps = val_generator.samples // self.config.params_batch_size

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.params_epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        # Save the trained model
        model.save(self.config.trained_model_path)
        logger.info(f"Model training completed. Model saved at: {self.config.trained_model_path}")

        return history