# Import missing modules and add advanced imports
import os
import imghdr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import class_weight_utils
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        # Enable mixed precision for faster training on compatible GPUs
        if tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Using mixed precision training")

    def get_image_dataset(self):
        """Create training and validation datasets with advanced augmentation"""
        # Find correct dataset path
        training_path = self._find_dataset_path()

        # Advanced augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            validation_split=self.config.params_validation_split
        )

        # Minimal processing for validation
        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=self.config.params_validation_split
        )

        # Training generator with class balancing
        train_generator = train_datagen.flow_from_directory(
            training_path,
            target_size=tuple(self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            training_path,
            target_size=tuple(self.config.params_image_size),
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        # Calculate class weights to handle imbalance
        class_weights = self._calculate_class_weights(train_generator.classes)

        logger.info(f"Found {train_generator.samples} training images in {train_generator.num_classes} classes")
        logger.info(f"Found {val_generator.samples} validation images")

        return train_generator, val_generator, class_weights

    def _find_dataset_path(self):
        """Find the correct path to class directories"""
        base_path = self.config.training_data
        nested_paths = [
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals", "animals"),
            os.path.join(base_path, "animal-image-dataset-90-different-animals", "animals"),
            base_path
        ]

        for path in nested_paths:
            if os.path.exists(path) and len([d for d in os.listdir(path)
                                             if os.path.isdir(os.path.join(path, d))]) > 0:
                logger.info(f"Using training data at: {path}")
                return path

        raise ValueError("Could not find valid training data directory")

    def _calculate_class_weights(self, classes):
        """Calculate weights for class imbalance"""
        unique_classes = np.unique(classes)
        class_weights = {}

        if len(unique_classes) > 1:
            try:
                weights = class_weight_utils.compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=classes
                )
                class_weights = {i: w for i, w in enumerate(weights)}
                logger.info("Using balanced class weights")
            except:
                logger.warning("Could not compute class weights, using equal weighting")

        return class_weights

    def validate_dataset(self):
        """Validate the dataset structure and files"""
        logger.info(f"Validating dataset at {self.config.training_data}")

        # Find correct dataset path
        training_path = self._find_dataset_path()

        # Get class directories
        class_dirs = [d for d in os.listdir(training_path)
                      if os.path.isdir(os.path.join(training_path, d))]

        logger.info(f"Found {len(class_dirs)} classes")

        # Check for valid images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        problem_files = []

        for class_dir in class_dirs:
            class_path = os.path.join(training_path, class_dir)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                if not os.path.isfile(file_path):
                    continue

                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    problem_files.append((file_path, "Invalid extension"))
                    continue

                try:
                    image_type = imghdr.what(file_path)
                    if image_type is None:
                        problem_files.append((file_path, "Not a valid image"))
                except Exception as e:
                    problem_files.append((file_path, f"Error: {str(e)}"))

        # Remove problematic files
        if problem_files:
            logger.warning(f"Found {len(problem_files)} problem files")
            for path, reason in problem_files[:10]:
                logger.warning(f"  - {path}: {reason}")

            for path, _ in problem_files:
                try:
                    os.remove(path)
                    logger.info(f"Removed problem file: {path}")
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {str(e)}")

        return class_dirs

    def _create_callbacks(self):
        """Create training callbacks with best practices"""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(self.config.root_dir, "best_model.h5"),
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            # Early stopping to prevent overfitting
            EarlyStopping(
                patience=7,
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1
            ),
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.config.root_dir, "logs"),
                update_freq='epoch',
                histogram_freq=1
            )
        ]
        return callbacks

    def _setup_progressive_unfreezing(self, model):
        """Implement progressive layer unfreezing for transfer learning"""
        # Identify the base model layers (EfficientNet)
        base_model_layers = []
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Identify the base model
                base_model_layers = layer.layers
                break

        # If we couldn't identify base model layers, freeze all but last 20
        if not base_model_layers:
            logger.info("Progressive unfreezing: Using default approach")
            # Start with all layers frozen except the last 20
            for layer in model.layers[:-20]:
                layer.trainable = False
            for layer in model.layers[-20:]:
                layer.trainable = True
            return

        # More advanced: Progressive unfreezing of EfficientNet layers
        logger.info(f"Progressive unfreezing: Found {len(base_model_layers)} base model layers")

        # Keep classification head trainable
        for layer in model.layers:
            if not hasattr(layer, 'layers'):
                layer.trainable = True

        # Freeze base model layers, will selectively unfreeze during training
        for layer in base_model_layers:
            layer.trainable = False

        # Unfreeze the BatchNorm layers for better fine-tuning
        for layer in base_model_layers:
            if 'batch_normalization' in layer.name:
                layer.trainable = True

    def _advanced_compile(self, model, lr=None):
        """Compile model with advanced options"""
        if lr is None:
            lr = self.config.params_learning_rate / 10  # Lower LR for fine-tuning

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC()]
        )

    def train(self):
        """Train the model using advanced strategies"""
        # Validate dataset and clean up problematic files
        self.validate_dataset()

        # Load model
        logger.info(f"Loading model from {self.config.updated_base_model_path}")
        model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Setup progressive unfreezing
        self._setup_progressive_unfreezing(model)

        # Get datasets
        train_generator, val_generator, class_weights = self.get_image_dataset()

        # Create callbacks
        callbacks = self._create_callbacks()

        # Calculate steps
        steps_per_epoch = train_generator.samples // self.config.params_batch_size
        validation_steps = val_generator.samples // self.config.params_batch_size

        # Initial training phase - train only the head
        logger.info("Starting initial training phase (head only)")
        self._advanced_compile(model, lr=self.config.params_learning_rate)

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=max(2, self.config.params_epochs // 3),
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Fine-tuning phase - unfreeze more layers
        logger.info("Starting fine-tuning phase")

        # Unfreeze more layers for fine-tuning
        if hasattr(model.layers[0], 'layers'):
            # Selectively unfreeze deeper layers of the base model
            base_layers = model.layers[0].layers
            for layer in base_layers[-20:]:  # Unfreeze the last 20 layers
                layer.trainable = True

        # Recompile with lower learning rate for fine-tuning
        self._advanced_compile(model, lr=self.config.params_learning_rate / 100)

        # Continue training
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.params_epochs,
            initial_epoch=history.epoch[-1] + 1 if history.epoch else 0,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Load the best model from callbacks
        best_model_path = os.path.join(self.config.root_dir, "best_model.h5")
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            model = tf.keras.models.load_model(best_model_path)

        # Save final model
        model.save(self.config.trained_model_path)
        logger.info(f"Model training completed. Model saved at: {self.config.trained_model_path}")

        return history