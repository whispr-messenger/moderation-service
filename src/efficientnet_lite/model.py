"""
EfficientNet Lite Model Architecture for Food Classification

This module implements:
- EfficientNet Lite architecture optimized for on-device inference
- Transfer learning from pretrained models
- Model compilation and optimization
- TensorFlow Lite conversion for deployment
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock TensorFlow implementation for development
class MockTensorFlow:
    """Mock TensorFlow implementation for development without dependencies"""
    
    class keras:
        class layers:
            @staticmethod
            def GlobalAveragePooling2D():
                return "GlobalAveragePooling2D"
            
            @staticmethod
            def Dense(units, activation=None, name=None):
                return f"Dense({units}, activation={activation}, name={name})"
            
            @staticmethod
            def Dropout(rate):
                return f"Dropout({rate})"
            
            @staticmethod
            def BatchNormalization():
                return "BatchNormalization"
        
        class Model:
            def __init__(self, inputs, outputs, name=None):
                self.inputs = inputs
                self.outputs = outputs
                self.name = name
                self.layers = []
                
            def compile(self, optimizer, loss, metrics):
                logger.info(f"Mock model compiled with optimizer={optimizer}, loss={loss}, metrics={metrics}")
                
            def summary(self):
                logger.info("Mock model summary - EfficientNet Lite architecture")
                
            def fit(self, *args, **kwargs):
                # Mock training with fake history
                epochs = kwargs.get('epochs', 10)
                return {
                    'loss': np.random.uniform(0.1, 2.0, epochs).tolist(),
                    'accuracy': np.random.uniform(0.1, 0.95, epochs).tolist(),
                    'val_loss': np.random.uniform(0.1, 2.5, epochs).tolist(),
                    'val_accuracy': np.random.uniform(0.1, 0.9, epochs).tolist()
                }
                
            def evaluate(self, *args, **kwargs):
                return [0.5, 0.85]  # Mock loss and accuracy
                
            def predict(self, x):
                batch_size = x.shape[0] if hasattr(x, 'shape') else 1
                num_classes = 34  # Food classes
                return np.random.softmax(np.random.randn(batch_size, num_classes))
                
            def save(self, path):
                logger.info(f"Mock model saved to {path}")
        
        class optimizers:
            @staticmethod
            def Adam(learning_rate=0.001):
                return f"Adam(lr={learning_rate})"
        
        class callbacks:
            class EarlyStopping:
                def __init__(self, monitor='val_loss', patience=5, restore_best_weights=True):
                    self.monitor = monitor
                    self.patience = patience
                    self.restore_best_weights = restore_best_weights
            
            class ReduceLROnPlateau:
                def __init__(self, monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001):
                    self.monitor = monitor
                    self.factor = factor
                    self.patience = patience
                    self.min_lr = min_lr
            
            class ModelCheckpoint:
                def __init__(self, filepath, monitor='val_accuracy', save_best_only=True):
                    self.filepath = filepath
                    self.monitor = monitor
                    self.save_best_only = save_best_only
        
        class applications:
            @staticmethod
            def EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)):
                class MockBaseModel:
                    def __init__(self):
                        self.trainable = True
                        self.output = "efficientnet_output"
                        
                return MockBaseModel()
    
    class lite:
        class TFLiteConverter:
            @staticmethod
            def from_keras_model(model):
                class Converter:
                    def convert(self):
                        return b"mock_tflite_model"
                return Converter()

# Try to import TensorFlow, fall back to mock if not available
try:
    import tensorflow as tf
    logger.info("Using real TensorFlow")
except ImportError:
    tf = MockTensorFlow()
    logger.warning("TensorFlow not found, using mock implementation for development")


class EfficientNetLiteModel:
    """
    EfficientNet Lite model for food classification with on-device optimization
    
    Features:
    - Lightweight architecture optimized for mobile devices
    - Transfer learning from ImageNet pretrained weights
    - Configurable architecture depth and width
    - TensorFlow Lite conversion support
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 base_model: str = 'EfficientNetB0',
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 freeze_base: bool = True):
        """
        Initialize EfficientNet Lite model
        
        Args:
            num_classes: Number of food classes to classify
            input_shape: Input image shape (height, width, channels)
            base_model: Base EfficientNet model to use
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            freeze_base: Whether to freeze base model weights initially
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model_name = base_model
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.freeze_base = freeze_base
        
        self.model = None
        self.base_model = None
        self.history = None
        
        logger.info(f"Initializing EfficientNet Lite model with {num_classes} classes")
    
    def build_model(self) -> None:
        """Build the complete EfficientNet Lite model"""
        logger.info("Building EfficientNet Lite model...")
        
        # Load pretrained base model
        if hasattr(tf.keras.applications, self.base_model_name):
            base_model_class = getattr(tf.keras.applications, self.base_model_name)
            self.base_model = base_model_class(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            # Fallback to EfficientNetB0
            self.base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Configure base model trainability with partial freezing
        if self.freeze_base:
            # Freeze the base model completely
            self.base_model.trainable = False
            logger.info("Base model frozen - only training classification head")
        else:
            # Unfreeze base model but freeze early layers for stable fine-tuning
            self.base_model.trainable = True
            # Freeze first 100 layers (early feature extractors)
            for layer in self.base_model.layers[:100]:
                layer.trainable = False
            # Allow fine-tuning of later layers (higher-level features)
            for layer in self.base_model.layers[100:]:
                layer.trainable = True
            
            trainable_count = sum([1 for layer in self.base_model.layers if layer.trainable])
            total_count = len(self.base_model.layers)
            logger.info(f"Partial fine-tuning: {trainable_count}/{total_count} layers trainable")
        
        # Build classification head
        inputs = self.base_model.input
        x = self.base_model.output
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Batch normalization (optional)
        if self.use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='batch_norm')(x)
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x)
        
        # Final classification layer
        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create the complete model
        self.model = tf.keras.Model(inputs, outputs, name='EfficientNet_Lite_Food_Classifier')
        
        logger.info("Model architecture built successfully")
        self.model.summary()
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'categorical_crossentropy',
                     metrics: List[str] = None) -> None:
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            optimizer: Optimizer to use
            learning_rate: Learning rate for optimization
            loss: Loss function to use
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def get_callbacks(self, 
                     checkpoint_path: Optional[str] = None,
                     early_stopping_patience: int = 10,
                     reduce_lr_patience: int = 5) -> List:
        """
        Get training callbacks for model optimization
        
        Args:
            checkpoint_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            List of callback objects
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=reduce_lr_patience,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Only add ModelCheckpoint if an explicit checkpoint path is provided
        if checkpoint_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )

        return callbacks
    
    def fine_tune(self, 
                  unfreeze_layers: int = 20,
                  fine_tune_lr: float = 0.0001) -> None:
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the top
            fine_tune_lr: Learning rate for fine-tuning
        """
        if self.base_model is None:
            raise ValueError("Base model not available for fine-tuning")
        
        # Unfreeze top layers
        self.base_model.trainable = True
        
        # Freeze all layers except the top ones
        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Fine-tuning enabled: unfroze top {unfreeze_layers} layers")
    
    def convert_to_tflite(self, 
                         output_path: Optional[str] = None,
                         quantize: bool = True,
                         optimize_for_size: bool = True) -> Optional[str]:
        """
        Convert the model to TensorFlow Lite format for deployment
        
        Args:
            output_path: Path to save the TFLite model
            quantize: Whether to apply quantization
            optimize_for_size: Whether to optimize for model size
            
        Returns:
            Path to the saved TFLite model
        """
        if self.model is None:
            raise ValueError("Model must be trained before conversion")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimizations
        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()

        # If an output path was provided, write the tflite file; otherwise return bytes
        if output_path:
            tflite_path = Path(output_path)
            tflite_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"TensorFlow Lite model saved to {tflite_path}")
            return str(tflite_path)
        else:
            logger.info("TensorFlow Lite conversion completed; no file written (output_path=None)")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "Model not built"}
        
        total_params = sum([np.prod(layer.get_weights()[0].shape) 
                           for layer in self.model.layers if layer.get_weights()])
        
        info = {
            "architecture": self.base_model_name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "total_parameters": total_params,
            "trainable_parameters": total_params,  # Simplified for mock
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "freeze_base": self.freeze_base,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Rough estimate
        }
        
        return info
    
    def save_model(self, save_path: str, format_type: str = 'keras', allow_save: bool = False) -> None:
        """
        Save the complete model (Keras 3 compatible)
        
        Args:
            save_path: Path to save the model
            format_type: Format to save ('keras', 'h5', 'tf')
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        if not allow_save:
            logger.warning("Model saving is disabled by default to avoid creating large files. Set allow_save=True to enable saving.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure proper file extension based on format
        if format_type == 'keras' and not str(save_path).endswith('.keras'):
            save_path = save_path.with_suffix('.keras')
        elif format_type == 'h5' and not str(save_path).endswith('.h5'):
            save_path = save_path.with_suffix('.h5')

        # Use Keras 3 compatible saving (no save_format argument)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise


class ModelFactory:
    """Factory class for creating different EfficientNet Lite variants"""
    
    @staticmethod
    def create_efficientnet_lite0(num_classes: int = 34) -> EfficientNetLiteModel:
        """Create EfficientNet-Lite0 model (smallest variant)"""
        return EfficientNetLiteModel(
            num_classes=num_classes,
            input_shape=(224, 224, 3),
            base_model='EfficientNetB0',
            dropout_rate=0.2
        )
    
    @staticmethod
    def create_efficientnet_lite1(num_classes: int = 34) -> EfficientNetLiteModel:
        """Create EfficientNet-Lite1 model (medium variant)"""
        return EfficientNetLiteModel(
            num_classes=num_classes,
            input_shape=(240, 240, 3),
            base_model='EfficientNetB1',
            dropout_rate=0.2
        )
    
    @staticmethod
    def create_custom_model(config: Dict[str, Any]) -> EfficientNetLiteModel:
        """Create custom model from configuration"""
        return EfficientNetLiteModel(**config)


def create_model(model_type: str = 'lite0', num_classes: int = 34) -> EfficientNetLiteModel:
    """
    Convenience function to create EfficientNet Lite models
    
    Args:
        model_type: Type of model to create ('lite0', 'lite1', 'custom')
        num_classes: Number of classes for classification
        
    Returns:
        EfficientNetLiteModel instance
    """
    factory = ModelFactory()
    
    if model_type == 'lite0':
        return factory.create_efficientnet_lite0(num_classes)
    elif model_type == 'lite1':
        return factory.create_efficientnet_lite1(num_classes)
    else:
        return factory.create_efficientnet_lite0(num_classes)


if __name__ == "__main__":
    # Example usage
    logger.info("Creating EfficientNet Lite model for food classification...")
    
    # Create model
    model = create_model('lite0', num_classes=34)
    
    # Build and compile
    model.build_model()
    model.compile_model()
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("EfficientNet Lite model created successfully!")