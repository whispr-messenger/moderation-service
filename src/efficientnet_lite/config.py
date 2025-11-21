"""
Configuration Management Module for Food Classification Project

This module handles:
- Project configuration management
- Model parameter configurations
- Training and deployment settings
- Environment-specific configurations
- Configuration validation and defaults
"""

import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data-related configuration parameters"""
    dataset_path: str = "../../Food Classification dataset/Train"
    image_size: tuple = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    use_augmentation: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation parameters
    rotation_range: float = 15.0
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)
    saturation_range: tuple = (0.8, 1.2)
    blur_probability: float = 0.1
    flip_probability: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = 'EfficientNetB0'
    num_classes: int = 7
    input_shape: tuple = (224, 224, 3)
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    freeze_base: bool = False
    
    # Transfer learning parameters
    pretrained_weights: str = 'imagenet'
    unfreeze_layers: int = 20
    fine_tune_learning_rate: float = 0.0001
    
    # Model optimization
    use_mixed_precision: bool = False
    quantization: bool = True
    optimize_for_mobile: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    epochs: int = 10
    learning_rate: float = 0.0001
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'
    metrics: list = field(default_factory=lambda: ['accuracy'])
    
    # Regularization
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = 'cosine_annealing'  # 'cosine_annealing', 'reduce_on_plateau', 'exponential'
    lr_patience: int = 5
    lr_factor: float = 0.2
    min_learning_rate: float = 1e-7
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_loss'
    early_stopping_mode: str = 'min'
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_monitor: str = 'val_accuracy'
    checkpoint_mode: str = 'max'
    checkpoint_frequency: int = 1


@dataclass
class HyperparameterConfig:
    """Hyperparameter tuning configuration"""
    search_method: str = 'random'  # 'random', 'grid', 'bayesian'
    max_trials: int = 20
    objective_metric: str = 'val_accuracy'
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    
    # Search space
    learning_rates: list = field(default_factory=lambda: [0.01, 0.001, 0.0001])
    batch_sizes: list = field(default_factory=lambda: [16, 32, 64])
    dropout_rates: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    optimizers: list = field(default_factory=lambda: ['adam', 'sgd', 'rmsprop'])
    architectures: list = field(default_factory=lambda: ['EfficientNetB0', 'EfficientNetB1'])


@dataclass
class EvaluationConfig:
    """Model evaluation configuration"""
    metrics: list = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1_score'])
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    calculate_top_k: bool = True
    top_k_values: list = field(default_factory=lambda: [1, 3, 5])
    
    # Inference benchmarking
    benchmark_samples: int = 100
    benchmark_image_sizes: list = field(default_factory=lambda: [(224, 224), (240, 240)])


@dataclass
class DeploymentConfig:
    """Model deployment configuration"""
    export_format: str = 'tflite'  # 'tflite', 'onnx', 'torchscript', 'savedmodel'
    quantization: bool = True
    optimization_level: str = 'default'  # 'default', 'aggressive', 'size'
    
    # TensorFlow Lite specific
    tflite_quantize_int8: bool = False
    tflite_quantize_float16: bool = True
    tflite_optimize_for_size: bool = True
    
    # Model serving
    serve_model: bool = False
    serving_port: int = 8000
    max_batch_size: int = 32
    enable_gpu: bool = False


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Experiment tracking
    use_tensorboard: bool = True
    tensorboard_dir: str = 'tensorboard_logs'
    
    # Model monitoring
    save_training_plots: bool = True
    save_evaluation_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'


@dataclass
class ProjectConfig:
    """Complete project configuration"""
    project_name: str = "food_classification_efficientnet_lite"
    version: str = "1.0.0"
    description: str = "Food classification using EfficientNet Lite on-device model"
    
    # Configuration sections
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hyperparameter: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = "Food Classification AI"
    environment: str = "development"  # 'development', 'testing', 'production'


class ConfigurationManager:
    """Manage project configurations with validation and persistence"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[ProjectConfig] = None
        
    def load_config(self, config_path: Optional[str] = None) -> ProjectConfig:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded ProjectConfig instance
        """
        path = Path(config_path) if config_path else self.config_path
        
        if path is None or not path.exists():
            logger.warning(f"Configuration file not found: {path}. Using default configuration.")
            self.config = ProjectConfig()
            return self.config
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            self.config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            logger.info("Using default configuration")
            self.config = ProjectConfig()
        
        return self.config
    
    def save_config(self, 
                   config: Optional[ProjectConfig] = None,
                   config_path: Optional[str] = None,
                   format_type: str = 'yaml') -> None:
        """
        Save configuration to file
        
        Args:
            config: ProjectConfig instance to save
            config_path: Path to save configuration
            format_type: File format ('yaml', 'json')
        """
        cfg = config or self.config
        if cfg is None:
            raise ValueError("No configuration to save")
        
        path = Path(config_path) if config_path else self.config_path
        if path is None:
            path = Path(f"config.{format_type}")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(cfg)
        
        try:
            with open(path, 'w') as f:
                if format_type.lower() == 'yaml' or format_type.lower() == 'yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            raise
    
    def create_default_config(self, save_path: str = "config.yaml") -> ProjectConfig:
        """
        Create and save default configuration
        
        Args:
            save_path: Path to save default configuration
            
        Returns:
            Default ProjectConfig instance
        """
        config = ProjectConfig()
        self.save_config(config, save_path)
        self.config = config
        return config
    
    def validate_config(self, config: Optional[ProjectConfig] = None) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: ProjectConfig instance to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        cfg = config or self.config
        if cfg is None:
            logger.error("No configuration to validate")
            return False
        
        validation_errors = []
        
        # Validate data configuration
        if cfg.data.batch_size <= 0:
            validation_errors.append("batch_size must be positive")
        
        if not (0 < cfg.data.validation_split < 1):
            validation_errors.append("validation_split must be between 0 and 1")
        
        if not (0 < cfg.data.test_split < 1):
            validation_errors.append("test_split must be between 0 and 1")
        
        if cfg.data.validation_split + cfg.data.test_split >= 1:
            validation_errors.append("validation_split + test_split must be less than 1")
        
        # Validate model configuration
        if cfg.model.num_classes <= 0:
            validation_errors.append("num_classes must be positive")
        
        if not (0 <= cfg.model.dropout_rate <= 1):
            validation_errors.append("dropout_rate must be between 0 and 1")
        
        # Validate training configuration
        if cfg.training.epochs <= 0:
            validation_errors.append("epochs must be positive")
        
        if cfg.training.learning_rate <= 0:
            validation_errors.append("learning_rate must be positive")
        
        # Log validation results
        if validation_errors:
            logger.error("Configuration validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_config(self) -> ProjectConfig:
        """
        Get current configuration
        
        Returns:
            Current ProjectConfig instance
        """
        if self.config is None:
            self.config = ProjectConfig()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        if self.config is None:
            self.config = ProjectConfig()
        
        # Update configuration recursively
        self._update_config_recursive(self.config, updates)
        
        logger.info("Configuration updated")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ProjectConfig:
        """Convert dictionary to ProjectConfig"""
        # Create nested configuration objects
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        hyperparameter_config = HyperparameterConfig(**config_dict.get('hyperparameter', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        deployment_config = DeploymentConfig(**config_dict.get('deployment', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Create main configuration
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['data', 'model', 'training', 'hyperparameter', 
                                   'evaluation', 'deployment', 'logging']}
        
        return ProjectConfig(
            data=data_config,
            model=model_config,
            training=training_config,
            hyperparameter=hyperparameter_config,
            evaluation=evaluation_config,
            deployment=deployment_config,
            logging=logging_config,
            **main_config
        )
    
    def _update_config_recursive(self, config_obj: Any, updates: Dict[str, Any]) -> None:
        """Recursively update configuration object"""
        for key, value in updates.items():
            if hasattr(config_obj, key):
                if isinstance(value, dict) and hasattr(getattr(config_obj, key), '__dict__'):
                    # Recursively update nested objects
                    self._update_config_recursive(getattr(config_obj, key), value)
                else:
                    # Update simple value
                    setattr(config_obj, key, value)


def load_config(config_path: str = "config.yaml") -> ProjectConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ProjectConfig instance
    """
    manager = ConfigurationManager(config_path)
    return manager.load_config()


def save_config(config: ProjectConfig, 
               config_path: str = "config.yaml",
               format_type: str = 'yaml') -> None:
    """
    Convenience function to save configuration
    
    Args:
        config: ProjectConfig instance to save
        config_path: Path to save configuration
        format_type: File format ('yaml', 'json')
    """
    manager = ConfigurationManager()
    manager.save_config(config, config_path, format_type)


def create_default_config(config_path: str = "config.yaml") -> ProjectConfig:
    """
    Convenience function to create default configuration
    
    Args:
        config_path: Path to save default configuration
        
    Returns:
        Default ProjectConfig instance
    """
    manager = ConfigurationManager()
    return manager.create_default_config(config_path)


if __name__ == "__main__":
    # Example usage
    logger.info("Creating default configuration...")
    
    # Create configuration manager
    manager = ConfigurationManager()
    
    # Create and save default configuration
    config = manager.create_default_config("config.yaml")
    
    # Validate configuration
    is_valid = manager.validate_config()
    
    # Update some configuration values
    updates = {
        'training': {
            'epochs': 100,
            'learning_rate': 0.0005
        },
        'model': {
            'dropout_rate': 0.3
        }
    }
    manager.update_config(updates)
    
    # Save updated configuration
    manager.save_config(config_path="updated_config.yaml")
    
    print("Configuration management example completed successfully!")
    print(f"Project: {config.project_name}")
    print(f"Model architecture: {config.model.architecture}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")