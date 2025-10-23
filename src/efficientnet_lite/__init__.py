"""
EfficientNet-Lite Food Classification Package

A comprehensive on-device food classification system using EfficientNet Lite architecture.
Includes data loading, model training, evaluation, and deployment capabilities.
"""

# Core pipeline components
from .main import FoodClassificationPipeline
from .data_loader import FoodDataLoader, DataAugmentation, create_data_loader
from .model import EfficientNetLiteModel, ModelFactory, create_model
from .trainer import ModelTrainer, TrainingConfig, HyperparameterSpace, create_trainer, create_tuner
from .evaluator import ModelEvaluator, InferenceEngine, create_evaluator, create_inference_engine
from .visualizer import DataVisualizer, create_visualizer
from .config import ProjectConfig, ConfigurationManager, create_default_config

# Version information
__version__ = "1.0.0"
__author__ = "Food Classification AI Team"

# Main exports
__all__ = [
    # Main pipeline
    'FoodClassificationPipeline',
    
    # Data components
    'FoodDataLoader', 
    'DataAugmentation',
    'create_data_loader',
    
    # Model components
    'EfficientNetLiteModel',
    'ModelFactory', 
    'create_model',
    
    # Training components
    'ModelTrainer',
    'TrainingConfig',
    'HyperparameterSpace',
    'create_trainer',
    'create_tuner',
    
    # Evaluation components
    'ModelEvaluator',
    'InferenceEngine', 
    'create_evaluator',
    'create_inference_engine',
    
    # Visualization components
    'DataVisualizer',
    'create_visualizer',
    
    # Configuration components
    'ProjectConfig',
    'ConfigurationManager',
    'create_default_config'
]