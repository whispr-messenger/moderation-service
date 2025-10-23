"""
Training and Hyperparameter Tuning Module for Food Classification

This module provides:
- Comprehensive training pipeline with advanced features
- Hyperparameter optimization and tuning
- Learning rate scheduling and optimization strategies  
- Training monitoring and logging
- Model validation and performance tracking
"""

import numpy as np
from typing import Dict, List, Any, Callable
import logging
import json
from pathlib import Path
import time
import random
from dataclasses import dataclass
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'
    validation_split: float = 0.2
    
    # Regularization parameters
    dropout_rate: float = 0.2
    weight_decay: float = 1e-4
    
    # Callback parameters
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_learning_rate: float = 1e-7
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Model checkpointing
    save_best_model: bool = True
    checkpoint_monitor: str = 'val_accuracy'
    checkpoint_mode: str = 'max'
    
    # Training strategy
    use_mixed_precision: bool = False
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 4


@dataclass 
class HyperparameterSpace:
    """Define hyperparameter search space"""
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    dropout_rates: List[float] = None
    optimizers: List[str] = None
    architectures: List[str] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [0.01, 0.001, 0.0001]
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3, 0.5]
        if self.optimizers is None:
            self.optimizers = ['adam', 'sgd', 'rmsprop']
        if self.architectures is None:
            self.architectures = ['lite0', 'lite1']


class LearningRateScheduler:
    """Advanced learning rate scheduling strategies"""
    
    @staticmethod
    def cosine_annealing(initial_lr: float, 
                        total_epochs: int, 
                        min_lr: float = 1e-7) -> Callable:
        """Cosine annealing learning rate schedule"""
        def schedule(epoch):
            cos_inner = (np.pi * epoch) / total_epochs
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(cos_inner))
        return schedule
    
    @staticmethod
    def warm_restart(initial_lr: float, 
                    restart_period: int = 10,
                    t_mult: int = 2,
                    m_mult: float = 0.5) -> Callable:
        """Cosine annealing with warm restarts"""
        def schedule(epoch):
            if epoch == 0:
                return initial_lr
            
            # Calculate current cycle
            cycle = 0
            epoch_in_cycle = epoch
            period = restart_period
            
            while epoch_in_cycle >= period:
                epoch_in_cycle -= period
                cycle += 1
                period *= t_mult
            
            # Cosine annealing within current cycle
            cos_inner = (np.pi * epoch_in_cycle) / period
            lr = initial_lr * (m_mult ** cycle) * 0.5 * (1 + np.cos(cos_inner))
            return max(lr, initial_lr * 1e-4)
        
        return schedule
    
    @staticmethod
    def exponential_decay(initial_lr: float, 
                         decay_rate: float = 0.95,
                         decay_steps: int = 5) -> Callable:
        """Exponential decay schedule"""
        def schedule(epoch):
            return initial_lr * (decay_rate ** (epoch // decay_steps))
        return schedule


class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.training_history = {}
        self.best_metrics = {}
        self.start_time = None
    
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        logger.info("Training started")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        if not self.training_history:
            for key in metrics.keys():
                self.training_history[key] = []
        
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Update best metrics
        if 'val_accuracy' in metrics:
            if 'best_val_accuracy' not in self.best_metrics or \
               metrics['val_accuracy'] > self.best_metrics['best_val_accuracy']:
                self.best_metrics['best_val_accuracy'] = metrics['val_accuracy']
                self.best_metrics['best_epoch'] = epoch
        
        logger.info(f"Epoch {epoch}: {metrics}")
    
    def end_training(self):
        """Mark the end of training and save logs"""
        if self.start_time:
            training_time = time.time() - self.start_time
            self.best_metrics['training_time_seconds'] = training_time
            
        # Save training history
        history_file = self.log_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save best metrics
        metrics_file = self.log_dir / "best_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
        
        logger.info(f"Training completed in {self.best_metrics.get('training_time_seconds', 0):.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_metrics.get('best_val_accuracy', 0):.4f}")


class ModelTrainer:
    """Comprehensive model training with advanced features"""
    
    def __init__(self, 
                 model,
                 data_loader, 
                 config: TrainingConfig = None):
        """
        Initialize the trainer
        
        Args:
            model: EfficientNet Lite model instance
            data_loader: Data loader instance
            config: Training configuration
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config or TrainingConfig()
        self.monitor = TrainingMonitor()
        self.training_history = {}
        
    def prepare_model(self):
        """Prepare model for training"""
        if not hasattr(self.model, 'model') or self.model.model is None:
            self.model.build_model()
        
        self.model.compile_model(
            learning_rate=self.config.learning_rate,
            optimizer=self.config.optimizer
        )
    
    def get_callbacks(self):
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        callbacks.append({
            'name': 'EarlyStopping',
            'params': {
                'monitor': 'val_loss',
                'patience': self.config.early_stopping_patience,
                'restore_best_weights': True
            }
        })
        
        # Learning rate reduction
        callbacks.append({
            'name': 'ReduceLROnPlateau', 
            'params': {
                'monitor': 'val_loss',
                'factor': self.config.reduce_lr_factor,
                'patience': self.config.reduce_lr_patience,
                'min_lr': self.config.min_learning_rate
            }
        })
        
        # Model checkpointing
        if self.config.save_best_model:
            callbacks.append({
                'name': 'ModelCheckpoint',
                'params': {
                    'filepath': 'best_model.h5',
                    'monitor': self.config.checkpoint_monitor,
                    'save_best_only': True
                }
            })
        
        return callbacks
    
    def train(self, 
              save_dir: str = "models",
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with comprehensive monitoring
        
        Args:
            save_dir: Directory to save model and results
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info("Starting model training...")
        self.monitor.start_training()
        
        # Prepare model
        self.prepare_model()
        
        # Mock training process (since we're using mock TensorFlow)
        history = self._mock_training()
        
        # Save training results
        self._save_training_results(save_path, history)
        
        self.monitor.end_training()
        return history
    
    def _mock_training(self) -> Dict[str, List[float]]:
        """Mock training process for development"""
        logger.info("Running mock training (replace with real training when TensorFlow is available)")
        
        # Generate realistic training curves
        epochs = self.config.epochs
        
        # Initialize metrics with some noise
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        # Simulate training progress
        base_train_acc = 0.1
        base_val_acc = 0.1
        
        for epoch in range(epochs):
            # Simulate learning with some randomness
            progress = epoch / epochs
            
            # Training accuracy gradually improves with some noise
            train_accuracy = min(0.95, base_train_acc + progress * 0.8 + np.random.normal(0, 0.02))
            train_accuracy = max(0.05, train_accuracy)
            
            # Validation accuracy improves more slowly with more noise
            val_accuracy = min(0.90, base_val_acc + progress * 0.7 + np.random.normal(0, 0.03))
            val_accuracy = max(0.05, val_accuracy)
            
            # Loss decreases over time
            train_l = max(0.1, 2.5 * np.exp(-progress * 2) + np.random.normal(0, 0.05))
            val_l = max(0.1, 2.8 * np.exp(-progress * 1.5) + np.random.normal(0, 0.1))
            
            train_acc.append(float(train_accuracy))
            val_acc.append(float(val_accuracy))
            train_loss.append(float(train_l))
            val_loss.append(float(val_l))
            
            # Log epoch metrics
            epoch_metrics = {
                'loss': train_l,
                'accuracy': train_accuracy,
                'val_loss': val_l,
                'val_accuracy': val_accuracy
            }
            self.monitor.log_epoch(epoch + 1, epoch_metrics)
        
        history = {
            'loss': train_loss,
            'accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        
        return history
    
    def _save_training_results(self, save_path: Path, history: Dict):
        """Save training results and model"""
        # Save training history
        history_file = save_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save training configuration
        config_file = save_path / "training_config.json"
        config_dict = {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'optimizer': self.config.optimizer,
            'dropout_rate': self.config.dropout_rate
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model (mock)
        model_path = save_path / "trained_model"
        self.model.save_model(str(model_path))
        
        logger.info(f"Training results saved to {save_path}")


class HyperparameterTuner:
    """Automated hyperparameter optimization"""
    
    def __init__(self, 
                 model_factory,
                 data_loader,
                 search_space: HyperparameterSpace = None,
                 max_trials: int = 20,
                 objective_metric: str = 'val_accuracy'):
        """
        Initialize hyperparameter tuner
        
        Args:
            model_factory: Function to create model instances
            data_loader: Data loader instance
            search_space: Hyperparameter search space
            max_trials: Maximum number of trials to run
            objective_metric: Metric to optimize
        """
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.search_space = search_space or HyperparameterSpace()
        self.max_trials = max_trials
        self.objective_metric = objective_metric
        
        self.trial_results = []
        self.best_params = None
        self.best_score = -np.inf if 'acc' in objective_metric else np.inf
    
    def generate_random_params(self) -> Dict[str, Any]:
        """Generate random hyperparameters from search space"""
        params = {
            'learning_rate': random.choice(self.search_space.learning_rates),
            'batch_size': random.choice(self.search_space.batch_sizes),
            'dropout_rate': random.choice(self.search_space.dropout_rates),
            'optimizer': random.choice(self.search_space.optimizers),
            'architecture': random.choice(self.search_space.architectures)
        }
        return params
    
    def run_trial(self, params: Dict[str, Any], trial_id: int) -> float:
        """Run a single hyperparameter trial"""
        logger.info(f"Trial {trial_id + 1}/{self.max_trials}: {params}")
        
        # Create model with current parameters
        model = self.model_factory(
            model_type=params['architecture'],
            num_classes=self.data_loader.num_classes
        )
        
        # Update model parameters
        model.dropout_rate = params['dropout_rate']
        
        # Create training config
        config = TrainingConfig(
            epochs=10,  # Reduced for hyperparameter search
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer'],
            dropout_rate=params['dropout_rate']
        )
        
        # Train model
        trainer = ModelTrainer(model, self.data_loader, config)
        history = trainer.train(save_dir=f"trial_{trial_id}", verbose=False)
        
        # Get final metric score
        if self.objective_metric in history:
            score = history[self.objective_metric][-1]  # Last epoch score
        else:
            score = 0.5  # Default score if metric not found
        
        # Record trial result
        result = {
            'trial_id': trial_id,
            'params': params,
            'score': score,
            'history': history
        }
        self.trial_results.append(result)
        
        logger.info(f"Trial {trial_id + 1} score: {score:.4f}")
        return score
    
    def optimize(self, 
                search_method: str = 'random',
                save_results: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            search_method: Search method ('random', 'grid')
            save_results: Whether to save optimization results
            
        Returns:
            Best hyperparameters and results
        """
        logger.info(f"Starting hyperparameter optimization with {search_method} search")
        
        if search_method == 'random':
            self._random_search()
        elif search_method == 'grid':
            self._grid_search()
        else:
            raise ValueError(f"Unknown search method: {search_method}")
        
        # Find best trial
        if 'acc' in self.objective_metric:
            best_trial = max(self.trial_results, key=lambda x: x['score'])
        else:
            best_trial = min(self.trial_results, key=lambda x: x['score'])
        
        self.best_params = best_trial['params']
        self.best_score = best_trial['score']
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        if save_results:
            self._save_optimization_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_trials': self.trial_results
        }
    
    def _random_search(self):
        """Perform random hyperparameter search"""
        for trial_id in range(self.max_trials):
            params = self.generate_random_params()
            self.run_trial(params, trial_id)
    
    def _grid_search(self):
        """Perform grid hyperparameter search"""
        param_combinations = list(itertools.product(
            self.search_space.learning_rates,
            self.search_space.batch_sizes,
            self.search_space.dropout_rates,
            self.search_space.optimizers,
            self.search_space.architectures
        ))
        
        # Limit to max_trials
        param_combinations = param_combinations[:self.max_trials]
        
        for trial_id, (lr, bs, dr, opt, arch) in enumerate(param_combinations):
            params = {
                'learning_rate': lr,
                'batch_size': bs, 
                'dropout_rate': dr,
                'optimizer': opt,
                'architecture': arch
            }
            self.run_trial(params, trial_id)
    
    def _save_optimization_results(self):
        """Save hyperparameter optimization results"""
        results_dir = Path("hyperparameter_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save all results
        results_file = results_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)
        
        # Save best parameters
        best_file = results_dir / "best_parameters.json"
        with open(best_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'objective_metric': self.objective_metric
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_dir}")


def create_trainer(model, data_loader, **config_kwargs) -> ModelTrainer:
    """
    Convenience function to create a model trainer
    
    Args:
        model: EfficientNet Lite model instance
        data_loader: Data loader instance
        **config_kwargs: Configuration parameters
        
    Returns:
        Configured ModelTrainer instance
    """
    config = TrainingConfig(**config_kwargs)
    return ModelTrainer(model, data_loader, config)


def create_tuner(model_factory, data_loader, **tuner_kwargs) -> HyperparameterTuner:
    """
    Convenience function to create a hyperparameter tuner
    
    Args:
        model_factory: Function to create model instances
        data_loader: Data loader instance
        **tuner_kwargs: Tuner parameters
        
    Returns:
        Configured HyperparameterTuner instance
    """
    return HyperparameterTuner(model_factory, data_loader, **tuner_kwargs)


if __name__ == "__main__":
    # Example usage
    from model import create_model
    from data_loader import FoodDataLoader
    
    logger.info("Setting up training pipeline...")
    
    # Create data loader
    dataset_path = "../../Food Classification dataset"
    data_loader = FoodDataLoader(dataset_path, batch_size=32)
    
    # Create model
    model = create_model('lite0', num_classes=data_loader.num_classes)
    
    # Create and run trainer
    config = TrainingConfig(epochs=5, learning_rate=0.001)
    trainer = ModelTrainer(model, data_loader, config)
    history = trainer.train()
    
    print("Training completed successfully!")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")