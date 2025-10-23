"""
Main Pipeline Script for Food Classification Project

This is the central orchestration script that brings together all components:
- Data loading and preprocessing
- Model training and evaluation
- Hyperparameter tuning
- Data visualization
- Model deployment
- CLI interface for easy usage
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

# Import project modules
from data_loader import FoodDataLoader, create_data_loader
from model import EfficientNetLiteModel, ModelFactory, create_model
from trainer import ModelTrainer, TrainingConfig, HyperparameterSpace, create_trainer, create_tuner
from evaluator import ModelEvaluator, InferenceEngine, create_evaluator, create_inference_engine
from visualizer import DataVisualizer, create_visualizer
from config import ConfigurationManager, create_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FoodClassificationPipeline:
    """
    Complete pipeline for food classification with EfficientNet Lite
    
    This class orchestrates the entire machine learning pipeline from
    data loading to model deployment.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigurationManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.data_loader: Optional[FoodDataLoader] = None
        self.model: Optional[EfficientNetLiteModel] = None
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.visualizer: Optional[DataVisualizer] = None
        
        logger.info(f"Initialized pipeline for project: {self.config.project_name}")
    
    def setup_data(self) -> FoodDataLoader:
        """
        Set up data loader with configuration
        
        Returns:
            Configured FoodDataLoader instance
        """
        logger.info("Setting up data loader...")
        
        self.data_loader = create_data_loader(
            dataset_path=self.config.data.dataset_path,
            image_size=self.config.data.image_size,
            batch_size=self.config.data.batch_size,
            validation_split=self.config.data.validation_split,
            test_split=self.config.data.test_split,
            augmentation=self.config.data.use_augmentation
        )
        
        # Update model config with actual number of classes
        self.config.model.num_classes = len(self.data_loader.class_names)
        
        logger.info(f"Data loaded: {len(self.data_loader.class_names)} classes, {len(self.data_loader.file_paths)} images")
        return self.data_loader
    
    def setup_model(self) -> EfficientNetLiteModel:
        """
        Set up model with configuration
        
        Returns:
            Configured EfficientNetLiteModel instance
        """
        logger.info("Setting up model...")
        
        if self.data_loader is None:
            self.setup_data()
        
        self.model = create_model(
            model_type='lite0',  # Based on architecture config
            num_classes=self.config.model.num_classes
        )
        
        # Apply model configuration
        self.model.dropout_rate = self.config.model.dropout_rate
        self.model.use_batch_norm = self.config.model.use_batch_norm
        self.model.freeze_base = self.config.model.freeze_base
        
        logger.info(f"Model created: {self.config.model.architecture} with {self.config.model.num_classes} classes")
        return self.model
    
    def setup_visualizer(self) -> DataVisualizer:
        """
        Set up data visualizer
        
        Returns:
            Configured DataVisualizer instance
        """
        self.visualizer = create_visualizer()
        return self.visualizer
    
    def explore_data(self, save_dir: str = "data_exploration") -> None:
        """
        Perform comprehensive data exploration and visualization
        
        Args:
            save_dir: Directory to save exploration results
        """
        logger.info("Starting data exploration...")
        
        if self.data_loader is None:
            self.setup_data()
        
        if self.visualizer is None:
            self.setup_visualizer()
        
        # Create comprehensive dashboard
        self.visualizer.create_dashboard(
            self.data_loader,
            save_dir=save_dir
        )
        
        # Save dataset statistics
        stats = self.data_loader.get_dataset_stats()
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "dataset_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save class mapping
        self.data_loader.save_class_mapping(str(save_path / "class_mapping.json"))
        
        logger.info(f"Data exploration completed. Results saved to {save_dir}")
    
    def train_model(self, 
                   save_dir: str = "training_results",
                   custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model with comprehensive monitoring
        
        Args:
            save_dir: Directory to save training results
            custom_config: Custom training configuration
            
        Returns:
            Training history and results
        """
        logger.info("Starting model training...")
        
        # Setup components if needed
        if self.data_loader is None:
            self.setup_data()
        if self.model is None:
            self.setup_model()
        
        # Create training configuration
        training_config = TrainingConfig(
            epochs=self.config.training.epochs,
            batch_size=self.config.data.batch_size,
            learning_rate=self.config.training.learning_rate,
            optimizer=self.config.training.optimizer,
            dropout_rate=self.config.training.dropout_rate
        )
        
        # Apply custom configuration if provided
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(training_config, key):
                    setattr(training_config, key, value)
        
        # Create trainer
        self.trainer = create_trainer(self.model, self.data_loader, **training_config.__dict__)
        
        # Train model
        history = self.trainer.train(save_dir=save_dir)
        
        # Visualize training results
        if self.visualizer is None:
            self.setup_visualizer()
        
        self.visualizer.plot_training_history(
            history,
            save_path=f"{save_dir}/training_history.png"
        )
        
        logger.info("Model training completed")
        return history
    
    def tune_hyperparameters(self,
                           max_trials: int = None,
                           search_method: str = 'random',
                           save_dir: str = "hyperparameter_results") -> Dict[str, Any]:
        """
        Perform hyperparameter optimization
        
        Args:
            max_trials: Maximum number of trials
            search_method: Search method ('random', 'grid')
            save_dir: Directory to save results
            
        Returns:
            Optimization results
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Setup components if needed
        if self.data_loader is None:
            self.setup_data()
        
        # Create search space
        search_space = HyperparameterSpace(
            learning_rates=self.config.hyperparameter.learning_rates,
            batch_sizes=self.config.hyperparameter.batch_sizes,
            dropout_rates=self.config.hyperparameter.dropout_rates,
            optimizers=self.config.hyperparameter.optimizers,
            architectures=self.config.hyperparameter.architectures
        )
        
        # Create tuner
        max_trials = max_trials or self.config.hyperparameter.max_trials
        tuner = create_tuner(
            ModelFactory.create_efficientnet_lite0,
            self.data_loader,
            search_space=search_space,
            max_trials=max_trials,
            objective_metric=self.config.hyperparameter.objective_metric
        )
        
        # Run optimization
        results = tuner.optimize(search_method=search_method)
        
        # Save results
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "optimization_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Hyperparameter optimization completed. Best score: {results['best_score']:.4f}")
        logger.info(f"Best parameters: {results['best_params']}")
        
        return results
    
    def evaluate_model(self,
                      split: str = 'test',
                      save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            split: Data split to evaluate ('train', 'val', 'test')
            save_dir: Directory to save results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting model evaluation on {split} set...")
        
        # Setup components if needed
        if self.data_loader is None:
            self.setup_data()
        if self.model is None:
            self.setup_model()
        
        # Create evaluator
        self.evaluator = create_evaluator(self.model, self.data_loader.class_names)
        
        # Evaluate model
        results = self.evaluator.evaluate_model(
            self.data_loader,
            split=split,
            save_results=True,
            results_dir=save_dir
        )
        
        # Create visualizations
        if self.visualizer is None:
            self.setup_visualizer()
        
        # Plot confusion matrix
        self.visualizer.plot_confusion_matrix(
            results.confusion_matrix,
            self.data_loader.class_names,
            save_path=f"{save_dir}/confusion_matrix.png"
        )
        
        # Plot performance metrics
        performance_metrics = {
            'Accuracy': results.accuracy,
            'Precision': results.precision,
            'Recall': results.recall,
            'F1-Score': results.f1_score
        }
        
        self.visualizer.plot_model_performance(
            performance_metrics,
            save_path=f"{save_dir}/performance_metrics.png"
        )
        
        # Plot per-class performance
        self.visualizer.plot_class_performance(
            results.class_report,
            save_path=f"{save_dir}/class_performance.png"
        )
        
        logger.info("Model evaluation completed")
        return {
            'accuracy': results.accuracy,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'confusion_matrix': results.confusion_matrix.tolist(),
            'class_report': results.class_report,
            'inference_time': results.inference_time
        }
    
    def setup_inference(self) -> InferenceEngine:
        """
        Set up inference engine for real-time predictions
        
        Returns:
            Configured InferenceEngine instance
        """
        if self.data_loader is None:
            self.setup_data()
        if self.model is None:
            self.setup_model()
        
        self.inference_engine = create_inference_engine(
            self.model,
            self.data_loader.class_names
        )
        
        return self.inference_engine
    
    def predict_image(self, 
                     image_path: str,
                     top_k: int = 5) -> Dict[str, Any]:
        """
        Predict food class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Prediction results
        """
        if self.inference_engine is None:
            self.setup_inference()
        
        result = self.inference_engine.predict_image(image_path, top_k=top_k)
        
        return {
            'image_path': result.image_path,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'top_predictions': result.top_k_predictions,
            'inference_time': result.inference_time
        }
    
    def benchmark_inference(self, 
                           num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            num_samples: Number of test samples
            
        Returns:
            Benchmark results
        """
        if self.inference_engine is None:
            self.setup_inference()
        
        return self.inference_engine.benchmark_inference(num_samples=num_samples)
    
    def export_model(self, 
                    output_path: str = "deployed_model",
                    format_type: str = 'tflite') -> str:
        """
        Export model for deployment
        
        Args:
            output_path: Path to save exported model
            format_type: Export format ('tflite', 'savedmodel')
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model must be trained before export")
        
        if format_type.lower() == 'tflite':
            output_file = f"{output_path}.tflite"
            exported_path = self.model.convert_to_tflite(
                output_file,
                quantize=self.config.deployment.quantization,
                optimize_for_size=self.config.deployment.tflite_optimize_for_size
            )
        else:
            output_file = output_path / "model.keras"
            self.model.save_model(output_file, format_type='keras')
            exported_path = output_file
        
        logger.info(f"Model exported to {exported_path}")
        return exported_path
    
    def run_complete_pipeline(self, 
                             output_dir: str = "pipeline_results",
                             include_tuning: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline from data exploration to model deployment
        
        Args:
            output_dir: Directory to save all results
            include_tuning: Whether to include hyperparameter tuning
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete pipeline execution...")
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        try:
            # 1. Data exploration
            logger.info("Step 1: Data Exploration")
            self.explore_data(save_dir=str(output_path / "data_exploration"))
            results['data_exploration'] = "completed"
            
            # 2. Hyperparameter tuning (optional)
            if include_tuning:
                logger.info("Step 2: Hyperparameter Optimization")
                tuning_results = self.tune_hyperparameters(
                    save_dir=str(output_path / "hyperparameter_results")
                )
                results['hyperparameter_tuning'] = tuning_results
            
            # 3. Model training
            logger.info("Step 3: Model Training")
            training_results = self.train_model(
                save_dir=str(output_path / "training_results")
            )
            results['training'] = training_results
            
            # 4. Model evaluation
            logger.info("Step 4: Model Evaluation")
            evaluation_results = self.evaluate_model(
                save_dir=str(output_path / "evaluation_results")
            )
            results['evaluation'] = evaluation_results
            
            # 5. Inference benchmarking
            logger.info("Step 5: Inference Benchmarking")
            benchmark_results = self.benchmark_inference()
            results['benchmark'] = benchmark_results
            
            # 6. Model export
            logger.info("Step 6: Model Export")
            exported_path = self.export_model(
                output_path=str(output_path / "deployed_model")
            )
            results['export'] = {'path': exported_path}
            
            # Save complete results
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['status'] = 'completed'
            
            with open(output_path / "pipeline_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Complete pipeline executed successfully in {execution_time:.2f} seconds")
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results


def main():
    """Main CLI interface for the food classification pipeline"""
    parser = argparse.ArgumentParser(
        description="Food Classification Pipeline with EfficientNet Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode pipeline --output results

  # Explore data only
  python main.py --mode explore --output data_exploration

  # Train model
  python main.py --mode train --config config.yaml --output training_results

  # Evaluate model
  python main.py --mode evaluate --output evaluation_results

  # Predict single image
  python main.py --mode predict --image path/to/image.jpg

  # Hyperparameter tuning
  python main.py --mode tune --trials 50 --output tuning_results
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['pipeline', 'explore', 'train', 'evaluate', 'predict', 'tune', 'benchmark', 'export'],
                       required=True,
                       help='Pipeline mode to run')
    
    parser.add_argument('--config', 
                       type=str,
                       default='config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--output', 
                       type=str,
                       default='results',
                       help='Output directory for results')
    
    parser.add_argument('--image', 
                       type=str,
                       help='Path to image file for prediction')
    
    parser.add_argument('--trials', 
                       type=int,
                       default=20,
                       help='Number of hyperparameter tuning trials')
    
    parser.add_argument('--samples', 
                       type=int,
                       default=100,
                       help='Number of samples for benchmarking')
    
    parser.add_argument('--create-config', 
                       action='store_true',
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    try:
        # Create default configuration if requested
        if args.create_config:
            create_default_config(args.config)
            print(f"Default configuration created at {args.config}")
            return
        
        # Initialize pipeline
        pipeline = FoodClassificationPipeline(args.config)
        
        # Execute based on mode
        if args.mode == 'pipeline':
            results = pipeline.run_complete_pipeline(
                output_dir=args.output,
                include_tuning=False
            )
            print("Complete pipeline executed successfully!")
            print(f"Final accuracy: {results.get('evaluation', {}).get('accuracy', 'N/A'):.4f}")
            
        elif args.mode == 'explore':
            pipeline.explore_data(save_dir=args.output)
            print("Data exploration completed!")
            
        elif args.mode == 'train':
            history = pipeline.train_model(save_dir=args.output)
            print("Model training completed!")
            print(f"Final validation accuracy: {history.get('val_accuracy', [0])[-1]:.4f}")
            
        elif args.mode == 'evaluate':
            results = pipeline.evaluate_model(save_dir=args.output)
            print("Model evaluation completed!")
            print(f"Test accuracy: {results['accuracy']:.4f}")
            print(f"Test F1-score: {results['f1_score']:.4f}")
            
        elif args.mode == 'predict':
            if not args.image:
                print("Error: --image argument required for prediction mode")
                sys.exit(1)
            
            result = pipeline.predict_image(args.image)
            print(f"Prediction for {args.image}:")
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print("  Top predictions:")
            for pred in result['top_predictions']:
                print(f"    {pred['class_name']}: {pred['confidence']:.4f}")
            
        elif args.mode == 'tune':
            results = pipeline.tune_hyperparameters(
                max_trials=args.trials,
                save_dir=args.output
            )
            print("Hyperparameter optimization completed!")
            print(f"Best score: {results['best_score']:.4f}")
            print(f"Best parameters: {results['best_params']}")
            
        elif args.mode == 'benchmark':
            results = pipeline.benchmark_inference(num_samples=args.samples)
            print("Inference benchmarking completed!")
            print(f"Mean inference time: {results['mean_inference_time']:.4f}s")
            print(f"FPS: {results['frames_per_second']:.2f}")
            
        elif args.mode == 'export':
            exported_path = pipeline.export_model(output_path=args.output)
            print(f"Model exported to {exported_path}")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
