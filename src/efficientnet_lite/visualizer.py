"""
Data Visualization Module for Food Classification Project

This module provides comprehensive visualization tools for:
- Dataset exploration and analysis
- Class distribution visualization
- Sample image displays
- Training metrics and performance
- Model evaluation results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")


class DataVisualizer:
    """Comprehensive visualization tools for food classification dataset"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Initialize the visualizer
        
        Args:
            figsize: Default figure size for plots
            style: Seaborn style to use
        """
        self.figsize = figsize
        sns.set_style(style)
        
    def plot_class_distribution(self, 
                              class_distribution: Dict[str, int],
                              title: str = "Food Classes Distribution",
                              save_path: Optional[str] = "results/class_distribution.png",
                              show_values: bool = True) -> None:
        """
        Plot the distribution of classes in the dataset
        
        Args:
            class_distribution: Dictionary with class names as keys and counts as values
            title: Title for the plot
            save_path: Path to save the plot (optional)
            show_values: Whether to show values on bars
        """
        plt.figure(figsize=(15, 8))
        
        # Sort classes by count for better visualization
        sorted_items = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_items)
        
        # Create bar plot
        bars = plt.bar(range(len(classes)), counts, alpha=0.8, 
                      color=sns.color_palette("husl", len(classes)))
        
        # Add value labels on bars if requested
        if show_values:
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Food Classes', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_dataset_statistics(self, 
                               stats: Dict[str, Any],
                               save_path: Optional[str] = "results/dataset_statistics.png") -> None:
        """
        Plot comprehensive dataset statistics
        
        Args:
            stats: Dataset statistics dictionary
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Statistics Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall dataset composition
        ax1 = axes[0, 0]
        split_data = [
            stats['train_images'],
            stats['validation_images'], 
            stats['test_images']
        ]
        split_labels = ['Train', 'Validation', 'Test']
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        wedges, texts, autotexts = ax1.pie(split_data, labels=split_labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax1.set_title('Dataset Split Distribution', fontweight='bold')
        
        # Plot 2: Class size statistics
        ax2 = axes[0, 1]
        class_stats = [
            stats['min_class_size'],
            stats['mean_class_size'],
            stats['max_class_size']
        ]
        stat_labels = ['Min', 'Mean', 'Max']
        bars = ax2.bar(stat_labels, class_stats, color=['red', 'blue', 'green'], alpha=0.7)
        ax2.set_title('Class Size Statistics', fontweight='bold')
        ax2.set_ylabel('Number of Images')
        
        # Add value labels on bars
        for bar, value in zip(bars, class_stats):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 3: Top 10 largest classes
        ax3 = axes[1, 0]
        sorted_classes = sorted(stats['class_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)
        top_classes = sorted_classes[:10]
        classes, counts = zip(*top_classes)
        
        bars = ax3.barh(range(len(classes)), counts, alpha=0.8)
        ax3.set_yticks(range(len(classes)))
        ax3.set_yticklabels(classes)
        ax3.set_xlabel('Number of Images')
        ax3.set_title('Top 10 Largest Classes', fontweight='bold')
        ax3.invert_yaxis()
        
        # Plot 4: Dataset summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Dataset Summary:
        
        Total Images: {stats['total_images']:,}
        Number of Classes: {stats['num_classes']}
        
        Training Images: {stats['train_images']:,}
        Validation Images: {stats['validation_images']:,}
        Test Images: {stats['test_images']:,}
        
        Class Statistics:
        • Min class size: {stats['min_class_size']}
        • Max class size: {stats['max_class_size']}
        • Mean class size: {stats['mean_class_size']:.1f}
        • Std class size: {stats['std_class_size']:.1f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dataset statistics plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_sample_images(self, 
                          images: List[np.ndarray],
                          labels: List[str],
                          title: str = "Sample Food Images",
                          save_path: Optional[str] = "results/sample_images.png",
                          grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Display a grid of sample images from the dataset
        
        Args:
            images: List of image arrays
            labels: List of corresponding labels
            title: Title for the plot
            save_path: Path to save the plot (optional)
            grid_size: Grid dimensions (rows, cols). Auto-calculated if None
        """
        n_images = len(images)
        if n_images == 0:
            logger.warning("No images provided for visualization")
            return
        
        # Calculate grid size if not provided
        if grid_size is None:
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Handle single image case
        if n_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            if i < n_images:
                # Display image
                axes[i].imshow(images[i])
                axes[i].set_title(labels[i], fontsize=12, fontweight='bold')
                axes[i].axis('off')
            else:
                # Hide empty subplots
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample images plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = "results/training_history.png") -> None:
        """
        Plot training history including loss and accuracy curves
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot training and validation loss
        if 'loss' in history and 'val_loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0].set_title('Model Loss', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot training and validation accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[1].set_title('Model Accuracy', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            normalize: bool = True,
                            save_path: Optional[str] = "results/confusion_matrix.png") -> None:
        """
        Plot confusion matrix for model evaluation
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot (optional)
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(max(12, len(class_names) * 1.2), max(10, len(class_names) * 1.0)))
        
        # Create heatmap with better styling
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='RdYlBu_r', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                   square=True, linewidths=0.5, linecolor='white')
        
        # Calculate and display accuracy
        if normalize:
            accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
            plt.title(f'{title} (Accuracy: {accuracy:.2%})', fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_model_performance(self, 
                             metrics: Dict[str, float],
                             save_path: Optional[str] = "results/model_performance.png") -> None:
        """
        Plot model performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # Plot overall metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[0].bar(metric_names, metric_values, alpha=0.8,
                          color=sns.color_palette("viridis", len(metric_names)))
        axes[0].set_title('Overall Performance Metrics', fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        axes[0].tick_params(axis='x', rotation=45)
        
        # Radar chart removed to avoid complex/large visualizations on constrained environments
        # Use the right subplot to show a concise message instead
        axes[1].axis('off')
        if len(metrics) >= 3:
            axes[1].text(0.5, 0.5, 'Radar chart disabled',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=12, style='italic')
        else:
            axes[1].text(0.5, 0.5, 'At least 3 metrics required',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=12, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def plot_class_performance(self, 
                             class_metrics: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = "results/class_performance.png") -> None:
        """
        Plot per-class performance metrics
        
        Args:
            class_metrics: Dictionary with class names as keys and metrics as values
            save_path: Path to save the plot (optional)
        """
        if not class_metrics:
            logger.warning("No class metrics provided for visualization")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(class_metrics).T
        
        plt.figure(figsize=(15, 8))
        
        # Create grouped bar plot
        x = np.arange(len(df.index))
        width = 0.25
        
        metrics = df.columns
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, df[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Food Classes', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(metrics) - 1) / 2, df.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class performance plot saved to {save_path}")
            plt.close()  # Close figure to prevent display
        else:
            plt.show()
    
    def create_dashboard(self, 
                        data_loader,
                        training_history: Optional[Dict] = None,
                        evaluation_results: Optional[Dict] = None,
                        save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all visualizations
        
        Args:
            data_loader: FoodDataLoader instance
            training_history: Training history dictionary (optional)
            evaluation_results: Evaluation results dictionary (optional)
            save_dir: Directory to save plots (optional)
        """
        logger.info("Creating comprehensive visualization dashboard...")
        
        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
        
        # 1. Dataset statistics
        stats = data_loader.get_dataset_stats()
        self.plot_dataset_statistics(
            stats, 
            save_path / "dataset_statistics.png" if save_dir else None
        )
        
        # 2. Class distribution
        self.plot_class_distribution(
            stats['class_distribution'],
            save_path=save_path / "class_distribution.png" if save_dir else None
        )
        
        # 3. Sample images
        sample_images, sample_labels = data_loader.get_sample_images(num_samples=16)
        self.plot_sample_images(
            sample_images, 
            sample_labels,
            save_path=save_path / "sample_images.png" if save_dir else None
        )
        
        # 4. Training history (if available)
        if training_history:
            self.plot_training_history(
                training_history,
                save_path=save_path / "training_history.png" if save_dir else None
            )
        
        # 5. Evaluation results (if available)
        if evaluation_results:
            if 'confusion_matrix' in evaluation_results:
                self.plot_confusion_matrix(
                    evaluation_results['confusion_matrix'],
                    data_loader.class_names,
                    save_path=save_path / "confusion_matrix.png" if save_dir else None
                )
            
            if 'metrics' in evaluation_results:
                self.plot_model_performance(
                    evaluation_results['metrics'],
                    save_path=save_path / "model_performance.png" if save_dir else None
                )
            
            if 'class_metrics' in evaluation_results:
                self.plot_class_performance(
                    evaluation_results['class_metrics'],
                    save_path=save_path / "class_performance.png" if save_dir else None
                )
        
        logger.info("Dashboard creation completed!")


def create_visualizer(**kwargs) -> DataVisualizer:
    """
    Convenience function to create a data visualizer
    
    Args:
        **kwargs: Arguments for DataVisualizer
        
    Returns:
        Configured DataVisualizer instance
    """
    return DataVisualizer(**kwargs)


if __name__ == "__main__":
    # Example usage
    from data_loader import FoodDataLoader
    
    # Create data loader and visualizer
    dataset_path = "../../Food Classification dataset"
    data_loader = FoodDataLoader(dataset_path, batch_size=16)
    visualizer = DataVisualizer()
    
    # Create comprehensive dashboard
    visualizer.create_dashboard(data_loader, save_dir="visualizations")
    
    print("Visualization dashboard created successfully!")