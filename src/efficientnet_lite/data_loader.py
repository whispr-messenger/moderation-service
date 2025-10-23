"""
Data Loading and Preprocessing Module for Food Classification

This module handles:
- Dataset loading from the Food Classification dataset
- Data preprocessing and augmentation
- Train/validation/test splitting
- Batch processing for training
"""

import json
import random
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import logging
from PIL import Image, ImageEnhance, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """Advanced data augmentation for food images"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 saturation_range: Tuple[float, float] = (0.8, 1.2),
                 blur_probability: float = 0.1,
                 flip_probability: float = 0.5):
        """
        Initialize data augmentation parameters
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            blur_probability: Probability of applying blur
            flip_probability: Probability of horizontal flip
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.blur_probability = blur_probability
        self.flip_probability = flip_probability
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentations to an image
        
        Args:
            image: PIL Image to augment
            
        Returns:
            Augmented PIL Image
        """
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        # Random horizontal flip
        if random.random() < self.flip_probability:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random brightness
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Random saturation
        if self.saturation_range != (1.0, 1.0):
            saturation_factor = random.uniform(*self.saturation_range)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_factor)
        
        # Random blur
        if random.random() < self.blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return image


class FoodDataLoader:
    """Comprehensive data loader for the Food Classification dataset"""
    
    def __init__(self, 
                 dataset_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 test_split: float = 0.1,
                 augmentation: bool = True,
                 seed: int = 42):
        """
        Initialize the data loader
        
        Args:
            dataset_path: Path to the Food Classification dataset
            image_size: Target image size (width, height)
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            augmentation: Whether to apply data augmentation
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.augmentation = augmentation
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize augmentation
        self.augmenter = DataAugmentation() if augmentation else None
        
        # Load dataset information
        self.class_names = []
        self.class_to_idx = {}
        self.file_paths = []
        self.labels = []
        
        self._load_dataset_info()
        self._split_dataset()
        
        logger.info(f"Loaded dataset with {len(self.class_names)} classes and {len(self.file_paths)} images")
        logger.info(f"Classes: {', '.join(self.class_names)}")
    
    def _load_dataset_info(self):
        """Load dataset information including class names and file paths"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Get all class directories
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        class_dirs.sort()  # Ensure consistent ordering
        
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load all image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in supported_extensions:
                    self.file_paths.append(str(img_file))
                    self.labels.append(class_idx)
        
        if len(self.file_paths) == 0:
            raise ValueError("No valid image files found in dataset")
    
    def _split_dataset(self):
        """Split dataset into train, validation, and test sets"""
        # Create indices and shuffle
        indices = list(range(len(self.file_paths)))
        random.shuffle(indices)
        
        # Calculate split points
        total_size = len(indices)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.validation_split)
        
        # Split indices
        self.test_indices = indices[:test_size]
        self.val_indices = indices[test_size:test_size + val_size]
        self.train_indices = indices[test_size + val_size:]
        
        logger.info(f"Dataset split - Train: {len(self.train_indices)}, "
                   f"Validation: {len(self.val_indices)}, Test: {len(self.test_indices)}")
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image_path: Path to the image file
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply augmentation if requested
            if augment and self.augmenter:
                image = self.augmenter.augment_image(image)
            
            # Resize image
            image = image.resize(self.image_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a blank image in case of error
            return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset"""
        distribution = {}
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            distribution[class_name] = count
        return distribution
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics"""
        distribution = self.get_class_distribution()
        
        stats = {
            'total_images': len(self.file_paths),
            'num_classes': len(self.class_names),
            'train_images': len(self.train_indices),
            'validation_images': len(self.val_indices),
            'test_images': len(self.test_indices),
            'class_distribution': distribution,
            'min_class_size': min(distribution.values()),
            'max_class_size': max(distribution.values()),
            'mean_class_size': np.mean(list(distribution.values())),
            'std_class_size': np.std(list(distribution.values()))
        }
        
        return stats
    
    def generate_batch(self, indices: List[int], augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of images and labels
        
        Args:
            indices: List of indices to include in batch
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            # Preprocess image
            image = self.preprocess_image(self.file_paths[idx], augment=augment)
            batch_images.append(image)
            
            # One-hot encode label
            label = np.zeros(len(self.class_names))
            label[self.labels[idx]] = 1.0
            batch_labels.append(label)
        
        return np.array(batch_images), np.array(batch_labels)
    
    def train_generator(self):
        """Generator for training data with augmentation"""
        while True:
            # Shuffle training indices each epoch
            train_indices = self.train_indices.copy()
            random.shuffle(train_indices)
            
            # Generate batches
            for i in range(0, len(train_indices), self.batch_size):
                batch_indices = train_indices[i:i + self.batch_size]
                yield self.generate_batch(batch_indices, augment=True)
    
    def validation_generator(self):
        """Generator for validation data without augmentation"""
        while True:
            # Generate batches
            for i in range(0, len(self.val_indices), self.batch_size):
                batch_indices = self.val_indices[i:i + self.batch_size]
                yield self.generate_batch(batch_indices, augment=False)
    
    def test_generator(self):
        """Generator for test data without augmentation"""
        for i in range(0, len(self.test_indices), self.batch_size):
            batch_indices = self.test_indices[i:i + self.batch_size]
            yield self.generate_batch(batch_indices, augment=False)
    
    def get_sample_images(self, num_samples: int = 9) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get sample images for visualization
        
        Args:
            num_samples: Number of sample images to return
            
        Returns:
            Tuple of (images, class_names) for the samples
        """
        # Select random samples from each class if possible
        samples_per_class = max(1, num_samples // len(self.class_names))
        sample_images = []
        sample_labels = []
        
        for class_name in self.class_names[:num_samples]:
            class_idx = self.class_to_idx[class_name]
            class_files = [i for i, label in enumerate(self.labels) if label == class_idx]
            
            if class_files:
                # Take random samples from this class
                selected = random.sample(class_files, min(samples_per_class, len(class_files)))
                
                for file_idx in selected:
                    if len(sample_images) < num_samples:
                        image = self.preprocess_image(self.file_paths[file_idx], augment=False)
                        sample_images.append(image)
                        sample_labels.append(class_name)
        
        return sample_images, sample_labels
    
    def save_class_mapping(self, output_path: str):
        """Save class names and mapping to file"""
        mapping = {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'num_classes': len(self.class_names)
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Class mapping saved to {output_path}")


def create_data_loader(dataset_path: str, **kwargs) -> FoodDataLoader:
    """
    Convenience function to create a data loader
    
    Args:
        dataset_path: Path to the dataset
        **kwargs: Additional arguments for FoodDataLoader
        
    Returns:
        Configured FoodDataLoader instance
    """
    return FoodDataLoader(dataset_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    dataset_path = "../../Food Classification dataset"
    
    # Create data loader
    data_loader = FoodDataLoader(dataset_path)
    
    # Print dataset statistics
    stats = data_loader.get_dataset_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        if key != 'class_distribution':
            print(f"  {key}: {value}")
    
    # Test data generation
    train_gen = data_loader.train_generator()
    batch_x, batch_y = next(train_gen)
    print(f"\nBatch shape: Images {batch_x.shape}, Labels {batch_y.shape}")
    
    # Save class mapping
    data_loader.save_class_mapping("class_mapping.json")