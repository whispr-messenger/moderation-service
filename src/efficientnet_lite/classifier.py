"""
EfficientNet-Lite Food Classifier

Professional on-device food classification using EfficientNet-Lite architecture.
This is the main classifier module for the food classification system.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock TensorFlow for development without dependencies
class MockTensorFlow:
    """Mock TensorFlow implementation for development without dependencies"""
    
    class lite:
        class Interpreter:
            def __init__(self, model_path=None, **kwargs):
                self.model_path = model_path
                self._input_details = [{'index': 0, 'shape': [1, 224, 224, 3], 'dtype': np.float32}]
                self._output_details = [{'index': 0, 'shape': [1, 101], 'dtype': np.float32}]
                
            def allocate_tensors(self):
                pass
                
            def get_input_details(self):
                return self._input_details
                
            def get_output_details(self):
                return self._output_details
                
            def set_tensor(self, index, value):
                pass
                
            def invoke(self):
                pass
                
            def get_tensor(self, index):
                # Return mock prediction with some randomness
                np.random.seed(42)  # For consistent results
                logits = np.random.randn(1, 101) * 2
                probabilities = np.exp(logits) / np.sum(np.exp(logits))
                return probabilities

    @staticmethod
    def cast(tensor, dtype):
        return tensor.astype(dtype)
    
    @staticmethod
    def expand_dims(tensor, axis):
        return np.expand_dims(tensor, axis)

# Try to import TensorFlow, fall back to mock if not available
try:
    import tensorflow as tf
except ImportError:
    tf = MockTensorFlow()
    logger.warning("TensorFlow not found, using mock implementation for development")


class FoodClassifier:
    """
    Professional EfficientNet-Lite based food classifier for on-device inference
    
    Features:
    - Optimized for mobile/edge deployment
    - Supports 101 food categories
    - Real-time inference capability
    - Robust error handling
    """
    
    def __init__(self, model_path: str = None, labels_path: str = None):
        """
        Initialize the food classifier
        
        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to the class labels file
        """
        self.model_path = model_path or self._get_default_model_path()
        self.labels_path = labels_path or self._get_default_labels_path()
        self.interpreter = None
        self.class_names = None
        self.input_details = None
        self.output_details = None
        
        # Load model and labels
        self._load_model()
        self._load_labels()
    
    def _get_default_model_path(self) -> str:
        """Get default model path"""
        return os.path.join(os.path.dirname(__file__), "models", "food_classifier.tflite")
    
    def _get_default_labels_path(self) -> str:
        """Get default labels path"""
        return os.path.join(os.path.dirname(__file__), "models", "food_classes.txt")
    
    def _load_model(self):
        """Load the TFLite model"""
        try:
            if os.path.exists(self.model_path):
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using mock implementation")
                # Create a mock interpreter for development
                self.interpreter = tf.lite.Interpreter()
                self.input_details = [{'index': 0, 'shape': [1, 224, 224, 3], 'dtype': np.float32}]
                self.output_details = [{'index': 0, 'shape': [1, 101], 'dtype': np.float32}]
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_labels(self):
        """Load class labels"""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.class_names)} class labels")
            else:
                # Default food classes for development
                self.class_names = self._get_default_food_classes()
                logger.warning(f"Labels file not found, using {len(self.class_names)} default classes")
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            self.class_names = self._get_default_food_classes()
    
    def _get_default_food_classes(self) -> List[str]:
        """Default food classes (Food-101 dataset classes)"""
        return [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
            "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
            "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
            "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
            "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
            "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters",
            "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib",
            "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
            "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
            "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
            "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
            "waffles"
        ]
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for EfficientNet-Lite model input
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image array ready for inference
        """
        # Get input shape from model
        input_shape = self.input_details[0]['shape']
        target_size = (input_shape[2], input_shape[1])  # PIL expects (width, height)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image using high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict food class for an image
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples sorted by confidence
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                processed_image
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output probabilities
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]  # Remove batch dimension
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                confidence = float(predictions[idx])
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return [("unknown", 0.0)]
    
    def predict_from_path(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict food class from image file path
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
        try:
            with Image.open(image_path) as image:
                return self.predict(image, top_k)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return [("error", 0.0)]
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "input_shape": self.input_details[0]['shape'] if self.input_details else None,
            "output_shape": self.output_details[0]['shape'] if self.output_details else None,
            "num_classes": len(self.class_names),
            "class_names": self.class_names[:10] if self.class_names else [],  # First 10 for brevity
            "model_loaded": self.interpreter is not None,
            "architecture": "EfficientNet-Lite"
        }


# Convenience functions for easy usage
def classify_image(image_path: str, model_path: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Quick function to classify a single image
    
    Args:
        image_path: Path to image file
        model_path: Optional path to custom model file
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, confidence) tuples
    """
    classifier = FoodClassifier(model_path=model_path)
    return classifier.predict_from_path(image_path, top_k)


def batch_classify(image_paths: List[str], model_path: str = None, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """
    Classify multiple images efficiently
    
    Args:
        image_paths: List of image file paths
        model_path: Optional path to custom model file
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary mapping image paths to their predictions
    """
    classifier = FoodClassifier(model_path=model_path)
    results = {}
    
    for image_path in image_paths:
        results[image_path] = classifier.predict_from_path(image_path, top_k)
    
    return results