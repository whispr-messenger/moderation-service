# Moderation Service

A service for content moderation and classification using various AI models.

## Features

- Image classification using EfficientNet-Lite
- Food image detection and classification
- More features coming soon!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/whispr-messenger/moderation-service.git
cd moderation-service
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Food Image Classification

You can use the food classification model to identify food items in images:

```python
from src.efficientnet_lite import classify_image

# Classify an image
results = classify_image(image_path="path/to/your/image.jpg")
print(results)
```

Or use the test script:

```bash
python test_food_classifier.py --image path/to/your/image.jpg
```

### Download Models

For first-time use, you may need to download the models:

```python
from src.efficientnet_lite.food_classifier import download_model

model_path = download_model()
print(f"Model downloaded to: {model_path}")
```

Or use the test script:

```bash
python test_food_classifier.py --download
```
