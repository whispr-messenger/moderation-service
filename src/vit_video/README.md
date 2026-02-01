# Video Classification for Healthy vs Unhealthy Food

A Vision Transformer (ViT) based video classifier to identify whether food in videos is healthy or unhealthy.

## Features

- **Vision Transformer Architecture**: Uses pretrained ViT-B/16 from ImageNet for feature extraction
- **Temporal Processing**: Samples multiple frames from videos and uses temporal average pooling
- **Multiple Data Sources**: Supports synthetic data generation, web downloads, or manual video collection
- **Real-time Inference**: Includes webcam support for live classification
- **Complete Pipeline**: Training, validation, evaluation, and inference

## Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quickest Start (Recommended for Beginners)

Run the interactive guide that walks you through everything:

```bash
python quick_start.py
```

This script will:
1. ✓ Check if all dependencies are installed
2. ✓ Help you get a video dataset (automated or manual)
3. ✓ Train the model with your chosen settings
4. ✓ Test the trained model

## Manual Quick Start

### 1. Get Dataset (Choose One)

#### 🚀 Option A: Automated Download (Recommended)

**From Pexels/Pixabay (Free Stock Videos)**
```bash
# 1. Run the download helper
python download_videos.py

# 2. Get free API keys (takes 2 minutes):
#    - Pexels: https://www.pexels.com/api/
#    - Pixabay: https://pixabay.com/accounts/register/

# 3. Edit video_download_config.json with your API keys

# 4. Download videos automatically
python download_videos.py --auto
```

**From YouTube (Creative Commons)**
```bash
# 1. Install yt-dlp
pip install yt-dlp

# 2. Run the helper to create config
python download_youtube.py

# 3. Find Creative Commons videos on YouTube and add URLs to youtube_urls.json

# 4. Download
python download_youtube.py
```

#### 📥 Option B: Manual Download
1. Visit free video sites:
   - [Pexels Videos](https://www.pexels.com/videos/)
   - [Pixabay Videos](https://pixabay.com/videos/)
   - [Coverr](https://coverr.co/)
   - [Videvo](https://www.videvo.net/)

2. Search for:
   - **Healthy**: "salad", "fresh fruit", "vegetables", "healthy meal"
   - **Unhealthy**: "burger", "pizza", "fast food", "junk food"

3. Download and place in:
   - `food_video_dataset/healthy/`
   - `food_video_dataset/unhealthy/`

4. Aim for **20-50 videos per category**

#### 🧪 Option C: Synthetic Videos (Quick Testing)
```bash
python generatedata.py
```
Creates sample videos with visual patterns (not realistic, for testing only)

#### 🖼️ Option D: Images to Videos
If you have food images:
```bash
# 1. Create folders and add images
mkdir food_video_dataset/healthy_images
mkdir food_video_dataset/unhealthy_images
# Add your .jpg/.png images to these folders

# 2. Convert to videos
python generatedata.py
```

### 2. Train the Model

**Option A: Using Jupyter Notebook (Recommended for exploration)**

```bash
jupyter notebook vit_video.ipynb
```

The notebook includes:
1. **Model Architecture** - ViT-based video classifier
2. **Dataset Generation** - Create or load videos
3. **Data Loading** - VideoDataset class with frame sampling
4. **Training Loop** - Train for 10 epochs with validation
5. **Evaluation** - Confusion matrix and metrics
6. **Inference** - Test on new videos
7. **Webcam Mode** - Real-time classification

**Option B: Using Command Line (Recommended for production)**

```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --epochs 20 --batch-size 4 --lr 5e-5 --num-frames 16
```

### 3. Run Inference

**Single Video Inference**
```bash
python inference.py --video path/to/video.mp4 --model best_food_classifier.pth
```

**Webcam Real-time Inference**
```bash
python inference.py --webcam --model best_food_classifier.pth
```

**Python API**
```python
from train import VideoViTClassifier
from inference import predict_video
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoViTClassifier(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('best_food_classifier.pth'))
model.to(device)

# Predict
prediction, confidence = predict_video(model, 'path/to/video.mp4', device)
print(f"Prediction: {prediction} ({confidence*100:.1f}% confidence)")
```

## Model Architecture

```
Input Video (8 frames, 224x224)
    ↓
ViT-B/16 Backbone (pretrained on ImageNet)
    ↓
Temporal Average Pooling
    ↓
Linear Classifier (2 classes)
    ↓
Output: [Healthy, Unhealthy]
```

## Dataset Structure

```
food_video_dataset/
├── healthy/
│   ├── apple_1.mp4
│   ├── salad_1.mp4
│   └── ...
└── unhealthy/
    ├── burger_1.mp4
    ├── pizza_1.mp4
    └── ...
```

### Dataset Requirements

**Minimum (for testing):**
- 10 videos per category (20 total)
- Split: 8 train, 2 validation per category

**Recommended (for good results):**
- 50-100 videos per category (100-200 total)
- Split: 80% train, 20% validation

**Optimal (for production):**
- 200+ videos per category (400+ total)
- Diverse food types, angles, lighting conditions
- Split: 70% train, 15% validation, 15% test

**Video Quality Guidelines:**
- Duration: 3-10 seconds
- Resolution: 480p or higher (will be resized to 224x224)
- Format: MP4 (most compatible)
- Clear visibility of food
- Varied angles and presentations
- Good lighting
- Minimal text/watermarks

## Configuration

Key parameters you can adjust:

- `num_frames`: Number of frames to sample per video (default: 8)
- `img_size`: Frame resolution (default: 224x224)
- `batch_size`: Batch size for training (default: 2)
- `NUM_EPOCHS`: Training epochs (default: 10)
- `learning_rate`: Initial learning rate (default: 1e-4)

## Performance Tips

1. **More Data**: Collect 50-100 videos per class for better results
2. **Data Augmentation**: Add random flips, rotations, color jitter
3. **Fine-tuning**: Train for more epochs or unfreeze ViT backbone
4. **Frame Sampling**: Experiment with different numbers of frames (4-16)
5. **Ensemble**: Combine predictions from multiple models

## Troubleshooting

**Issue**: Low accuracy with synthetic data
- **Solution**: Use real food videos for better results

**Issue**: Out of memory error
- **Solution**: Reduce batch size or number of frames

**Issue**: Slow training
- **Solution**: Use GPU if available, reduce image size

**Issue**: Model predicts same class for all videos
- **Solution**: Check dataset balance, increase training epochs

## Extending the Project

- **Multi-class Classification**: Extend to specific food categories (fruits, vegetables, desserts, etc.)
- **Nutritional Analysis**: Add calorie estimation or macro counting
- **Meal Planning**: Integrate with recipe databases
- **Mobile Deployment**: Convert to TorchScript or ONNX for mobile apps

## References

- Vision Transformer (ViT): [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- PyTorch Vision Transformer: [torchvision.models.vit](https://pytorch.org/vision/stable/models/vision_transformer.html)

## File Structure

```
vit_video/
├── quick_start.py              # Interactive guide (start here!)
├── train.py                    # Training script (CLI)
├── inference.py                # Inference script (single video or webcam)
├── generatedata.py             # Generate synthetic videos
├── download_videos.py          # Download from Pexels/Pixabay
├── download_youtube.py         # Download from YouTube
├── vit_video.ipynb            # Complete Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore patterns
└── food_video_dataset/         # Your video dataset (created automatically)
    ├── healthy/                # Healthy food videos
    └── unhealthy/              # Unhealthy food videos
```

## Common Commands

```bash
# Complete setup guide
python quick_start.py

# Generate synthetic dataset (for testing)
python generatedata.py

# Download real videos (automated)
python download_videos.py --auto

# Train model
python train.py --epochs 10 --batch-size 2

# Test on video
python inference.py --video path/to/video.mp4

# Test with webcam
python inference.py --webcam

# Interactive training (Jupyter)
jupyter notebook vit_video.ipynb
```

## License

MIT License
