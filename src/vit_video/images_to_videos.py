# Script to convert Food Classification dataset images into videos for healthy/unhealthy classes
# Requires: pip install opencv-python

import cv2
import numpy as np
from pathlib import Path
import random

# Map food classes to healthy/unhealthy (only those present in the dataset)
HEALTHY_CLASSES = [
    'Baked Potato'
]
UNHEALTHY_CLASSES = [
    'Burger', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Pizza'
]

DATASET_ROOT = (Path(__file__).parent.parent.parent / 'Food Classification dataset' / 'Train').resolve()
OUT_DIR = Path('food_video_dataset')
HEALTHY_OUT = OUT_DIR / 'healthy_from_images'
UNHEALTHY_OUT = OUT_DIR / 'unhealthy_from_images'
HEALTHY_OUT.mkdir(parents=True, exist_ok=True)
UNHEALTHY_OUT.mkdir(parents=True, exist_ok=True)

VIDEO_DURATION = 3  # seconds
FPS = 15
FRAME_SIZE = (224, 224)
IMAGES_PER_VIDEO = VIDEO_DURATION * FPS
NUM_VIDEOS_PER_CLASS = 3

def get_images_for_class(class_dir):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(class_dir.glob(ext))
    return image_files

def make_video_from_images(image_paths, save_path, frame_size=FRAME_SIZE, fps=FPS):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, frame_size)
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(img, frame_size)
        out.write(frame)
    out.release()

def generate_videos_for_group(class_names, out_dir, label):
    for class_name in class_names:
        class_dir = DATASET_ROOT / class_name
        images = get_images_for_class(class_dir)
        print(f"Found {len(images)} images for {class_name} in {class_dir}")
        if len(images) == 0:
            print(f"No images found for {class_name}")
            continue
        for vid_idx in range(NUM_VIDEOS_PER_CLASS):
            # Randomly sample images for the video
            if len(images) < IMAGES_PER_VIDEO:
                selected = random.choices(images, k=IMAGES_PER_VIDEO)
            else:
                selected = random.sample(images, IMAGES_PER_VIDEO)
            video_name = f"{class_name.replace(' ', '_')}_{label}_{vid_idx+1}.mp4"
            save_path = out_dir / video_name
            make_video_from_images(selected, save_path)
            print(f"Generated video: {save_path}")

if __name__ == "__main__":
    print("Generating healthy videos from images...")
    generate_videos_for_group(HEALTHY_CLASSES, HEALTHY_OUT, 'healthy')
    print("Generating unhealthy videos from images...")
    generate_videos_for_group(UNHEALTHY_CLASSES, UNHEALTHY_OUT, 'unhealthy')