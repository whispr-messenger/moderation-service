import os
import cv2
from pathlib import Path
from yt_dlp import YoutubeDL

# --- CONFIGURATION ---
DATASET_DIR = Path('food_data')
# Key = Folder name, Value = List of search keywords
FOOD_CATEGORIES = {
    'healthy': ['banana fruit', 'fresh apple', 'broccoli vegetable', 'green salad bowl'],
    'unhealthy': ['cheeseburger close up', 'pepperoni pizza slice', 'french fries', 'glazed donut']
}

# Settings
NUM_VIDEOS_PER_KEYWORD = 1  # How many videos to find per food item
MAX_FRAMES_PER_VIDEO = 20    # Max frames to extract to keep dataset balanced
FRAME_SIZE = (224, 224)

def setup_folders():
    """Creates the folder structure."""
    for cat in FOOD_CATEGORIES.keys():
        (DATASET_DIR / 'raw_videos' / cat).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'frames' / cat).mkdir(parents=True, exist_ok=True)
    print("✔ Folders initialized.")

def download_web_videos():
    """Searches and downloads videos using yt-dlp."""
    for category, keywords in FOOD_CATEGORIES.items():
        for query in keywords:
            save_path = DATASET_DIR / 'raw_videos' / category / f"%(title)s.%(ext)s"
            
            # yt-dlp options: search for query, limit results, download as mp4
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(save_path),
                'quiet': True,
                'noplaylist': True,
            }
            
            print(f"🔍 Searching the web for: '{query}'...")
            with YoutubeDL(ydl_opts) as ydl:
                # 'ytsearchN:query' tells it to download the first N results
                try:
                    ydl.download([f"ytsearch{NUM_VIDEOS_PER_KEYWORD}:{query}"])
                except Exception as e:
                    print(f"❌ Could not download {query}: {e}")

def extract_frames():
    """Processes all downloaded videos into 224x224 frames."""
    print("\n🎞  Processing videos into frames...")
    for category in FOOD_CATEGORIES.keys():
        video_folder = DATASET_DIR / 'raw_videos' / category
        output_folder = DATASET_DIR / 'frames' / category
        
        for video_path in video_folder.glob('*.mp4'):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames at even intervals
            if total_frames <= 0: continue
            step = max(1, total_frames // MAX_FRAMES_PER_VIDEO)
            
            count = 0
            saved_count = 0
            while cap.isOpened() and saved_count < MAX_FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret: break
                
                if count % step == 0:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    filename = f"{video_path.stem}_frame_{saved_count}.jpg"
                    cv2.imwrite(str(output_folder / filename), frame)
                    saved_count += 1
                count += 1
            cap.release()
            print(f"✔ Extracted {saved_count} frames from {video_path.name}")

if __name__ == "__main__":
    setup_folders()
    download_web_videos()
    extract_frames()
    print(f"\n🚀 DONE! Your dataset is ready in: {DATASET_DIR / 'frames'}")