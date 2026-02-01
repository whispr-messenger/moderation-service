"""
Quick Start Script for Food Video Classifier
This script guides you through the entire process step by step
"""

from pathlib import Path
import subprocess
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_requirements():
    """Check if required packages are installed"""
    print_header("STEP 1: Checking Requirements")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages installed!")
        return True

def check_dataset():
    """Check if dataset exists"""
    print_header("STEP 2: Checking Dataset")
    
    dataset_dir = Path('food_video_dataset')
    healthy_dir = dataset_dir / 'healthy'
    unhealthy_dir = dataset_dir / 'unhealthy'
    
    healthy_videos = list(healthy_dir.glob('*.mp4')) if healthy_dir.exists() else []
    unhealthy_videos = list(unhealthy_dir.glob('*.mp4')) if unhealthy_dir.exists() else []
    
    print(f"Healthy videos: {len(healthy_videos)}")
    print(f"Unhealthy videos: {len(unhealthy_videos)}")
    
    if len(healthy_videos) == 0 or len(unhealthy_videos) == 0:
        print("\n⚠️  No dataset found!")
        return False
    elif len(healthy_videos) < 10 or len(unhealthy_videos) < 10:
        print(f"\n⚠️  Dataset is too small (minimum: 10 per category)")
        print(f"   Recommended: 50+ per category for good results")
        return False
    else:
        print(f"\n✓ Dataset ready!")
        if len(healthy_videos) < 50 or len(unhealthy_videos) < 50:
            print(f"   Note: For better accuracy, consider adding more videos")
            print(f"   Current: {len(healthy_videos)} + {len(unhealthy_videos)} = {len(healthy_videos) + len(unhealthy_videos)}")
            print(f"   Recommended: 50+ per category")
        return True

def get_dataset():
    """Guide user through dataset acquisition"""
    print("\nHow would you like to get the dataset?\n")
    print("1. Download from Pexels/Pixabay (automated, requires API keys)")
    print("2. Download from YouTube (requires yt-dlp)")
    print("3. Generate synthetic videos (quick test, not realistic)")
    print("4. Manual instructions (I'll download myself)")
    print("5. Skip (I already have videos)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        print("\n📥 Setting up automated download from Pexels/Pixabay...")
        print("\n1. Run: python download_videos.py")
        print("2. Get free API keys:")
        print("   - Pexels: https://www.pexels.com/api/")
        print("   - Pixabay: https://pixabay.com/accounts/register/")
        print("3. Edit video_download_config.json with your API keys")
        print("4. Run: python download_videos.py --auto")
        print("\nRun these commands, then come back to this script.")
        
    elif choice == '2':
        print("\n📺 Setting up YouTube download...")
        print("\n1. Install yt-dlp: pip install yt-dlp")
        print("2. Run: python download_youtube.py")
        print("3. Add video URLs to youtube_urls.json")
        print("4. Run: python download_youtube.py")
        print("\nRun these commands, then come back to this script.")
        
    elif choice == '3':
        print("\n🧪 Generating synthetic videos...")
        try:
            subprocess.run([sys.executable, 'generatedata.py'], check=True)
            print("\n✓ Synthetic videos generated!")
            print("⚠️  Note: These are not realistic. For actual training, use real videos.")
            return True
        except Exception as e:
            print(f"\n❌ Error generating videos: {e}")
            return False
            
    elif choice == '4':
        print("\n📖 Manual Download Instructions:")
        print("\n1. Visit these free video websites:")
        print("   - https://www.pexels.com/videos/")
        print("   - https://pixabay.com/videos/")
        print("   - https://coverr.co/")
        print("\n2. Search for:")
        print("   Healthy: 'salad', 'fresh fruit', 'vegetables'")
        print("   Unhealthy: 'burger', 'pizza', 'fast food'")
        print("\n3. Download and save to:")
        print("   - food_video_dataset/healthy/")
        print("   - food_video_dataset/unhealthy/")
        print("\n4. Aim for 20-50 videos per category")
        print("\nCome back when you have videos ready!")
        
    elif choice == '5':
        print("\n✓ Skipping dataset acquisition")
        return True
        
    else:
        print("\n❌ Invalid choice")
        return False
    
    return False

def train_model():
    """Guide through training"""
    print_header("STEP 3: Training the Model")
    
    print("How would you like to train?\n")
    print("1. Jupyter Notebook (interactive, with visualizations)")
    print("2. Command line (faster, automated)")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '1':
        print("\n📓 Starting Jupyter Notebook...")
        print("\nRun: jupyter notebook vit_video.ipynb")
        print("Then execute all cells in the notebook.")
        
    elif choice == '2':
        print("\n⚙️  Training with default settings...")
        print("This will take several minutes depending on your hardware.\n")
        
        epochs = input("Number of epochs (default 10, press Enter to use default): ").strip()
        epochs = int(epochs) if epochs else 10
        
        batch_size = input("Batch size (default 2, press Enter to use default): ").strip()
        batch_size = int(batch_size) if batch_size else 2
        
        print(f"\nStarting training with {epochs} epochs and batch size {batch_size}...")
        print("Command: python train.py --epochs {} --batch-size {}\n".format(epochs, batch_size))
        
        try:
            subprocess.run([
                sys.executable, 'train.py',
                '--epochs', str(epochs),
                '--batch-size', str(batch_size)
            ], check=True)
            print("\n✓ Training complete!")
            return True
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return False
    else:
        print("\n❌ Invalid choice")
        return False
    
    return False

def test_model():
    """Test the trained model"""
    print_header("STEP 4: Testing the Model")
    
    model_path = Path('best_food_classifier.pth')
    if not model_path.exists():
        print("❌ Model not found. Please train the model first.")
        return False
    
    print("✓ Trained model found: best_food_classifier.pth\n")
    print("How would you like to test?\n")
    print("1. Test on a video file")
    print("2. Real-time webcam test")
    print("3. Skip testing")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        video_path = input("\nEnter video path: ").strip().strip('"\'')
        if Path(video_path).exists():
            print(f"\n🎬 Testing on: {video_path}")
            try:
                subprocess.run([
                    sys.executable, 'inference.py',
                    '--video', video_path,
                    '--model', 'best_food_classifier.pth'
                ], check=True)
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print(f"\n❌ Video not found: {video_path}")
            
    elif choice == '2':
        print("\n📷 Starting webcam test...")
        print("Press 'Q' to quit the webcam window\n")
        try:
            subprocess.run([
                sys.executable, 'inference.py',
                '--webcam',
                '--model', 'best_food_classifier.pth'
            ], check=True)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
    elif choice == '3':
        print("\n✓ Skipping testing")
        return True
    else:
        print("\n❌ Invalid choice")
        return False
    
    return True

def main():
    """Main quick start flow"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          🍎 FOOD VIDEO CLASSIFIER - QUICK START GUIDE 🍔                  ║
║                                                                            ║
║                  Healthy vs Unhealthy Food Classification                 ║
║                        Using Vision Transformer (ViT)                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("This script will guide you through:")
    print("1. Installing dependencies")
    print("2. Getting/creating a dataset")
    print("3. Training the model")
    print("4. Testing the model\n")
    
    input("Press Enter to continue...")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Please install requirements first: pip install -r requirements.txt")
        return
    
    # Step 2: Check dataset
    has_dataset = check_dataset()
    
    if not has_dataset:
        print("\nLet's get a dataset!")
        dataset_ready = get_dataset()
        
        if not dataset_ready:
            print("\n⏸️  Dataset setup incomplete. Run this script again when ready!")
            return
    
    # Recheck dataset
    if not check_dataset():
        print("\n⏸️  Dataset still not ready. Please add videos and run this script again.")
        return
    
    # Step 3: Train
    print("\nDataset ready! Let's train the model.")
    input("Press Enter to continue...")
    
    trained = train_model()
    
    if not trained:
        print("\n⏸️  Training incomplete. You can train manually with:")
        print("   - Notebook: jupyter notebook vit_video.ipynb")
        print("   - CLI: python train.py")
        return
    
    # Step 4: Test
    print("\nModel trained! Let's test it.")
    input("Press Enter to continue...")
    
    test_model()
    
    # Done!
    print_header("CONGRATULATIONS! 🎉")
    print("You've successfully set up and trained a food video classifier!")
    print("\nNext steps:")
    print("- Improve accuracy: Add more diverse videos")
    print("- Fine-tune: Adjust hyperparameters in train.py")
    print("- Deploy: Export model for production use")
    print("- Extend: Add more food categories")
    print("\nFor more information, see README.md")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
