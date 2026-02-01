"""
Training script for Video Food Classifier
Usage: python train.py --epochs 10 --batch-size 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import json


class VideoViTClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, frames=8):
        super().__init__()
        self.frames = frames
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit.heads = nn.Identity()
        self.classifier = nn.Linear(self.vit.hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.vit(x)
        feats = feats.view(b, t, -1)
        feats = feats.mean(dim=1)
        out = self.classifier(feats)
        return out


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=8, img_size=224, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            frame_indices = np.random.choice(total_frames, self.num_frames, replace=True)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video(video_path)
        transformed_frames = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames)
        return video_tensor, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc='Validation'):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Collect video paths
    dataset_dir = Path(args.dataset_dir)
    video_paths = []
    labels = []
    
    for video_path in (dataset_dir / 'healthy').glob('*.mp4'):
        video_paths.append(video_path)
        labels.append(0)
    
    for video_path in (dataset_dir / 'unhealthy').glob('*.mp4'):
        video_paths.append(video_path)
        labels.append(1)
    
    print(f"Total videos: {len(video_paths)}")
    print(f"Healthy: {labels.count(0)}, Unhealthy: {labels.count(1)}")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_paths, train_labels, 
                                 num_frames=args.num_frames, 
                                 img_size=args.img_size)
    val_dataset = VideoDataset(val_paths, val_labels,
                               num_frames=args.num_frames,
                               img_size=args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    model = VideoViTClassifier(num_classes=2, pretrained=True, frames=args.num_frames)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_model)
            print(f"✓ Model saved to {args.output_model} (Best val acc: {best_val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Training history saved to training_history.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Video Food Classifier')
    parser.add_argument('--dataset-dir', type=str, default='food_video_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-frames', type=int, default=8,
                       help='Number of frames to sample from each video')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (height and width)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--output-model', type=str, default='best_food_classifier.pth',
                       help='Path to save best model')
    
    args = parser.parse_args()
    main(args)
