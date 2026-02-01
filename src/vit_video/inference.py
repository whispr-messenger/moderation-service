"""
Inference script for Video Food Classifier
Usage: python inference.py --video path/to/video.mp4 --model best_food_classifier.pth
"""

import torch
import cv2
import numpy as np
from torchvision import transforms
from train import VideoViTClassifier
import argparse
from pathlib import Path


def predict_video(model, video_path, device, num_frames=8, img_size=224):
    """
    Predict whether a video shows healthy or unhealthy food
    """
    model.eval()
    
    # Load and preprocess video
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    if total_frames < num_frames:
        frame_indices = np.random.choice(total_frames, num_frames, replace=True)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    cap.release()
    
    # Transform frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transformed_frames = []
    for frame in frames:
        frame_tensor = transform(frame)
        transformed_frames.append(frame_tensor)
    
    video_tensor = torch.stack(transformed_frames).unsqueeze(0)  # (1, num_frames, 3, H, W)
    video_tensor = video_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    label_names = ['Healthy', 'Unhealthy']
    return label_names[predicted_class], confidence


def webcam_inference(model, device, num_frames=8, img_size=224):
    """
    Real-time inference from webcam
    Press 'q' to quit
    """
    model.eval()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    frame_buffer = []
    label_names = ['Healthy', 'Unhealthy']
    
    print("Starting webcam... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store frame in buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        frame_buffer.append(frame_resized)
        
        # Keep only last num_frames
        if len(frame_buffer) > num_frames:
            frame_buffer.pop(0)
        
        # Make prediction when we have enough frames
        if len(frame_buffer) == num_frames:
            # Transform frames
            transformed_frames = []
            for f in frame_buffer:
                frame_tensor = transform(f)
                transformed_frames.append(frame_tensor)
            
            video_tensor = torch.stack(transformed_frames).unsqueeze(0)
            video_tensor = video_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(video_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Display prediction on frame
            prediction_text = f"{label_names[predicted_class]}: {confidence*100:.1f}%"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            cv2.putText(frame, prediction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Collecting frames...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Food Classifier (Press Q to quit)', frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VideoViTClassifier(num_classes=2, pretrained=False, frames=args.num_frames)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train a model first using train.py")
        return
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.model}")
    
    if args.webcam:
        # Webcam mode
        webcam_inference(model, device, num_frames=args.num_frames, img_size=args.img_size)
    else:
        # Single video mode
        if not args.video:
            print("Error: Please provide --video path or use --webcam flag")
            return
        
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
        
        print(f"\nProcessing video: {video_path}")
        prediction, confidence = predict_video(model, video_path, device,
                                              num_frames=args.num_frames,
                                              img_size=args.img_size)
        
        print("\n" + "=" * 60)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Video Food Classifier')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='best_food_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time inference')
    parser.add_argument('--num-frames', type=int, default=8,
                       help='Number of frames to sample from video')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (height and width)')
    
    args = parser.parse_args()
    main(args)
