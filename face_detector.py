import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import warnings
from config import Config

# Try to import MTCNN for better face detection
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
    print("✓ MTCNN successfully imported for advanced face detection")
except ImportError:
    MTCNN_AVAILABLE = False
    print("⚠ MTCNN not available, using Haar Cascade for face detection")

warnings.filterwarnings('ignore')

class AdvancedFaceDetector:
    """
    Professional face detection with configurable parameters and fallback options.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        if MTCNN_AVAILABLE:
            print("🎯 Initializing MTCNN face detector...")
            self.detector = MTCNN(
                min_face_size=config.face_detection_min_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device='cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
            )
            self.detection_method = 'mtcnn'
        else:
            print("🎯 Initializing Haar Cascade face detector...")
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.detection_method = 'haar'
    
    def detect_and_align_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and align face with nose-tip centering as per ISTVT paper"""
        if self.detection_method == 'mtcnn':
            return self._detect_mtcnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_mtcnn(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """MTCNN-based face detection and alignment"""
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Detect faces and landmarks
            boxes, probs, landmarks = self.detector.detect(rgb_frame, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Get face with highest confidence
            best_idx = probs.argmax()
            box = boxes[best_idx]
            landmark = landmarks[best_idx]
            
            # Extract face coordinates
            x1, y1, x2, y2 = box.astype(int)
            w, h = x2 - x1, y2 - y1
            
            # Use nose tip as center (landmark[2] is nose tip)
            nose_x, nose_y = landmark[2].astype(int)
            
            # Calculate bounding box size
            size = int(self.config.face_scale_factor * max(w, h))
            
            return self._crop_and_resize(frame, nose_x, nose_y, size)
            
        except Exception as e:
            print(f"⚠️  MTCNN detection error: {e}")
            return None
    
    def _detect_haar(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Haar cascade-based face detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Estimate nose position as face center
            nose_x, nose_y = x + w // 2, y + h // 2
            size = int(self.config.face_scale_factor * max(w, h))
            
            return self._crop_and_resize(frame, nose_x, nose_y, size)
            
        except Exception as e:
            print(f"⚠️  Haar detection error: {e}")
            return None
    
    def _crop_and_resize(self, frame: np.ndarray, center_x: int, center_y: int, size: int) -> Optional[np.ndarray]:
        """Crop around center point and resize to target dimensions"""
        # Calculate crop boundaries
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(frame.shape[1], center_x + size // 2)
        y2 = min(frame.shape[0], center_y + size // 2)
        
        # Crop face region
        crop = frame[y1:y2, x1:x2]
        
        # Resize to target dimensions
        if crop.size > 0:
            return cv2.resize(crop, (self.config.image_width, self.config.image_height))
        
        return None

class XceptionBlock(nn.Module):
    """Xception block implementation with configurable parameters"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 start_with_relu: bool = True):
        super().__init__()
        
        self.start_with_relu = start_with_relu
        
        if start_with_relu:
            self.relu1 = nn.ReLU(inplace=True)
        
        # Depthwise separable convolution layers
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu2 = nn.ReLU(inplace=True)
        
        self.depthwise2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu3 = nn.ReLU(inplace=True)
        
        self.depthwise3 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels)
        self.pointwise3 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        if self.start_with_relu:
            x = self.relu1(x)
        
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.bn3(x)
        
        return x + residual

class XceptionFeatureExtractor(nn.Module):
    """Xception feature extractor with configurable dimensions"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Xception blocks
        self.block1 = XceptionBlock(64, 128, stride=2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, stride=2)
        self.block3 = XceptionBlock(256, config.embed_dim, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        return x

class VideoSequenceProcessor:
    """Professional video processing with configurable parameters"""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_detector = AdvancedFaceDetector(config)
    
    def extract_face_sequences(self, video_path: str) -> List[np.ndarray]:
        """Extract face sequences from video with configurable parameters"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        faces = []
        frame_count = 0
        
        # Process video frames
        while frame_count < self.config.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and align face
            face = self.face_detector.detect_and_align_face(frame)
            if face is not None:
                # Convert to RGB and normalize
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                faces.append(rgb_face)
            
            frame_count += 1
        
        cap.release()
        
        # Create sequences of consecutive frames
        sequences = []
        for i in range(len(faces) - self.config.sequence_length + 1):
            if len(sequences) >= self.config.max_sequences_per_video:
                break
            
            sequence = np.stack(faces[i:i + self.config.sequence_length])
            sequences.append(sequence)
        
        return sequences
