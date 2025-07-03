import os
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

@dataclass
class Config:
    """
    Central configuration class for ISTVT deepfake detection system.
    All hyperparameters and paths are configurable from here.
    """
    
    # ============ PATH CONFIGURATION ============
    # Dataset paths - adjust these for your data
    dataset_path: str = "/kaggle/input/celeb-df-v2"
    raw_video_dir: str = "/kaggle/input/deepfake-dataset/videos"
    
    # Output paths - where everything gets saved
    base_output_dir: str = "/kaggle/working/istvt_output"
    splits_dir: str = None
    features_dir: str = None
    models_dir: str = None
    results_dir: str = None
    metadata_dir: str = None
    
    # ============ DATASET CONFIGURATION ============
    # Dataset sampling and processing
    use_full_dataset: bool = False
    dataset_sample_percentage: float = 0.1
    use_stratified_sampling: bool = True
    
    # Dataset split ratios
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # CelebDF specific settings
    celeb_df_test_file: str = "List_of_testing_videos.txt"
    celeb_df_folders: List[str] = None
    
    # ============ VIDEO/IMAGE PROCESSING ============
    # Image dimensions
    image_height: int = 300
    image_width: int = 300
    
    # Video processing parameters
    sequence_length: int = 6
    max_sequences_per_video: int = 5
    max_frames_per_video: int = 270
    
    # Face detection parameters
    face_detection_min_size: int = 50
    face_scale_factor: float = 1.25
    
    # ============ MODEL ARCHITECTURE ============
    # Model dimensions
    embed_dim: int = 728
    num_transformer_blocks: int = 12
    num_attention_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    num_classes: int = 2
    
    # ============ TRAINING HYPERPARAMETERS ============
    # Batch sizes
    train_batch_size: int = 4
    val_batch_size: int = 4
    test_batch_size: int = 4
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.0005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # ============ DATA AUGMENTATION ============
    use_augmentation: bool = True
    augmentation_probability: float = 0.5
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip_prob: float = 0.5
    
    # ============ HARDWARE CONFIGURATION ============
    use_cuda: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    
    # ============ EXPERIMENT SETTINGS ============
    random_seed: int = 42
    save_checkpoints: bool = True
    save_visualizations: bool = True
    evaluation_frequency: int = 1
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation"""
        if self.celeb_df_folders is None:
            self.celeb_df_folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        
        # Set derived paths
        if self.splits_dir is None:
            self.splits_dir = os.path.join(self.base_output_dir, "splits")
        if self.features_dir is None:
            self.features_dir = os.path.join(self.base_output_dir, "features")
        if self.models_dir is None:
            self.models_dir = os.path.join(self.base_output_dir, "models")
        if self.results_dir is None:
            self.results_dir = os.path.join(self.base_output_dir, "results")
        if self.metadata_dir is None:
            self.metadata_dir = os.path.join(self.base_output_dir, "metadata")
        
        # Validate configuration
        assert self.train_split + self.val_split + self.test_split == 1.0, \
            "Dataset splits must sum to 1.0"
        assert 0 < self.dataset_sample_percentage <= 1.0, \
            "Dataset sample percentage must be between 0 and 1"
        assert self.embed_dim % self.num_attention_heads == 0, \
            "Embedding dimension must be divisible by number of attention heads"
    
    def ensure_dirs(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.base_output_dir,
            self.splits_dir,
            self.features_dir,
            self.models_dir,
            self.results_dir,
            self.metadata_dir,
            os.path.join(self.models_dir, "checkpoints"),
            os.path.join(self.results_dir, "visualizations")
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_device(self):
        """Returns the optimal device for processing"""
        if self.use_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def save_config(self, save_path: str):
        """Save configuration to JSON file"""
        import json
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
