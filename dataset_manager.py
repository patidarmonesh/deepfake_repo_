import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
from config import Config

class DatasetManager:
    """
    Professional dataset management with configurable sampling and splitting.
    Handles CelebDF and other video datasets with consistent interface.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.dataset_path)
        
    def scan_celeb_df(self, split_type: str = 'train') -> pd.DataFrame:
        """
        Scan CelebDF dataset with professional error handling and logging.
        
        Args:
            split_type: Type of split to generate ('train', 'val', 'test', 'full')
            
        Returns:
            DataFrame with video paths and labels
        """
        entries = []
        
        print(f"📂 Scanning CelebDF dataset for {split_type} split...")
        
        if split_type == 'test':
            # Load test videos from provided text file
            test_file = self.base_path / self.config.celeb_df_test_file
            if test_file.exists():
                with open(test_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                label = int(parts[0])
                                rel_path = parts[1]
                                full_path = self.base_path / rel_path
                                if full_path.exists():
                                    entries.append({
                                        'path': str(full_path), 
                                        'label': label,
                                        'source': 'official_test'
                                    })
                        except (ValueError, IndexError) as e:
                            print(f"⚠️  Warning: Skipping line {line_num} in test file: {e}")
            else:
                print(f"❌ Test file not found: {test_file}")
        
        else:
            # Get test videos to exclude from training/validation
            test_videos = set()
            test_file = self.base_path / self.config.celeb_df_test_file
            if test_file.exists():
                with open(test_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            rel_path = parts[1]
                            full_path = self.base_path / rel_path
                            test_videos.add(str(full_path))
            
            # Scan all dataset folders
            for folder in self.config.celeb_df_folders:
                folder_path = self.base_path / folder
                if folder_path.exists():
                    print(f"  📁 Processing folder: {folder}")
                    video_files = list(folder_path.glob('*.mp4')) + \
                                 list(folder_path.glob('*.avi')) + \
                                 list(folder_path.glob('*.mov'))
                    
                    for video_file in video_files:
                        full_path = str(video_file)
                        
                        # Skip if video is in test set
                        if full_path in test_videos:
                            continue
                        
                        # Assign labels based on folder names
                        if folder in ['Celeb-real', 'YouTube-real']:
                            label = 1  # Real videos
                        else:  # Celeb-synthesis
                            label = 0  # Fake videos
                        
                        entries.append({
                            'path': full_path, 
                            'label': label,
                            'source': folder
                        })
                else:
                    print(f"⚠️  Warning: Folder not found: {folder_path}")
        
        df = pd.DataFrame(entries)
        
        if len(df) > 0:
            print(f"✅ Found {len(df)} videos")
            print(f"   📊 Real videos: {(df['label'] == 1).sum()}")
            print(f"   📊 Fake videos: {(df['label'] == 0).sum()}")
        else:
            print("❌ No videos found!")
        
        return df
    
    def apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stratified sampling to reduce dataset size while maintaining class balance"""
        if self.config.use_full_dataset:
            print("📊 Using full dataset (no sampling)")
            return df
        
        target_size = int(len(df) * self.config.dataset_sample_percentage)
        
        if self.config.use_stratified_sampling:
            print(f"📊 Applying stratified sampling: {target_size} samples ({self.config.dataset_sample_percentage*100:.1f}%)")
            
            splitter = StratifiedShuffleSplit(
                n_splits=1, 
                train_size=target_size,
                random_state=self.config.random_seed
            )
            
            sample_indices, _ = next(splitter.split(df, df['label']))
            sampled_df = df.iloc[sample_indices].reset_index(drop=True)
            
        else:
            print(f"📊 Applying random sampling: {target_size} samples")
            sampled_df = df.sample(
                n=target_size, 
                random_state=self.config.random_seed
            ).reset_index(drop=True)
        
        print(f"   📊 Sampled Real videos: {(sampled_df['label'] == 1).sum()}")
        print(f"   📊 Sampled Fake videos: {(sampled_df['label'] == 0).sum()}")
        
        return sampled_df
    
    def create_dataset_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and test splits with proper class balancing"""
        print("🔄 Creating dataset splits...")
        
        # First, get official test set if available
        test_df = self.scan_celeb_df('test')
        
        # Get training data (excluding test set)
        train_val_df = self.scan_celeb_df('train')
        
        # Apply sampling if configured
        train_val_df = self.apply_sampling(train_val_df)
        
        # Split train_val into train and validation
        val_size = self.config.val_split / (self.config.train_split + self.config.val_split)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df['label'],
            random_state=self.config.random_seed
        )
        
        print("✅ Dataset splits created:")
        print(f"   📊 Training: {len(train_df)} videos")
        print(f"   📊 Validation: {len(val_df)} videos")
        print(f"   📊 Test: {len(test_df)} videos")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save dataset splits to JSON files"""
        self.config.ensure_dirs()
        
        splits = {
            'train': train_df.to_dict('records'),
            'val': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        for split_name, split_data in splits.items():
            output_path = os.path.join(self.config.splits_dir, f"{split_name}.json")
            with open(output_path, 'w') as f:
                json.dump({'videos': split_data}, f, indent=2)
            print(f"💾 Saved {split_name} split to {output_path}")
    
    def load_splits(self) -> Dict[str, pd.DataFrame]:
        """Load existing split files"""
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(self.config.splits_dir, f"{split_name}.json")
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    data = json.load(f)['videos']
                    splits[split_name] = pd.DataFrame(data)
            else:
                print(f"⚠️ Warning: {split_path} not found")
        return splits
