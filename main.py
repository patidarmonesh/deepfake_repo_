import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import traceback
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import Config
from dataset_manager import DatasetManager
from trainer import ProfessionalTrainer, ProfessionalISTVT, DeepfakeDataset
from evaluator import ProfessionalEvaluator

def setup_logging(config: Config):
    """Setup logging for the pipeline"""
    log_dir = Path(config.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

def run_full_pipeline(config: Config):
    """Execute the complete ISTVT deepfake detection pipeline"""
    try:
        print("🎯 Starting ISTVT Deepfake Detection Pipeline")
        print("=" * 60)
        
        # Ensure all directories exist
        config.ensure_dirs()
        setup_logging(config)
        
        # Step 1: Dataset Organization
        print("\n📁 Step 1: Dataset Organization")
        print("-" * 30)
        
        dataset_manager = DatasetManager(config)
        train_df, val_df, test_df = dataset_manager.create_dataset_splits()
        dataset_manager.save_splits(train_df, val_df, test_df)
        
        # Step 2: Feature Extraction & Dataset Creation
        print("\n🎬 Step 2: Creating Datasets")
        print("-" * 30)
        
        train_dataset = DeepfakeDataset(
            train_df['path'].tolist(),
            train_df['label'].tolist(),
            config,
            is_training=True
        )
        
        val_dataset = DeepfakeDataset(
            val_df['path'].tolist(),
            val_df['label'].tolist(),
            config,
            is_training=False
        )
        
        test_dataset = DeepfakeDataset(
            test_df['path'].tolist(),
            test_df['label'].tolist(),
            config,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        print(f"✅ Datasets created successfully!")
        print(f"   📊 Training samples: {len(train_dataset)}")
        print(f"   📊 Validation samples: {len(val_dataset)}")
        print(f"   📊 Test samples: {len(test_dataset)}")
        
        # Step 3: Model Training
        print("\n🧠 Step 3: Training Model")
        print("-" * 30)
        
        trainer = ProfessionalTrainer(config)
        trained_model = trainer.train(train_loader, val_loader)
        
        # Step 4: Model Evaluation
        print("\n📊 Step 4: Evaluating Model")
        print("-" * 30)
        
        # Load best model for evaluation
        best_checkpoint_path = Path(config.models_dir) / 'best_checkpoint.pth'
        if best_checkpoint_path.exists():
            print(f"📂 Loading best checkpoint from {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=config.get_device())
            trained_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best model loaded (Epoch {checkpoint.get('epoch', 'Unknown')})")
        else:
            print("⚠️  No checkpoint found, using current model state")
        
        # Run comprehensive evaluation
        evaluator = ProfessionalEvaluator(config)
        
        print("🔍 Running comprehensive evaluation...")
        results = evaluator.evaluate_model(trained_model, test_loader)
        
        print("📄 Generating evaluation report...")
        evaluator.generate_report(results)
        
        # Save final configuration
        config.save_config(Path(config.results_dir) / 'final_config.json')
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final Accuracy: {results['accuracy']:.4f}")
        print(f"📈 Final AUC Score: {results['auc_score']:.4f}")
        print(f"📁 Results saved to: {config.results_dir}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print(f"📋 Full traceback:")
        traceback.print_exc()
        logging.error(f"Pipeline failed: {str(e)}")
        return None

def run_training_only(config: Config):
    """Run only the training phase"""
    try:
        print("🧠 Training Only Mode")
        print("=" * 30)
        
        config.ensure_dirs()
        setup_logging(config)
        
        # Load existing splits or create new ones
        dataset_manager = DatasetManager(config)
        splits_exist = all([
            Path(config.splits_dir / f"{split}.json").exists() 
            for split in ['train', 'val', 'test']
        ])
        
        if splits_exist:
            print("📂 Loading existing dataset splits...")
            splits = dataset_manager.load_splits()
            train_df, val_df = splits['train'], splits['val']
        else:
            print("📁 Creating new dataset splits...")
            train_df, val_df, test_df = dataset_manager.create_dataset_splits()
            dataset_manager.save_splits(train_df, val_df, test_df)
        
        # Create datasets and loaders
        train_dataset = DeepfakeDataset(
            train_df['path'].tolist(),
            train_df['label'].tolist(),
            config,
            is_training=True
        )
        
        val_dataset = DeepfakeDataset(
            val_df['path'].tolist(),
            val_df['label'].tolist(),
            config,
            is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Train model
        trainer = ProfessionalTrainer(config)
        trained_model = trainer.train(train_loader, val_loader)
        
        print("✅ Training completed successfully!")
        return trained_model
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        traceback.print_exc()
        return None

def run_evaluation_only(config: Config):
    """Run only the evaluation phase"""
    try:
        print("📊 Evaluation Only Mode")
        print("=" * 30)
        
        config.ensure_dirs()
        setup_logging(config)
        
        # Load test split
        dataset_manager = DatasetManager(config)
        splits = dataset_manager.load_splits()
        
        if 'test' not in splits:
            print("❌ Test split not found. Run full pipeline first.")
            return None
        
        test_df = splits['test']
        
        # Create test dataset
        test_dataset = DeepfakeDataset(
            test_df['path'].tolist(),
            test_df['label'].tolist(),
            config,
            is_training=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Load trained model
        best_checkpoint_path = Path(config.models_dir) / 'best_checkpoint.pth'
        if not best_checkpoint_path.exists():
            print("❌ No trained model found. Run training first.")
            return None
        
        print(f"📂 Loading model from {best_checkpoint_path}")
        model = ProfessionalISTVT(config)
        checkpoint = torch.load(best_checkpoint_path, map_location=config.get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.get_device())
        
        # Run evaluation
        evaluator = ProfessionalEvaluator(config)
        results = evaluator.evaluate_model(model, test_loader)
        evaluator.generate_report(results)
        
        print("✅ Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main entry point with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description='ISTVT Deepfake Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                    # Run complete pipeline
  python main.py --mode train                   # Training only
  python main.py --mode eval                    # Evaluation only
  python main.py --mode full --dataset-path /path/to/data
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'train', 'eval'], 
        default='full',
        help='Pipeline mode to run (default: full)'
    )
    
    parser.add_argument(
        '--dataset-path', 
        type=str, 
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        help='Training batch size'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        help='Learning rate'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true', 
        help='Force GPU usage'
    )
    
    parser.add_argument(
        '--cpu', 
        action='store_true', 
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Override config with command line arguments
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    
    if args.output_dir:
        config.base_output_dir = args.output_dir
        # Update derived paths
        config.__post_init__()
    
    if args.batch_size:
        config.train_batch_size = args.batch_size
        config.val_batch_size = args.batch_size
        config.test_batch_size = args.batch_size
    
    if args.epochs:
        config.num_epochs = args.epochs
    
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    if args.gpu:
        config.use_cuda = True
    
    if args.cpu:
        config.use_cuda = False
    
    # Print configuration summary
    print("🔧 Configuration Summary:")
    print(f"   📁 Dataset Path: {config.dataset_path}")
    print(f"   📁 Output Directory: {config.base_output_dir}")
    print(f"   🎯 Mode: {args.mode}")
    print(f"   💾 Device: {config.get_device()}")
    print(f"   📦 Batch Size: {config.train_batch_size}")
    print(f"   🔄 Epochs: {config.num_epochs}")
    print(f"   📈 Learning Rate: {config.learning_rate}")
    
    # Run selected mode
    if args.mode == 'full':
        results = run_full_pipeline(config)
    elif args.mode == 'train':
        results = run_training_only(config)
    elif args.mode == 'eval':
        results = run_evaluation_only(config)
    
    if results is not None:
        print("\n🎉 Execution completed successfully!")
    else:
        print("\n💥 Execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
