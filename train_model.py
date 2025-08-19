#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import gc
import psutil
import torch
from typing import Dict, Any
import argparse

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training import MobileNetV3Trainer
from src.visualization import EuroSATVisualizer
from configs.default import Config

def setup_logging() -> Path:
    log_file = Config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def check_system_resources() -> None:
    memory = psutil.virtual_memory()
    logging.info("System Resource Check:")
    logging.info(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    logging.info(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    logging.info(f"CPU Cores: {psutil.cpu_count()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        logging.info("GPU: Not available")

def check_data_availability() -> bool:
    processed_dir = Config.PROCESSED_DATA_DIR
    metadata_file = processed_dir / 'metadata.json'
    
    if not processed_dir.exists() or not metadata_file.exists():
        logging.error("Processed data not found. Run data preparation first.")
        return False
    
    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = processed_dir / dir_name
        if not dir_path.exists():
            logging.error(f"Missing {dir_name} directory in processed data")
            return False
    
    logging.info("Data availability check passed")
    return True

def get_training_config(args) -> Dict[str, Any]:
    return {
        'data_path': str(Config.PROCESSED_DATA_DIR),
        'batch_size': args.batch_size or Config.BATCH_SIZE,
        'lr': args.learning_rate or Config.INITIAL_LR,
        'weight_decay': args.weight_decay or Config.WEIGHT_DECAY,
        'epochs': args.epochs or Config.EPOCHS,
        'patience': args.patience or Config.EARLY_STOP_PATIENCE,
        'checkpoint_interval': Config.CHECKPOINT_INTERVAL,
        'save_dir': str(Config.TRAINED_MODEL_DIR),
        'seed': Config.SEED,
        'device': Config.DEVICE
    }

def save_training_metadata(config: Dict, best_f1: float, log_file: Path):
    training_metadata = {
        'training_date': datetime.now().isoformat(),
        'config': config,
        'best_f1_score': float(best_f1),
        'log_file': str(log_file),
        'system_info': {
            'cpu_cores': psutil.cpu_count(),
            'threads': Config.TORCH_THREADS,
            'memory_total': psutil.virtual_memory().total,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    }
    
    metadata_path = Config.TRAINED_MODEL_DIR / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    logging.info(f"Training metadata saved to {metadata_path}")
    return metadata_path

def main():
    parser = argparse.ArgumentParser(description='Train EuroSAT Land Cover Classification Model')
    parser.add_argument('--batch-size', type=int, help=f'Batch size (default: {Config.BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, help=f'Learning rate (default: {Config.INITIAL_LR})')
    parser.add_argument('--weight-decay', type=float, help=f'Weight decay (default: {Config.WEIGHT_DECAY})')
    parser.add_argument('--epochs', type=int, help=f'Number of epochs (default: {Config.EPOCHS})')
    parser.add_argument('--patience', type=int, help=f'Early stopping patience (default: {Config.EARLY_STOP_PATIENCE})')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    print("EuroSAT Model Training")
    print("=" * 30)
    
    try:
        log_file = setup_logging()
        logging.info("Starting EuroSAT model training")
        
        check_system_resources()
        
        if not check_data_availability():
            print("Data not available. Please run data preparation first.")
            sys.exit(1)
        
        config = get_training_config(args)
        
        if args.no_amp:
            Config.USE_AMP = False
            logging.info("Mixed precision training disabled")
        
        torch.set_num_threads(Config.TORCH_THREADS)
        torch.backends.cudnn.benchmark = True
        
        logging.info(f"Training configuration: {config}")
        
        print(f"Device: {config['device']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['lr']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Mixed precision: {'Enabled' if Config.USE_AMP else 'Disabled'}")
        
        logging.info("Initializing trainer...")
        trainer = MobileNetV3Trainer(config)
        
        visualizer = EuroSATVisualizer(Config.CLASSES)
        
        logging.info("Starting training process...")
        print("\nStarting training...")
        
        best_f1 = trainer.train()
        
        metadata_path = save_training_metadata(config, best_f1, log_file)
        
        print(f"\nTraining completed successfully!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Model saved to: {Config.TRAINED_MODEL_DIR / 'best_model.pth'}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Log file: {log_file}")
        
        logging.info(f"Training completed successfully. Best F1: {best_f1:.4f}")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        print("\nTraining interrupted by user")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)
        
    finally:
        logging.info("Cleaning up resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()