import os
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"
    TRAINED_MODEL_DIR = MODEL_DIR / "trained"
    EXPORT_DIR = MODEL_DIR / "exported"
    LOGS_DIR = BASE_DIR / "logs"

    # Hardware configuration
    NUM_WORKERS = min(4, (os.cpu_count() or 1))
    TORCH_THREADS = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_MEMORY_USAGE = 0.75
    
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training hyperparameters
    SEED = 42
    BATCH_SIZE = 16
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    IMAGE_SIZE = (224, 224)
    EPOCHS = 50
    
    # Learning rate and optimization
    INITIAL_LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5
    LR_PATIENCE = 3
    LR_FACTOR = 0.5
    
    # Regularization
    DROPOUT = 0.2
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 7
    
    # Data augmentation (disabled for stability)
    MIXUP_PROB = 0.0
    CUTMIX_PROB = 0.0
    AUGMENTATION_PROB = 0.8
    
    # Model configuration
    MODEL_NAME = "mobilenet_v3_large"
    NUM_CLASSES = 10
    USE_AMP = True
    CHECKPOINT_INTERVAL = 1
    
    # Data split ratios
    SPLIT_RATIO = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    # Class names
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    @classmethod
    def create_dirs(cls) -> None:
        """Create all necessary directories"""
        dirs = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TRAINED_MODEL_DIR,
            cls.LOGS_DIR,
            cls.EXPORT_DIR
        ]
        
        for dir_path in dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create directory {dir_path}: {str(e)}")
    
    @classmethod
    def get_trained_model_path(cls) -> str:
        """Get the path to the trained model"""
        model_path = cls.TRAINED_MODEL_DIR / "best_model.pth"
        return str(model_path)
    
    @classmethod
    def get_resource_limits(cls) -> Dict[str, Any]:
        """Get system resource limits"""
        return {
            "num_workers": cls.NUM_WORKERS,
            "torch_threads": cls.TORCH_THREADS,
            "max_memory": cls.MAX_MEMORY_USAGE,
            "gradient_accumulation": cls.GRADIENT_ACCUMULATION_STEPS,
            "use_amp": cls.USE_AMP,
            "dropout": cls.DROPOUT
        }

# Initialize directories
try:
    Config.create_dirs()
except Exception as e:
    raise RuntimeError(f"Initialization failed: {str(e)}")