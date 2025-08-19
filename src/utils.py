import os
import random
import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(class_counts):
    """Calculate class weights for balanced training"""
    weights = 1.0 / (np.array(class_counts) + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.4):
    """Apply CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    cy, cx = np.random.randint(h), np.random.randint(w)
    ch = int(h * np.sqrt(1 - lam))
    cw = int(w * np.sqrt(1 - lam))
    
    y1 = np.clip(cy - ch // 2, 0, h)
    y2 = np.clip(cy + ch // 2, 0, h)
    x1 = np.clip(cx - cw // 2, 0, w)
    x2 = np.clip(cx + cw // 2, 0, w)
    
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    return mixed_x, y_a, y_b, lam

def setup_logging(log_file: str = None, logger_name: str = "eurosat"):
    """Configure logging"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(logger_name)

def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.BILINEAR)
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")

def save_metadata(metadata: Dict, file_path: str):
    """Save metadata to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save metadata: {str(e)}")

def load_metadata(file_path: str) -> Dict:
    """Load metadata from JSON file"""
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load metadata: {str(e)}")

def get_class_colors() -> Dict[str, Tuple[int, int, int]]:
    """Get distinct colors for each class for visualization"""
    return {
        'AnnualCrop': (50, 205, 50),
        'Forest': (34, 139, 34),
        'HerbaceousVegetation': (152, 251, 152),
        'Highway': (105, 105, 105),
        'Industrial': (169, 169, 169),
        'Pasture': (238, 232, 170),
        'PermanentCrop': (255, 165, 0),
        'Residential': (220, 20, 60),
        'River': (30, 144, 255),
        'SeaLake': (64, 224, 208)
    }
