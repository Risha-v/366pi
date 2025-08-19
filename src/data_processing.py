import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Tuple
import gc

from configs.default import Config

class EuroSATProcessor:
    def __init__(self, raw_data_path: str = None, processed_path: str = None):
        self.raw_data_path = raw_data_path or str(Config.RAW_DATA_DIR)
        self.processed_path = processed_path or str(Config.PROCESSED_DATA_DIR)
        self.classes = Config.CLASSES
        self.metadata = {
            'class_distribution': {},
            'class_counts': [],
            'image_stats': {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]},
            'split_ratio': Config.SPLIT_RATIO
        }

    def _validate_dataset(self) -> None:
        """Validate dataset structure"""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data directory not found at {self.raw_data_path}")
        
        missing_classes = []
        corrupted_images = 0
        
        for cls in self.classes:
            cls_path = os.path.join(self.raw_data_path, cls)
            if not os.path.exists(cls_path):
                missing_classes.append(cls)
                continue
            
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                raise ValueError(f"No valid images found in {cls_path}")
            
            # Check for corrupted images
            for img_name in images[:min(5, len(images))]:
                try:
                    with Image.open(os.path.join(cls_path, img_name)) as img:
                        img.verify()
                except Exception as e:
                    corrupted_images += 1
                    logging.warning(f"Corrupted image detected: {os.path.join(cls_path, img_name)}")
            
            self.metadata['class_distribution'][cls] = len(images)
            self.metadata['class_counts'].append(len(images))
        
        if missing_classes:
            logging.warning(f"Missing classes: {missing_classes}")
        if corrupted_images:
            logging.warning(f"Found {corrupted_images} potentially corrupted images")

    def _calculate_image_stats(self, sample_size: int = 100) -> None:
        """Calculate dataset mean and standard deviation"""
        try:
            pixel_sum = np.zeros(3, dtype=np.float64)
            pixel_sq_sum = np.zeros(3, dtype=np.float64)
            pixel_count = 0
            
            for cls in tqdm(self.classes, desc="Calculating image statistics"):
                cls_path = os.path.join(self.raw_data_path, cls)
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_name in images[:sample_size]:
                    try:
                        with Image.open(os.path.join(cls_path, img_name)) as img:
                            img_array = np.array(img) / 255.0
                            if img_array.ndim != 3 or img_array.shape[2] != 3:
                                continue
                            
                            pixel_sum += img_array.sum(axis=(0, 1))
                            pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
                            pixel_count += img_array.shape[0] * img_array.shape[9]
                    except Exception as e:
                        logging.warning(f"Skipping corrupted image {img_name}: {str(e)}")
                        continue
                
                # Periodic garbage collection
                if pixel_count % 100000 == 0:
                    gc.collect()
            
            mean = pixel_sum / pixel_count
            std = np.sqrt((pixel_sq_sum / pixel_count) - (mean ** 2))
            
            self.metadata['image_stats'] = {
                'mean': mean.tolist(),
                'std': std.tolist()
            }
            
        except Exception as e:
            logging.error(f"Failed to calculate image stats: {str(e)}")
            # Use ImageNet stats as fallback
            self.metadata['image_stats'] = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }

    def process_and_split(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """Process and split dataset"""
        try:
            self._validate_dataset()
            self._calculate_image_stats()
            
            # Create output directories
            for split in ['train', 'val', 'test']:
                split_dir = Path(self.processed_path) / split
                split_dir.mkdir(parents=True, exist_ok=True)
                for cls in self.classes:
                    (split_dir / cls).mkdir(exist_ok=True)
            
            # Process and split data
            for cls in tqdm(self.classes, desc="Processing classes"):
                cls_path = os.path.join(self.raw_data_path, cls)
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Stratified split
                train, temp = train_test_split(images, test_size=0.3, random_state=Config.SEED, shuffle=True)
                val, test = train_test_split(temp, test_size=0.5, random_state=Config.SEED, shuffle=True)
                
                # Process images
                for split_name, split_images in zip(['train', 'val', 'test'], [train, val, test]):
                    for img_name in tqdm(split_images, desc=f"{cls} - {split_name}", leave=False):
                        src_path = os.path.join(cls_path, img_name)
                        dest_path = Path(self.processed_path) / split_name / cls / img_name
                        
                        try:
                            with Image.open(src_path) as img:
                                img = img.convert('RGB').resize(target_size, Image.BILINEAR)
                                img.save(dest_path, quality=95)
                        except Exception as e:
                            logging.warning(f"Failed to process {src_path}: {str(e)}")
                            continue
                    
                    # Periodic garbage collection
                    if len(split_images) % 100 == 0:
                        gc.collect()
            
            # Save metadata
            metadata_path = Path(self.processed_path) / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logging.info(f"Dataset processing complete. Data saved to {self.processed_path}")
            
        except Exception as e:
            logging.error(f"Data processing failed: {str(e)}")
            raise
        finally:
            gc.collect()
