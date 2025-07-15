import os
import shutil
import random
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from PIL import Image
import imagehash
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import cv2

class EuroSATProcessor:
    def __init__(self, rgb_source_path, multispectral_source_path, output_path, 
                 target_size=(224, 224), min_size=(64, 64), random_seed=42):
        self.rgb_source_path = Path(rgb_source_path)
        self.multispectral_source_path = Path(multispectral_source_path)
        self.output_path = Path(output_path)
        self.target_height, self.target_width = target_size
        self.min_width, self.min_height = min_size
        self.random_seed = random_seed
        
        self.expected_classes = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        
        # Setup logging and stats
        self.setup_logging()
        self.initialize_stats()
        
    def setup_logging(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('EuroSATProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        log_file = self.output_path / 'processing.log'
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("EuroSAT Dataset Processor")
        self.logger.info("=" * 50)
        self.logger.info(f"RGB Source: {self.rgb_source_path}")
        self.logger.info(f"Multispectral Source: {self.multispectral_source_path}")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info(f"Target size: {self.target_width}x{self.target_height}")
        self.logger.info(f"Random seed: {self.random_seed}")
    
    def initialize_stats(self):
        """Initialize statistics counters"""
        self.stats = {
            'rgb': {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'corrupted': 0,
                'class_distribution': defaultdict(int)
            },
            'multispectral': {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'corrupted': 0,
                'class_distribution': defaultdict(int)
            }
        }
    
    def create_directory_structure(self):
        self.logger.info("\nCreating directory structure...")
        
        for data_type in ['rgb', 'multispectral']:
            for split in ['train', 'val', 'test']:
                split_dir = self.output_path / data_type / split
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for class_name in self.expected_classes:
                    class_dir = split_dir / class_name
                    class_dir.mkdir(exist_ok=True)
            
            metadata_dir = self.output_path / data_type / 'metadata'
            metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Directory structure created")
    
    def scan_source_dataset(self):
        self.logger.info("\nScanning source datasets...")
        
        scan_results = {
            'rgb': {'available': False, 'classes': {}},
            'multispectral': {'available': False, 'classes': {}}
        }
        
        # Check for RGB data
        if self.rgb_source_path.exists():
            for class_name in self.expected_classes:
                class_folder = self.rgb_source_path / class_name
                if class_folder.exists():
                    jpg_files = list(class_folder.glob("*.jpg"))
                    if jpg_files:
                        scan_results['rgb']['available'] = True
                        scan_results['rgb']['classes'][class_name] = len(jpg_files)
        
        # Check for multispectral data
        if self.multispectral_source_path.exists():
            for class_name in self.expected_classes:
                class_folder = self.multispectral_source_path / class_name
                if class_folder.exists():
                    tif_files = list(class_folder.glob("*.tif"))
                    if tif_files:
                        scan_results['multispectral']['available'] = True
                        scan_results['multispectral']['classes'][class_name] = len(tif_files)
        
        self.logger.info(f"RGB data available: {scan_results['rgb']['available']}")
        if scan_results['rgb']['available']:
            self.logger.info(f"RGB class distribution: {dict(scan_results['rgb']['classes'])}")
        
        self.logger.info(f"Multispectral data available: {scan_results['multispectral']['available']}")
        if scan_results['multispectral']['available']:
            self.logger.info(f"Multispectral class distribution: {dict(scan_results['multispectral']['classes'])}")
        
        return scan_results
    
    def validate_rgb_image(self, file_path):
        """Validate RGB image file"""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return {'valid': False, 'error': 'Failed to read with OpenCV'}
            
            h, w = img.shape[:2]
            size_valid = w >= self.min_width and h >= self.min_height
            
            # Check for blank images
            if np.all(img == img[0,0]):
                return {'valid': False, 'error': 'Blank/uniform image'}
            
            return {
                'valid': True,
                'size_valid': size_valid,
                'shape': (h, w)
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def validate_multispectral_image(self, file_path):
        """Validate multispectral TIFF file"""
        try:
            with rasterio.open(file_path) as src:
                if src.count != 13:
                    return {'valid': False, 'error': f'Expected 13 bands, got {src.count}'}
                
                h, w = src.height, src.width
                size_valid = w >= self.min_width and h >= self.min_height
                
                # Quick check for corrupted data
                sample_band = src.read(1)
                if np.all(sample_band == sample_band[0,0]):
                    return {'valid': False, 'error': 'Blank/uniform band detected'}
                
                return {
                    'valid': True,
                    'size_valid': size_valid,
                    'shape': (h, w),
                    'bands': src.count
                }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def resize_rgb_image(self, file_path, output_path):
        """Resize RGB image to target dimensions"""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return False
            
            resized = cv2.resize(img, (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(output_path), resized)
            return True
        except Exception as e:
            self.logger.error(f"Error resizing RGB image {file_path.name}: {e}")
            return False
    
    def resize_multispectral_image(self, file_path, output_path):
        """Resize multispectral image while preserving all bands"""
        try:
            with rasterio.open(file_path) as src:
                # Read and resize all bands
                data = src.read(
                    out_shape=(src.count, self.target_height, self.target_width),
                    resampling=Resampling.bilinear
                )
                
                # Update metadata
                profile = src.profile
                profile.update({
                    'height': self.target_height,
                    'width': self.target_width,
                    'transform': src.transform * src.transform.scale(
                        (src.width / self.target_width),
                        (src.height / self.target_height)
                    )
                })
                
                # Write resized image
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)
                
                return True
        except Exception as e:
            self.logger.error(f"Error resizing multispectral image {file_path.name}: {e}")
            return False
    
    def process_rgb_dataset(self):
        """Process RGB dataset (JPEG images)"""
        self.logger.info("\nProcessing RGB dataset...")
        
        # Collect all valid files
        files_per_class = defaultdict(list)
        for class_name in self.expected_classes:
            class_dir = self.rgb_source_path / class_name
            if not class_dir.exists():
                self.logger.debug(f"RGB class directory not found: {class_dir}")
                continue
                
            for file_path in class_dir.glob("*.jpg"):
                validation = self.validate_rgb_image(file_path)
                
                if not validation['valid']:
                    self.stats['rgb']['corrupted'] += 1
                    self.logger.debug(f"Invalid RGB file: {file_path.name} - {validation['error']}")
                    continue
                
                if not validation['size_valid']:
                    self.stats['rgb']['skipped'] += 1
                    self.logger.debug(f"Small RGB file skipped: {file_path.name}")
                    continue
                
                files_per_class[class_name].append(file_path)
                self.stats['rgb']['class_distribution'][class_name] += 1
                self.stats['rgb']['total'] += 1
        
        if not files_per_class:
            self.logger.warning("No valid RGB files found!")
            return False
        
        # Split files
        split_data = self.split_files(files_per_class)
        
        # Process and save files
        for class_name, splits in split_data.items():
            for split_name, files in splits.items():
                if split_name == 'counts':
                    continue
                
                for file_path in files:
                    output_path = self.output_path / 'rgb' / split_name / class_name / file_path.name
                    success = self.resize_rgb_image(file_path, output_path)
                    
                    if success:
                        self.stats['rgb']['processed'] += 1
                    else:
                        self.stats['rgb']['skipped'] += 1
        
        self.logger.info(f"RGB processing complete. Processed: {self.stats['rgb']['processed']}/{self.stats['rgb']['total']}")
        return True
    
    def process_multispectral_dataset(self):
        """Process multispectral dataset (TIFF images)"""
        self.logger.info("\nProcessing multispectral dataset...")
        
        # Collect all valid files
        files_per_class = defaultdict(list)
        
        for class_name in self.expected_classes:
            class_dir = self.multispectral_source_path / class_name
            if not class_dir.exists():
                self.logger.debug(f"Multispectral class directory not found: {class_dir}")
                continue
                
            for file_path in class_dir.glob("*.tif"):
                validation = self.validate_multispectral_image(file_path)
                
                if not validation['valid']:
                    self.stats['multispectral']['corrupted'] += 1
                    self.logger.debug(f"Invalid multispectral file: {file_path.name} - {validation['error']}")
                    continue
                
                if not validation['size_valid']:
                    self.stats['multispectral']['skipped'] += 1
                    self.logger.debug(f"Small multispectral file skipped: {file_path.name}")
                    continue
                
                files_per_class[class_name].append(file_path)
                self.stats['multispectral']['class_distribution'][class_name] += 1
                self.stats['multispectral']['total'] += 1
        
        if not files_per_class:
            self.logger.warning("No valid multispectral files found!")
            return False
        
        # Split files
        split_data = self.split_files(files_per_class)
        
        # Process and save files
        for class_name, splits in split_data.items():
            for split_name, files in splits.items():
                if split_name == 'counts':
                    continue
                
                for file_path in files:
                    output_path = self.output_path / 'multispectral' / split_name / class_name / file_path.name
                    success = self.resize_multispectral_image(file_path, output_path)
                    
                    if success:
                        self.stats['multispectral']['processed'] += 1
                    else:
                        self.stats['multispectral']['skipped'] += 1
        
        self.logger.info(f"Multispectral processing complete. Processed: {self.stats['multispectral']['processed']}/{self.stats['multispectral']['total']}")
        return True
    
    def split_files(self, files_per_class):
        """Split files into train/val/test sets (70/15/15)"""
        self.logger.info("\nSplitting files into train/val/test sets...")
        
        split_data = {}
        random.seed(self.random_seed)
        
        for class_name, files in files_per_class.items():
            random.shuffle(files)
            total_files = len(files)
            
            train_count = int(total_files * 0.7)
            val_count = int(total_files * 0.15)
            test_count = total_files - train_count - val_count
            
            split_data[class_name] = {
                'train': files[:train_count],
                'val': files[train_count:train_count + val_count],
                'test': files[train_count + val_count:],
                'counts': {
                    'train': train_count,
                    'val': val_count,
                    'test': test_count,
                    'total': total_files
                }
            }
            
            self.logger.info(
                f"{class_name}: Train={train_count}, Val={val_count}, Test={test_count}"
            )
        
        return split_data
    
    def save_metadata(self):
        """Save processing metadata and statistics"""
        self.logger.info("\nSaving metadata...")
        
        metadata = {
            'dataset_info': {
                'name': 'EuroSAT_Processed',
                'rgb_source': str(self.rgb_source_path),
                'multispectral_source': str(self.multispectral_source_path),
                'output': str(self.output_path),
                'created': datetime.now().isoformat(),
                'target_size': [self.target_height, self.target_width],
                'random_seed': self.random_seed
            },
            'rgb_stats': self.stats['rgb'],
            'multispectral_stats': self.stats['multispectral'],
            'split_ratios': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
        }
        
        # Save metadata for both data types
        for data_type in ['rgb', 'multispectral']:
            metadata_dir = self.output_path / data_type / 'metadata'
            metadata_file = metadata_dir / 'dataset_info.json'
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info("Metadata saved")
    
    def run(self):
        """Execute the full processing pipeline"""
        scan_results = self.scan_source_dataset()
        
        if not scan_results['rgb']['available'] and not scan_results['multispectral']['available']:
            self.logger.error("No valid data found in source directories!")
            return
        
        self.create_directory_structure()
        
        # Process RGB dataset if available
        if scan_results['rgb']['available']:
            if not self.process_rgb_dataset():
                self.logger.warning("RGB processing encountered issues")
        
        # Process multispectral dataset if available
        if scan_results['multispectral']['available']:
            if not self.process_multispectral_dataset():
                self.logger.warning("Multispectral processing encountered issues")
        
        self.save_metadata()
        
        # Final summary
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("=" * 50)
        self.logger.info(f"RGB: Total={self.stats['rgb']['total']}, Processed={self.stats['rgb']['processed']}")
        self.logger.info(f"Multispectral: Total={self.stats['multispectral']['total']}, Processed={self.stats['multispectral']['processed']}")
        self.logger.info(f"\nOutput directory: {self.output_path}")

if __name__ == "__main__":
    # Configuration - adjust these paths as needed
    RGB_SOURCE_PATH = r"Dataset\EuroSAT" 
    MULTISPECTRAL_SOURCE_PATH = r"Dataset\EuroSATallBands" 
    OUTPUT_PATH = "EuroSAT_Processed" 
    TARGET_SIZE = (224, 224)  
    MIN_SIZE = (64, 64)  
    RANDOM_SEED = 42 
    
    # Initialize and run processor
    processor = EuroSATProcessor(
        rgb_source_path=RGB_SOURCE_PATH,
        multispectral_source_path=MULTISPECTRAL_SOURCE_PATH,
        output_path=OUTPUT_PATH,
        target_size=TARGET_SIZE,
        min_size=MIN_SIZE,
        random_seed=RANDOM_SEED
    )
    
    processor.run()
