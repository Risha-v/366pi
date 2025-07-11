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

# Additional imports for robust image validation
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

class EuroSATDatasetOrganizer:
    """
    Organize EuroSAT dataset into structured format for machine learning with:
    - Size filtering
    - Validation
    - Comprehensive logging
    """
    
    def __init__(self, source_path, output_path, min_size=(64, 64)):
        """
        Initialize the organizer with logging and configuration
        
        Args:
            source_path: Path to the raw EuroSAT dataset
            output_path: Path where organized dataset will be saved
            min_size: Minimum (width, height) required for images
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.min_width, self.min_height = min_size
        self.expected_classes = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        
        # Initialize statistics
        self.stats = defaultdict(dict)
        self.corrupted_files = []
        self.small_files = []
        self.processed_files = 0
        self.random_seed = 42
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging system"""
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('EuroSATOrganizer')
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler which logs even debug messages
        log_file = self.output_path / 'dataset_organization.log'
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level (only INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Add a filter to the console handler to suppress WARNING messages
        ch.addFilter(lambda record: record.levelno < logging.WARNING)
        
        # Log initialization
        self.logger.info("EuroSAT Dataset Organizer")
        self.logger.info("=" * 50)
        self.logger.info(f"Source directory: {self.source_path}")
        self.logger.info(f"Output directory: {self.output_path}")
        self.logger.info(f"Minimum image size: {self.min_width}x{self.min_height}")
        self.logger.debug("Logging system initialized")
    
    def check_libraries(self):
        """Check if required libraries are available"""
        self.logger.debug("Checking required libraries...")
        
        if not (RASTERIO_AVAILABLE or TIFFFILE_AVAILABLE):
            self.logger.warning("Neither rasterio nor tifffile is available!")
            self.logger.warning("Multispectral TIFF validation will be limited")
            self.logger.info("Install one of them for better TIFF handling:")
            self.logger.info("  pip install rasterio")
            self.logger.info("  OR")
            self.logger.info("  pip install tifffile")
            return False
        return True
    
    def validate_tiff_file(self, file_path):
        """
        Validate TIFF file integrity and check bands and size
        Returns dict with validation info and size check
        """
        try:
            if RASTERIO_AVAILABLE:
                with rasterio.open(file_path) as src:
                    bands = src.count
                    height, width = src.height, src.width
                    size_valid = width >= self.min_width and height >= self.min_height
                    sample = src.read(1, window=((0, 10), (0, 10)))
                    
                    self.logger.debug(f"Validated TIFF: {file_path.name} - "
                                    f"Size: {width}x{height}, Bands: {bands}")
                    return {
                        'valid': True,
                        'size_valid': size_valid,
                        'bands': bands,
                        'shape': (height, width),
                        'dtype': src.dtypes[0]
                    }
            elif TIFFFILE_AVAILABLE:
                img = tifffile.imread(file_path)
                if len(img.shape) == 3:
                    height, width, bands = img.shape
                else:
                    height, width = img.shape
                    bands = 1
                
                size_valid = width >= self.min_width and height >= self.min_height
                
                self.logger.debug(f"Validated TIFF: {file_path.name} - "
                                f"Size: {width}x{height}, Bands: {bands}")
                return {
                    'valid': True,
                    'size_valid': size_valid,
                    'bands': bands,
                    'shape': (height, width),
                    'dtype': str(img.dtype)
                }
        except Exception as e:
            self.logger.debug(f"TIFF validation failed: {file_path.name} - {str(e)}")
            return {
                'valid': False,
                'size_valid': False,
                'error': str(e)
            }
    
    def validate_rgb_file(self, file_path):
        """
        Robust RGB image validation with multiple fallback methods
        Returns dict with validation info and size check
        """
        validation_attempts = []
        
        # Attempt 1: Try with OpenCV if available
        if OPENCV_AVAILABLE:
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if img is not None:
                    height, width = img.shape[:2]
                    size_valid = width >= self.min_width and height >= self.min_height
                    
                    # Check for blank images
                    if np.all(img == img[0,0]):
                        return {
                            'valid': False,
                            'size_valid': False,
                            'error': "Image appears uniform (OpenCV)"
                        }
                    
                    return {
                        'valid': True,
                        'size_valid': size_valid,
                        'shape': (height, width),
                        'mode': 'BGR',
                        'method': 'OpenCV'
                    }
            except Exception as e:
                validation_attempts.append(f"OpenCV failed: {str(e)}")
        
        # Attempt 2: Try with imageio if available
        if IMAGEIO_AVAILABLE:
            try:
                img = imageio.imread(file_path)
                if len(img.shape) == 3:  # Color image
                    height, width, channels = img.shape
                else:  # Grayscale
                    height, width = img.shape
                    channels = 1
                
                size_valid = width >= self.min_width and height >= self.min_height
                
                return {
                    'valid': True,
                    'size_valid': size_valid,
                    'shape': (height, width),
                    'mode': f'{channels} channel',
                    'method': 'imageio'
                }
            except Exception as e:
                validation_attempts.append(f"imageio failed: {str(e)}")
        
        # Attempt 3: Fallback to PIL/Pillow
        try:
            with Image.open(file_path) as img:
                # Force loading the image data
                try:
                    img.load()
                except Exception as e:
                    return {
                        'valid': False,
                        'size_valid': False,
                        'error': f"PIL loading failed: {str(e)}"
                    }
                
                width, height = img.size
                size_valid = width >= self.min_width and height >= self.min_height
                
                # Check color mode
                if img.mode not in ['RGB', 'L']:
                    return {
                        'valid': False, 
                        'size_valid': False,
                        'error': f"Invalid color mode: {img.mode}"
                    }
                
                # Verify image integrity
                try:
                    img.verify()
                except Exception as e:
                    return {
                        'valid': False,
                        'size_valid': False,
                        'error': f"Image corrupted: {str(e)}"
                    }
                
                # Check for empty/blank images
                if img.mode == 'RGB':
                    extrema = img.getextrema()
                    if all(e[0] == e[1] for e in extrema):
                        return {
                            'valid': False,
                            'size_valid': False,
                            'error': "Image appears empty"
                        }
                
                return {
                    'valid': True,
                    'size_valid': size_valid,
                    'shape': (width, height),
                    'mode': img.mode,
                    'method': 'Pillow'
                }
        except Exception as e:
            validation_attempts.append(f"Pillow failed: {str(e)}")
        
        # If all methods failed
        error_msg = "All validation methods failed. Attempts: " + "; ".join(validation_attempts)
        self.logger.debug(f"RGB validation failed: {file_path.name} - {error_msg}")
        return {
            'valid': False,
            'size_valid': False,
            'error': error_msg
        }
    
    def is_valid_image(self, file_path, data_type):
        """
        Perform thorough validation of image files including size check
        Returns dict with validation info and size check
        """
        if data_type == 'multispectral':
            return self.validate_tiff_file(file_path)
        else:  # RGB
            return self.validate_rgb_file(file_path)
    
    def scan_source_dataset(self):
        """Scan the source dataset and collect statistics"""
        self.logger.info("\nScanning source dataset...")
        
        scan_results = {
            'rgb': {'available': False, 'classes': {}},
            'multispectral': {'available': False, 'classes': {}}
        }
        
        # Scan RGB folders
        rgb_found = False
        for class_name in self.expected_classes:
            class_folder = self.source_path / class_name
            if class_folder.exists() and class_folder.is_dir():
                jpg_files = list(class_folder.glob("*.jpg"))
                if jpg_files:
                    scan_results['rgb']['available'] = True
                    scan_results['rgb']['classes'][class_name] = len(jpg_files)
                    self.logger.debug(f"Found {len(jpg_files)} RGB images in {class_name}")
                    rgb_found = True
        
        # Scan multispectral folders
        multispectral_path = self.source_path / "allBands"
        if multispectral_path.exists():
            for class_name in self.expected_classes:
                class_folder = multispectral_path / class_name
                if class_folder.exists() and class_folder.is_dir():
                    tif_files = list(class_folder.glob("*.tif"))
                    if tif_files:
                        scan_results['multispectral']['available'] = True
                        scan_results['multispectral']['classes'][class_name] = len(tif_files)
                        self.logger.debug(f"Found {len(tif_files)} multispectral images in {class_name}")
        
        self.logger.info("Scan completed")
        return scan_results
    
    def create_output_structure(self):
        """Create organized output directory structure"""
        self.logger.info("\nCreating output structure...")
        
        data_types = ['rgb', 'multispectral']
        splits = ['train', 'val', 'test']
        
        for data_type in data_types:
            for split in splits:
                split_dir = self.output_path / data_type / split
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for class_name in self.expected_classes:
                    class_dir = split_dir / class_name
                    class_dir.mkdir(exist_ok=True)
            
            metadata_dir = self.output_path / data_type / 'metadata'
            metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Directory structure created at: {self.output_path}")
    
    def clean_dataset(self, files_per_class, source_path, data_type):
        """Perform comprehensive cleaning of dataset"""
        self.logger.info(f"\nCleaning {data_type} dataset...")
        
        cleaned_files = defaultdict(list)
        validation_report = {}
        
        for class_name, file_list in files_per_class.items():
            self.logger.info(f"Processing class: {class_name}")
            valid_files = []
            
            for file_path in file_list:
                validation = self.is_valid_image(file_path, data_type)
                
                if not validation['valid']:
                    self.logger.debug(f"Invalid file: {file_path.name} - {validation['error']}")
                    self.corrupted_files.append(str(file_path))
                    continue
                
                if not validation['size_valid']:
                    self.logger.debug(f"Small file skipped: {file_path.name} - Size: {validation.get('shape', 'unknown')}")
                    self.small_files.append(str(file_path))
                    continue
                
                valid_files.append(file_path)
            
            cleaned_files[class_name] = valid_files
            validation_report[class_name] = {
                'original_count': len(file_list),
                'valid_count': len(valid_files),
                'invalid_count': len(file_list) - len(valid_files),
                'small_files_count': sum(1 for f in file_list 
                                      if self.is_valid_image(f, data_type).get('valid') 
                                      and not self.is_valid_image(f, data_type).get('size_valid'))
            }
            
            self.logger.info(f"  Valid files: {len(valid_files)}/{len(file_list)}")
        
        return cleaned_files, validation_report
    
    def split_dataset(self, files_per_class, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test sets"""
        self.logger.debug("\nSplitting dataset...")
        
        random.seed(self.random_seed)
        split_info = {}
        
        for class_name, file_list in files_per_class.items():
            random.shuffle(file_list)
            total_files = len(file_list)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)
            test_count = total_files - train_count - val_count
            
            split_info[class_name] = {
                'train': file_list[:train_count],
                'val': file_list[train_count:train_count + val_count],
                'test': file_list[train_count + val_count:],
                'counts': {
                    'train': train_count,
                    'val': val_count,
                    'test': test_count,
                    'total': total_files
                }
            }
            
            self.logger.debug(f"Class {class_name} split: "
                            f"Train={train_count}, Val={val_count}, Test={test_count}")
        
        return split_info
    
    def copy_files_to_structure(self, split_info, data_type):
        """Copy files to organized structure with proper paths"""
        self.logger.info(f"\nCopying {data_type} files to organized structure...")
        
        for class_name, splits in split_info.items():
            self.logger.info(f"Processing class: {class_name}")
            
            for split_name, file_list in splits.items():
                if split_name == 'counts':
                    continue
                
                self.logger.info(f"  {split_name}: {len(file_list)} files")
                
                for i, file_path in enumerate(file_list):
                    try:
                        # Determine source path based on data type
                        if data_type == 'multispectral':
                            src_file = self.source_path / "allBands" / class_name / file_path.name
                        else:  # RGB
                            src_file = self.source_path / class_name / file_path.name
                        
                        dst_file = self.output_path / data_type / split_name / class_name / file_path.name
                        
                        shutil.copy2(src_file, dst_file)
                        self.processed_files += 1
                        
                        if (i + 1) % 100 == 0:
                            self.logger.debug(f"    Copied {i + 1}/{len(file_list)} files")
                            
                    except Exception as e:
                        self.logger.error(f"Error copying {file_path.name}: {e}")
                        self.corrupted_files.append(str(src_file))
        
        self.logger.info(f"\nTotal {data_type} files processed: {self.processed_files}")
        if self.corrupted_files:
            self.logger.warning(f"Corrupted files encountered: {len(self.corrupted_files)}")
    
    def save_validation_report(self, validation_report, data_type):
        """Save validation report to metadata"""
        metadata_dir = self.output_path / data_type / 'metadata'
        
        if validation_report:
            report_file = metadata_dir / 'validation_report.json'
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            if self.small_files:
                small_files_log = metadata_dir / 'small_files.txt'
                with open(small_files_log, 'w') as f:
                    f.write("\n".join(self.small_files))
            
            self.logger.debug(f"Saved validation report for {data_type}")
    
    def generate_metadata(self, split_info, data_type):
        """Generate metadata and statistics"""
        self.logger.info(f"\nGenerating {data_type} metadata...")
        
        metadata = {
            'dataset_info': {
                'name': 'EuroSAT',
                'data_type': data_type,
                'classes': self.expected_classes,
                'min_width': self.min_width,
                'min_height': self.min_height,
                'created_at': datetime.now().isoformat(),
                'total_files_processed': self.processed_files,
                'corrupted_files_count': len(self.corrupted_files),
                'small_files_count': len(self.small_files)
            },
            'class_distribution': {},
            'split_ratios': {},
            'file_statistics': {}
        }
        
        total_files = 0
        for class_name, splits in split_info.items():
            if 'counts' in splits:
                counts = splits['counts']
                metadata['class_distribution'][class_name] = counts
                total_files += counts['total']
        
        total_train = sum(splits['counts']['train'] for splits in split_info.values() if 'counts' in splits)
        total_val = sum(splits['counts']['val'] for splits in split_info.values() if 'counts' in splits)
        total_test = sum(splits['counts']['test'] for splits in split_info.values() if 'counts' in splits)
        
        metadata['split_ratios'] = {
            'train': total_train / total_files if total_files > 0 else 0,
            'val': total_val / total_files if total_files > 0 else 0,
            'test': total_test / total_files if total_files > 0 else 0
        }
        
        metadata_file = self.output_path / data_type / 'metadata' / 'dataset_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Metadata saved for {data_type}")
        return metadata
    
    def process_data_type(self, data_type, file_extension):
        """Process a specific data type (RGB or multispectral)"""
        self.logger.info(f"\nProcessing {data_type} data...")
        
        # Reset counters for new data type
        self.processed_files = 0
        self.corrupted_files = []
        self.small_files = []
        
        # Determine source path
        if data_type == 'multispectral':
            source_path = self.source_path / "allBands"
        else:
            source_path = self.source_path
        
        # Collect files per class
        files_per_class = {}
        for class_name in self.expected_classes:
            class_dir = source_path / class_name
            if class_dir.exists():
                files = list(class_dir.glob(file_extension))
                if files:
                    files_per_class[class_name] = files
        
        if not files_per_class:
            self.logger.warning(f"No {data_type} files found for processing")
            return None
        
        # Clean dataset
        cleaned_files, validation_report = self.clean_dataset(
            files_per_class, source_path, data_type)
        
        # Split dataset
        split_info = self.split_dataset(cleaned_files)
        
        # Copy files to organized structure
        self.copy_files_to_structure(split_info, data_type)
        
        # Save validation report
        self.save_validation_report(validation_report, data_type)
        
        # Generate metadata
        metadata = self.generate_metadata(split_info, data_type)
        
        return metadata
    
    def print_summary(self, rgb_metadata, multispectral_metadata):
        """Print dataset summary"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATASET ORGANIZATION SUMMARY")
        self.logger.info("=" * 60)
        
        if rgb_metadata:
            self.logger.info("\nRGB Data Summary:")
            self.logger.info(f"Total Files: {rgb_metadata['dataset_info']['total_files_processed']}")
            self.logger.info(f"Train/Val/Test Split: {rgb_metadata['split_ratios']['train']:.1%}/"
                           f"{rgb_metadata['split_ratios']['val']:.1%}/"
                           f"{rgb_metadata['split_ratios']['test']:.1%}")
        
        if multispectral_metadata:
            self.logger.info("\nMultispectral Data Summary:")
            self.logger.info(f"Total Files: {multispectral_metadata['dataset_info']['total_files_processed']}")
            self.logger.info(f"Train/Val/Test Split: {multispectral_metadata['split_ratios']['train']:.1%}/"
                           f"{multispectral_metadata['split_ratios']['val']:.1%}/"
                           f"{multispectral_metadata['split_ratios']['test']:.1%}")
        
        self.logger.info("\nOutput Structure:")
        self.logger.info(f"  {self.output_path}")
        self.logger.info(f"  ├── rgb/")
        self.logger.info(f"  │   ├── train/")
        self.logger.info(f"  │   ├── val/")
        self.logger.info(f"  │   └── test/")
        self.logger.info(f"  └── multispectral/")
        self.logger.info(f"      ├── train/")
        self.logger.info(f"      ├── val/")
        self.logger.info(f"      └── test/")
        
        self.logger.info("\n" + "=" * 60)
    
    def organize_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Main function to organize the dataset"""
        self.logger.info("\nStarting EuroSAT dataset organization...")
        
        if not self.check_libraries():
            self.logger.warning("Proceeding with limited functionality")
        
        scan_results = self.scan_source_dataset()
        
        if not scan_results['rgb']['available'] and not scan_results['multispectral']['available']:
            self.logger.error("No valid data found in source directory!")
            return
        
        self.create_output_structure()
        
        rgb_metadata = None
        if scan_results['rgb']['available']:
            rgb_metadata = self.process_data_type(
                data_type='rgb',
                file_extension="*.jpg"
            )
        
        multispectral_metadata = None
        if scan_results['multispectral']['available']:
            multispectral_metadata = self.process_data_type(
                data_type='multispectral',
                file_extension="*.tif"
            )
        
        self.print_summary(rgb_metadata, multispectral_metadata)
        self.logger.info("\nDataset organization completed successfully!")
        self.logger.info(f"Organized dataset saved to: {self.output_path}")
        self.logger.info(f"Detailed logs available at: {self.output_path / 'dataset_organizations.log'}")

if __name__ == "__main__":
    # Configuration
    SOURCE_PATH = r"EuroSAT"
    OUTPUT_PATH = r"Dataset_EuroSAT_Split"
    MIN_SIZE = (64, 64)  # Minimum image size for CNN/ViT models
    
    # Initialize and run organizer
    organizer = EuroSATDatasetOrganizer(SOURCE_PATH, OUTPUT_PATH, min_size=MIN_SIZE)
    
    # Set random seed for reproducible splits
    random.seed(42)
    np.random.seed(42)
    
    # Organize dataset
    organizer.organize_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
