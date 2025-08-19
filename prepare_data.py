#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_processing import EuroSATProcessor
from configs.default import Config

def setup_logging():
    """Setup logging configuration"""
    log_dir = Config.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def check_raw_data():
    """Check if raw data directory exists"""
    raw_data_path = Config.RAW_DATA_DIR
    if not raw_data_path.exists():
        logging.error(f"Raw data directory not found: {raw_data_path}")
        logging.info("Please ensure EuroSAT dataset is downloaded and placed in the raw data directory")
        return False
    
    # Check for at least one class directory
    class_dirs = [d for d in raw_data_path.iterdir() if d.is_dir()]
    if not class_dirs:
        logging.error("No class directories found in raw data path")
        return False
    
    logging.info(f"Found {len(class_dirs)} directories in raw data path")
    return True

def main():
    """Main data preparation function"""
    print("EuroSAT Dataset Preparation")
    print("=" * 40)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting data preparation process")
    
    try:
        # Check raw data
        if not check_raw_data():
            sys.exit(1)
        
        # Initialize processor
        processor = EuroSATProcessor()
        
        # Process and split dataset
        logging.info("Processing and splitting dataset...")
        processor.process_and_split()
        
        # Success message
        logging.info("Data preparation completed successfully!")
        logging.info(f"Processed data saved to: {Config.PROCESSED_DATA_DIR}")
        logging.info(f"Log file saved to: {log_file}")
        
        print("\nData preparation completed successfully!")
        print(f"Processed data location: {Config.PROCESSED_DATA_DIR}")
        print(f"Log file: {log_file}")
        
    except KeyboardInterrupt:
        logging.info("Data preparation interrupted by user")
        print("\nData preparation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Data preparation failed: {str(e)}", exc_info=True)
        print(f"\nData preparation failed: {str(e)}")
        print(f"Check log file for details: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()
