#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.inference import EuroSATClassifier
from configs.default import Config

def setup_logging():
    """Setup logging configuration"""
    log_file = Config.LOGS_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def export_onnx(classifier, output_path):
    """Export model to ONNX format"""
    logging.info(f"Exporting model to ONNX format: {output_path}")
    
    try:
        if classifier.export_to_onnx(str(output_path)):
            logging.info(f"Successfully exported ONNX model to {output_path}")  # Removed emoji
            return True
        else:
            logging.error("Failed to export ONNX model")
            return False
    except Exception as e:
        logging.error(f"ONNX export failed: {str(e)}")
        return False

def export_torchscript(classifier, output_path):
    """Export model to TorchScript format"""
    logging.info(f"Exporting model to TorchScript format: {output_path}")
    
    try:
        if classifier.export_to_torchscript(str(output_path)):
            logging.info(f"Successfully exported TorchScript model to {output_path}")  # Removed emoji
            return True
        else:
            logging.error("Failed to export TorchScript model")
            return False
    except Exception as e:
        logging.error(f"TorchScript export failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export EuroSAT model to different formats')
    
    parser.add_argument(
        '--model',
        type=str,
        default=str(Config.TRAINED_MODEL_DIR / 'best_model.pth'),
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.EXPORT_DIR),
        help='Output directory for exported models'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'torchscript', 'all'],
        default=['all'],
        help='Export formats (default: all)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='eurosat_mobilenetv3',
        help='Base name for exported models'
    )
    
    args = parser.parse_args()
    
    print("EuroSAT Model Export")
    print("=" * 30)
    
    # Setup logging
    log_file = setup_logging()
    
    try:
        # Check if model exists
        model_path = Path(args.model)
        if not model_path.exists():
            logging.error(f"Model file not found: {model_path}")
            print(f"Model file not found: {model_path}")
            return
        
        # Load model
        logging.info("Loading model...")
        classifier = EuroSATClassifier(str(model_path))
        logging.info(f"Model loaded successfully from {model_path}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine formats to export
        formats = args.formats
        if 'all' in formats:
            formats = ['onnx', 'torchscript']
        
        export_results = {}
        
        # Export to requested formats
        for format_type in formats:
            if format_type == 'onnx':
                output_path = output_dir / f"{args.model_name}.onnx"
                export_results['onnx'] = export_onnx(classifier, output_path)
                
            elif format_type == 'torchscript':
                output_path = output_dir / f"{args.model_name}.pt"
                export_results['torchscript'] = export_torchscript(classifier, output_path)
        
        # Print summary
        print("\nExport Summary:")
        print("-" * 20)
        for format_type, success in export_results.items():
            status = "Success" if success else "Failed"  # Removed emoji
            print(f"{format_type.upper()}: {status}")
        
        total_success = sum(export_results.values())
        total_formats = len(export_results)
        
        print(f"\nOverall: {total_success}/{total_formats} formats exported successfully")
        print(f"Log file: {log_file}")
        
        if total_success == total_formats:
            logging.info("All export operations completed successfully")
        else:
            logging.warning(f"Some export operations failed ({total_success}/{total_formats} successful)")
        
    except Exception as e:
        logging.error(f"Export process failed: {str(e)}", exc_info=True)
        print(f"Export process failed: {str(e)}")
        print(f"Check log file for details: {log_file}")

if __name__ == "__main__":
    main()