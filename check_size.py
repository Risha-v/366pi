import os
import numpy as np
from PIL import Image
import rasterio  

def check_eurosat_dataset_sizes(dataset_path):
    rgb_path = os.path.join(dataset_path, "RGB")
    if os.path.exists(rgb_path):
        print("Checking RGB dataset...")
        check_rgb_sizes(rgb_path)
    else:
        print(f"RGB directory not found at {rgb_path}")
    
    # Check Multispectral dataset
    ms_path = os.path.join(dataset_path, "MultiSpectral")
    if os.path.exists(ms_path):
        print("\nChecking Multispectral dataset...")
        check_multispectral_sizes(ms_path)
    else:
        print(f"Multispectral directory not found at {ms_path}")

def check_rgb_sizes(rgb_path):
    """Check sizes of RGB images"""
    sizes = set()
    channels = set()
    
    # Supported image extensions
    img_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    # Walk through all directories
    for root, dirs, files in os.walk(rgb_path):
        # Skip metadata directories
        if 'metadata' in root.lower():
            continue
            
        for file in files:
            # Check if file is an image
            if os.path.splitext(file)[1].lower() in img_extensions:
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        sizes.add(img.size)
                        channels.add(len(img.getbands()))
                        break  # Just need one image per directory to check size
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    print(f"RGB Image Sizes: {sizes}")
    print(f"RGB Channels: {channels}")

def check_multispectral_sizes(ms_path):
    """Check sizes and bands of multispectral images"""
    sizes = set()
    bands_counts = set()
    
    # Supported extensions for multispectral
    ms_extensions = {'.tif', '.tiff'}
    
    # Walk through all directories
    for root, dirs, files in os.walk(ms_path):
        # Skip metadata directories
        if 'metadata' in root.lower():
            continue
            
        for file in files:
            # Check if file is a multispectral image
            if os.path.splitext(file)[1].lower() in ms_extensions:
                img_path = os.path.join(root, file)
                try:
                    with rasterio.open(img_path) as src:
                        sizes.add((src.width, src.height))
                        bands_counts.add(src.count)
                        break  # Just need one image per directory to check size
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    print(f"Multispectral Image Sizes: {sizes}")
    print(f"Multispectral Bands Count: {bands_counts}")
  
if __name__ == "__main__":
    dataset_path = "EuroSAT_Processed"  
    check_eurosat_dataset_sizes(dataset_path)
