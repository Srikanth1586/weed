import os
import shutil

def setup_presplit_data_structure():
    """
    Setup project structure for pre-split dataset
    """
    base_dir = "weed-detector"
    
    # Define directories to create
    directories = [
        "data/raw/train/images",
        "data/raw/valid/images", 
        "data/raw/test/images",
        "data/processed/images/train",
        "data/processed/images/val",
        "data/processed/images/test",
        "data/processed/labels/train",
        "data/processed/labels/val", 
        "data/processed/labels/test",
        "models/training",
        "models/pretrained",
        "utils",
        "results"
    ]
    
    print("Setting up project structure for pre-split data...")
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")
    
    # Create all subdirectories
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created: {full_path}")
    
    # Create instructions file with proper encoding
    instructions = """INSTRUCTIONS FOR YOUR PRE-SPLIT DATA

Your dataset is already split into:
- Train: 4,780 images
- Valid: 359 images  
- Test: 317 images

NEXT STEPS:

1. COPY YOUR FILES:
   Copy these folders to weed-detector/data/raw/:
   - train/ (with images and _annotations.coco.json)
   - valid/ (with images and _annotations.coco.json) 
   - test/ (with images and _annotations.coco.json)

2. FINAL STRUCTURE SHOULD BE:
weed-detector/data/raw/
|-- train/
|   |-- images/ (4,780 images)
|   |-- _annotations.coco.json
|-- valid/
|   |-- images/ (359 images)
|   |-- _annotations.coco.json
|-- test/
|   |-- images/ (317 images)
|   |-- _annotations.coco.json

3. RUN DATA CONVERSION:
   python data_preparation_presplit_fixed.py
"""
    
    instructions_path = os.path.join(base_dir, "DATA_SETUP_INSTRUCTIONS.txt")
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"Created instructions: {instructions_path}")
    print("Project structure ready for your pre-split data!")
    return base_dir

if __name__ == "__main__":
    setup_presplit_data_structure()