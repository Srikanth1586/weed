import os
import yaml
from pathlib import Path

def create_absolute_path_dataset_yaml():
    """Create dataset.yaml with absolute paths to fix path resolution issues"""
    
    base_dir = Path("weed-detector").absolute()
    processed_dir = base_dir / "data" / "processed"
    
    # Create absolute paths
    yaml_content = {
        'path': str(processed_dir),  # Absolute path to dataset root
        'train': 'images/train',     # Relative to 'path'
        'val': 'images/val',         # Relative to 'path'
        'test': 'images/test',       # Relative to 'path'
        
        'nc': 2,  # number of classes
        'names': ['weed', 'plant']  # class names
    }
    
    yaml_path = processed_dir / "dataset.yaml"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created dataset.yaml with absolute paths")
    print(f"📁 Path: {yaml_path}")
    print(f"📋 Content:")
    print(f"   path: {yaml_content['path']}")
    print(f"   train: {yaml_content['train']}")
    print(f"   val: {yaml_content['val']}")
    print(f"   test: {yaml_content['test']}")
    
    # Verify the paths exist
    print(f"\n🔍 Verifying paths...")
    train_path = processed_dir / "images" / "train"
    val_path = processed_dir / "images" / "val" 
    test_path = processed_dir / "images" / "test"
    
    print(f"   Train images: {train_path} - {'✅ EXISTS' if train_path.exists() else '❌ MISSING'}")
    print(f"   Val images: {val_path} - {'✅ EXISTS' if val_path.exists() else '❌ MISSING'}")
    print(f"   Test images: {test_path} - {'✅ EXISTS' if test_path.exists() else '❌ MISSING'}")
    
    # Count files
    if train_path.exists():
        train_files = len(list(train_path.glob("*")))
        print(f"   Train images count: {train_files}")
    if val_path.exists():
        val_files = len(list(val_path.glob("*")))
        print(f"   Val images count: {val_files}")
    if test_path.exists():
        test_files = len(list(test_path.glob("*")))
        print(f"   Test images count: {test_files}")

if __name__ == "__main__":
    create_absolute_path_dataset_yaml()