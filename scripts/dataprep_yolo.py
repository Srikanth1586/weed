import json
import os
import shutil
from pathlib import Path
import yaml

def convert_coco_to_yolo_presplit():
    """
    Convert COCO format to YOLO format for pre-split dataset
    with custom category mapping for Plants/Weeds
    """
    print("🔄 Converting COCO to YOLO format for pre-split data...")
    
    base_dir = "weed-detector"
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # CUSTOM CLASS MAPPING FOR YOUR CATEGORIES
    class_mapping = {
        "Plants": 1,    # Plants = class 1
        "4": 1,         # "4" = also Plants (class 1)  
        "Weeds": 0      # Weeds = class 0
    }
    
    splits = ['train', 'valid', 'test']
    split_mapping = {'valid': 'val'}  # Map 'valid' to 'val' for YOLO
    
    stats = {}
    all_categories_found = set()
    
    for split in splits:
        print(f"\n📁 Processing {split} split...")
        
        # Input paths
        split_input_dir = os.path.join(raw_dir, split)
        coco_json_path = os.path.join(split_input_dir, "_annotations.coco.json")
        images_dir = os.path.join(split_input_dir, "images")
        
        # Output paths
        output_split = split_mapping.get(split, split)
        labels_output_dir = os.path.join(processed_dir, "labels", output_split)
        images_output_dir = os.path.join(processed_dir, "images", output_split)
        
        # Create output directories
        os.makedirs(labels_output_dir, exist_ok=True)
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Load COCO annotations
        if not os.path.exists(coco_json_path):
            print(f"❌ COCO JSON not found: {coco_json_path}")
            continue
            
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Print categories for verification
        categories = coco_data.get('categories', [])
        print(f"   Found categories: {[cat['name'] for cat in categories]}")
        
        # Create image id to file name mapping
        image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Create category id to class id mapping
        category_id_to_class = {}
        for category in categories:
            category_name = category['name']
            all_categories_found.add(category_name)
            
            # Map to YOLO class based on our custom mapping
            if category_name in class_mapping:
                category_id_to_class[category['id']] = class_mapping[category_name]
                print(f"   Mapping '{category_name}' (ID {category['id']}) → YOLO class {class_mapping[category_name]}")
            else:
                # Default: if not in mapping, treat as plant (class 1)
                category_id_to_class[category['id']] = 1
                print(f"   ⚠️  Unknown category '{category_name}', defaulting to class 1 (plant)")
        
        # Group annotations by image id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Process each image
        converted_count = 0
        for image_id, annotations in annotations_by_image.items():
            if image_id not in image_id_to_file:
                continue
                
            image_file = image_id_to_file[image_id]
            image_path = os.path.join(images_dir, image_file)
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue
            
            # Get image dimensions from COCO data
            image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Create YOLO format label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_output_dir, label_file)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # COCO bbox format: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Get class ID - use our custom mapping
                    class_id = category_id_to_class.get(ann['category_id'], 1)  # Default to plant
                    
                    # Write to label file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            # Copy image to processed directory
            shutil.copy2(image_path, os.path.join(images_output_dir, image_file))
            converted_count += 1
        
        stats[split] = converted_count
        print(f"✅ {split}: Converted {converted_count} images")
    
    print(f"\n📊 All categories found: {list(all_categories_found)}")
    return stats, class_mapping

def create_dataset_yaml_presplit():
    """
    Create dataset.yaml for pre-split data
    """
    print("\n📄 Creating dataset.yaml...")
    
    yaml_content = {
        'path': 'data/processed',  # dataset root dir
        'train': 'images/train',   # train images (relative to 'path')
        'val': 'images/val',       # val images (relative to 'path') 
        'test': 'images/test',     # test images (relative to 'path')
        
        'nc': 2,  # number of classes
        'names': ['weed', 'plant']  # class names (YOLO class 0 = weed, class 1 = plant)
    }
    
    yaml_path = "weed-detector/data/processed/dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created: {yaml_path}")
    print("   Class 0: 'weed'")
    print("   Class 1: 'plant'")
    return yaml_path

def verify_conversion():
    """
    Verify the conversion was successful
    """
    print("\n🔍 Verifying conversion...")
    
    processed_dir = "weed-detector/data/processed"
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = os.path.join(processed_dir, "labels", split)
        images_dir = os.path.join(processed_dir, "images", split)
        
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"📊 {split}:")
            print(f"   Labels: {len(label_files)} files")
            print(f"   Images: {len(image_files)} files")
            
            if len(label_files) == len(image_files):
                print(f"   ✅ Labels and images match")
            else:
                print(f"   ⚠️  Mismatch between labels and images")
                
            # Show class distribution in labels
            if label_files:
                class_counts = {0: 0, 1: 0}
                for label_file in label_files[:10]:  # Check first 10 files
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                
                print(f"   Class distribution (sample):")
                print(f"     - Weeds (class 0): {class_counts[0]}")
                print(f"     - Plants (class 1): {class_counts[1]}")

def main():
    """
    Main function for pre-split data conversion
    """
    print("🎯 Starting data preparation for pre-split dataset...")
    print("=" * 60)
    print("🔤 Category mapping:")
    print("   - 'Weeds' → YOLO class 0 (weed)")
    print("   - 'Plants' → YOLO class 1 (plant)") 
    print("   - '4' → YOLO class 1 (plant) - treated as plant")
    print("=" * 60)
    
    # Convert COCO to YOLO format
    stats, class_mapping = convert_coco_to_yolo_presplit()
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml_presplit()
    
    # Verify conversion
    verify_conversion()
    
    print("\n" + "=" * 60)
    print("🎉 Data preparation completed!")
    print(f"📊 Conversion stats:")
    for split, count in stats.items():
        print(f"   {split}: {count} images")
    print(f"🔤 Final class mapping: {class_mapping}")
    print(f"📁 YAML config: {yaml_path}")

if __name__ == "__main__":
    main()