import os

def quick_label_check():
    """
    Quick verification of YOLO label files
    """
    print("🔍 Quick YOLO Label Check")
    print("=" * 40)
    
    base_dir = "weed-detector/data/processed"
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = os.path.join(base_dir, "labels", split)
        images_dir = os.path.join(base_dir, "images", split)
        
        print(f"\n📁 {split.upper()}:")
        
        # Check if directories exist
        if not os.path.exists(labels_dir):
            print("❌ Labels directory missing")
            continue
        if not os.path.exists(images_dir):
            print("❌ Images directory missing")
            continue
        
        # Count files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   Images: {len(image_files)}")
        print(f"   Labels: {len(label_files)}")
        
        # Check for matching files
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        
        missing_labels = image_basenames - label_basenames
        missing_images = label_basenames - image_basenames
        
        if missing_labels:
            print(f"   ⚠️  {len(missing_labels)} images missing labels")
        if missing_images:
            print(f"   ⚠️  {len(missing_images)} labels missing images")
        
        if not missing_labels and not missing_images:
            print("   ✅ All images have corresponding labels")
        
        # Check a few label files for format
        if label_files:
            print(f"\n   📝 Sample label format check:")
            sample_file = os.path.join(labels_dir, label_files[0])
            with open(sample_file, 'r') as f:
                sample_lines = f.readlines()[:2]  # First 2 lines
            
            for i, line in enumerate(sample_lines):
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = parts
                    print(f"     Line {i+1}: class={class_id}, center=({x_center},{y_center}), size=({width},{height})")
                else:
                    print(f"     ⚠️  Invalid format: {line.strip()}")

if __name__ == "__main__":
    quick_label_check()