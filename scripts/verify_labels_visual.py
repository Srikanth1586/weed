import os
import cv2
import random
import numpy as np
from pathlib import Path

def visualize_yolo_labels():
    """
    Visually verify YOLO labels by drawing bounding boxes on images
    """
    print("🔍 Visualizing YOLO labels for verification...")
    
    base_dir = "weed-detector/data/processed"
    splits = ['train', 'val', 'test']
    class_names = ['weed', 'plant']  # Class 0 = weed, Class 1 = plant
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for weeds, Blue for plants
    
    for split in splits:
        print(f"\n📁 Checking {split} split...")
        
        images_dir = os.path.join(base_dir, "images", split)
        labels_dir = os.path.join(base_dir, "labels", split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ Missing directory for {split}")
            continue
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No images found in {split}")
            continue
        
        print(f"📊 Found {len(image_files)} images in {split}")
        
        # Check random samples
        sample_size = min(3, len(image_files))
        sample_images = random.sample(image_files, sample_size)
        
        for img_file in sample_images:
            # Paths
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Could not load image: {img_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Load and draw labels
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                print(f"📄 {img_file}: {len(lines)} annotations")
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, img_width-1))
                        y1 = max(0, min(y1, img_height-1))
                        x2 = max(0, min(x2, img_width-1))
                        y2 = max(0, min(y2, img_height-1))
                        
                        # Draw bounding box
                        color = colors[class_id]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_names[class_id]}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display image
                cv2.imshow(f"{split} - {img_file}", image)
                print("💡 Press any key to continue to next image...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print(f"⚠️ No label file for: {img_file}")

def check_label_statistics():
    """
    Check statistics about the generated labels
    """
    print("\n📊 Label Statistics")
    print("=" * 50)
    
    base_dir = "weed-detector/data/processed"
    splits = ['train', 'val', 'test']
    class_names = ['weed', 'plant']
    
    for split in splits:
        labels_dir = os.path.join(base_dir, "labels", split)
        
        if not os.path.exists(labels_dir):
            print(f"❌ Labels directory not found: {labels_dir}")
            continue
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        if not label_files:
            print(f"❌ No label files in {split}")
            continue
        
        total_annotations = 0
        class_counts = {0: 0, 1: 0}
        bbox_sizes = []
        
        for label_file in label_files:
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()
            
            total_annotations += len(lines)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    bbox_sizes.append(width * height)  # Normalized area
        
        print(f"\n📁 {split.upper()} Split:")
        print(f"   Label files: {len(label_files)}")
        print(f"   Total annotations: {total_annotations}")
        print(f"   Average annotations per image: {total_annotations/len(label_files):.2f}")
        print(f"   Class distribution:")
        for class_id, count in class_counts.items():
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"     - {class_names[class_id]}: {count} ({percentage:.1f}%)")
        
        if bbox_sizes:
            print(f"   BBox size stats:")
            print(f"     - Average area: {np.mean(bbox_sizes):.4f}")
            print(f"     - Min area: {np.min(bbox_sizes):.4f}")
            print(f"     - Max area: {np.max(bbox_sizes):.4f}")

if __name__ == "__main__":
    print("🎯 YOLO Label Verification")
    print("=" * 50)
    
    # Check statistics first
    check_label_statistics()
    
    # Ask user if they want visual verification
    print("\n👀 Would you like to visually verify labels?")
    print("This will open images with bounding boxes drawn.")
    response = input("Type 'yes' to continue or 'no' to skip: ")
    
    if response.lower() in ['yes', 'y']:
        visualize_yolo_labels()
    
    print("\n✅ Verification complete!")