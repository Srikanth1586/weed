import os
import yaml
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import seaborn as sns
from datetime import datetime

class TrainingVisualizer:
    """Comprehensive training metrics visualization"""
    
    def __init__(self, results_dir="weed-detector/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def setup_tensorboard(self):
        """Setup TensorBoard for real-time monitoring"""
        print("📊 TensorBoard will be available at: http://localhost:6006/")
        print("💡 Run this command in another terminal: tensorboard --logdir=weed-detector/results/tensorboard")
        
    def create_training_plots(self, results_path):
        """Create comprehensive training plots"""
        print("\n📈 Generating training visualization...")
        
        # Load results CSV
        results_file = Path(results_path) / "results.csv"
        if not results_file.exists():
            print("❌ Results file not found")
            return
            
        df = pd.read_csv(results_file)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLOv8 Training Metrics - Weed Detection', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        self._plot_loss_curves(df, axes[0, 0])
        
        # Plot 2: mAP metrics
        self._plot_map_metrics(df, axes[0, 1])
        
        # Plot 3: Precision-Recall
        self._plot_precision_recall(df, axes[0, 2])
        
        # Plot 4: Learning rate
        self._plot_learning_rate(df, axes[1, 0])
        
        # Plot 5: Class-wise performance
        self._plot_class_performance(df, axes[1, 1])
        
        # Plot 6: Training speed
        self._plot_training_speed(df, axes[1, 2])
        
        plt.tight_layout()
        plot_path = self.results_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plots saved: {plot_path}")
        
        # Create individual high-quality plots
        self._create_individual_plots(df)
        
    def _plot_loss_curves(self, df, ax):
        """Plot training and validation loss curves"""
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
            ax.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
            ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, linestyle='--')
            ax.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_map_metrics(self, df, ax):
        """Plot mAP metrics"""
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', linewidth=2, color='green')
            ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', linewidth=2, color='blue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('Mean Average Precision (mAP)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_precision_recall(self, df, ax):
        """Plot precision and recall"""
        if 'metrics/precision(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='red')
            ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Precision & Recall')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_learning_rate(self, df, ax):
        """Plot learning rate schedule"""
        if 'lr/pg0' in df.columns:
            ax.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
    def _plot_class_performance(self, df, ax):
        """Plot class-wise performance"""
        class_cols = [col for col in df.columns if 'class' in col.lower() and 'loss' not in col.lower()]
        if class_cols:
            for col in class_cols[:2]:  # Show first 2 classes
                ax.plot(df['epoch'], df[col], label=col, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Performance')
            ax.set_title('Class-wise Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Class metrics not available\nin results file', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Class-wise Performance')
            
    def _plot_training_speed(self, df, ax):
        """Plot training speed metrics"""
        if 'time' in df.columns:
            ax.plot(df['epoch'], df['time'], label='Epoch Time (s)', linewidth=2, color='brown')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Speed')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Time metrics not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Speed')
    
    def _create_individual_plots(self, df):
        """Create individual high-quality plots"""
        plots_dir = self.results_dir / "individual_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Loss comparison
        if 'train/box_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Bounding Box Loss - Training vs Validation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / "box_loss_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

def validate_dataset_paths():
    """Validate all dataset paths exist"""
    print("🔍 Validating dataset paths...")
    
    base_dir = Path("weed-detector").absolute()
    processed_dir = base_dir / "data" / "processed"
    
    required_paths = [
        processed_dir / "images" / "train",
        processed_dir / "images" / "val", 
        processed_dir / "images" / "test",
        processed_dir / "labels" / "train",
        processed_dir / "labels" / "val",
        processed_dir / "labels" / "test"
    ]
    
    all_paths_exist = True
    for path in required_paths:
        if path.exists():
            file_count = len(list(path.glob("*")))
            print(f"✅ {path.name}: {file_count} files")
        else:
            print(f"❌ {path.name}: MISSING")
            all_paths_exist = False
    
    return all_paths_exist

def setup_training_environment():
    """Setup and verify training environment"""
    print("🔧 Setting up training environment...")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✅ CUDA: {torch.version.cuda}")
    else:
        print("❌ No GPU detected - training will be slow!")
        return False
    
    # Validate dataset paths
    if not validate_dataset_paths():
        print("❌ Dataset paths validation failed")
        return False
        
    # Check dataset.yaml
    dataset_path = Path("weed-detector/data/processed/dataset.yaml").absolute()
    if not dataset_path.exists():
        print(f"❌ Dataset YAML not found: {dataset_path}")
        return False
        
    print(f"✅ Dataset YAML found: {dataset_path}")
    return True

def train_weed_detector():
    """Main training function for weed detection"""
    print("🎯 Starting YOLOv8 Weed Detection Training")
    print("=" * 60)
    
    # Setup environment
    if not setup_training_environment():
        print("❌ Training environment setup failed")
        return
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    visualizer.setup_tensorboard()
    
    # Load model
    print("\n📦 Loading YOLOv8 Medium model...")
    model = YOLO('yolov8m.pt')
    
    # Use absolute path for dataset
    dataset_path = Path("weed-detector/data/processed/dataset.yaml").absolute()
    
    # Training configuration
    training_config = {
        'data': str(dataset_path),  # Absolute path
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,  # Will auto-adjust based on VRAM
        'patience': 10,  # Early stopping
        'save': True,
        'exist_ok': True,  # Overwrite existing runs
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # Distribution Focal Loss gain
        'close_mosaic': 10,  # Disable mosaic last epochs
        'project': 'weed-detector/results',
        'name': 'train',
        'verbose': True
    }
    
    print("\n⚙️ Training Configuration:")
    print(f"   data: {training_config['data']}")
    print(f"   epochs: {training_config['epochs']}")
    print(f"   batch: {training_config['batch']}")
    
    print("\n🚀 Starting training...")
    print("💡 Monitor progress with TensorBoard: tensorboard --logdir=weed-detector/results")
    
    # Start training
    try:
        results = model.train(**training_config)
        
        print("\n✅ Training completed successfully!")
        
        # Generate visualizations
        results_path = Path('weed-detector/results/train')
        visualizer.create_training_plots(results_path)
        
        # Validate on test set
        print("\n🧪 Running final validation on test set...")
        test_results = model.val(split='test')
        
        # Print final metrics
        print("\n📊 Final Model Performance:")
        print(f"   mAP@50: {test_results.box.map50:.3f}")
        print(f"   mAP@50-95: {test_results.box.map:.3f}")
        print(f"   Precision: {test_results.box.precision:.3f}")
        print(f"   Recall: {test_results.box.recall:.3f}")
        
        # Save best model
        best_model_path = "weed-detector/models/training/best.pt"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        # Copy the best model from results
        results_best_model = "weed-detector/results/train/weights/best.pt"
        if os.path.exists(results_best_model):
            import shutil
            shutil.copy2(results_best_model, best_model_path)
            print(f"✅ Best model saved: {best_model_path}")
        else:
            print("⚠️ Best model not found in results directory")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    print("\n🎉 Training pipeline completed!")
    print("📁 Results saved in: weed-detector/results/")
    print("🚀 Next: Run webcam_detection.py for real-time testing")

if __name__ == "__main__":
    train_weed_detector()