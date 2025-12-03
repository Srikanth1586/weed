import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def analyze_training_results():
    """Comprehensive analysis of training results"""
    print("📊 Analyzing Training Results...")
    
    results_dir = Path("weed-detector/results/train")
    results_file = results_dir / "results.csv"
    
    if not results_file.exists():
        print("❌ Results file not found")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Create analysis dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weed Detection - Training Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Training Convergence
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', alpha=0.7)
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', alpha=0.7)
    axes[0, 0].set_title('Training Convergence')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. mAP Progress
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='blue')
    axes[0, 1].set_title('Detection Accuracy (mAP)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Trade-off
    axes[1, 0].plot(df['metrics/precision(B)'], df['metrics/recall(B)'], 'o-', alpha=0.6)
    axes[1, 0].set_title('Precision-Recall Trade-off')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning Rate Effect
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], color='orange')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weed-detector/results/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("\n🔍 Training Insights:")
    print(f"   Final mAP@50: {df['metrics/mAP50(B)'].iloc[-1]:.3f}")
    print(f"   Best mAP@50: {df['metrics/mAP50(B)'].max():.3f}")
    print(f"   Final Precision: {df['metrics/precision(B)'].iloc[-1]:.3f}")
    print(f"   Final Recall: {df['metrics/recall(B)'].iloc[-1]:.3f}")
    
    # Check for overfitting
    train_final = df['train/box_loss'].iloc[-1]
    val_final = df['val/box_loss'].iloc[-1]
    overfitting_ratio = val_final / train_final
    
    if overfitting_ratio > 1.5:
        print("⚠️  Potential overfitting detected")
    else:
        print("✅ Good generalization")

if __name__ == "__main__":
    analyze_training_results()