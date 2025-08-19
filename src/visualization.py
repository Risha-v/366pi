# Plot functions (matplotlib, seaborn)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import os
import logging

def get_class_colors() -> Dict[str, str]:
    """Get colors for each class"""
    return {
        'AnnualCrop': '#32CD32',
        'Forest': '#228B22',
        'HerbaceousVegetation': '#9ACD32',
        'Highway': '#696969',
        'Industrial': '#8B4513',
        'Pasture': '#90EE90',
        'PermanentCrop': '#FF6347',
        'Residential': '#FF69B4',
        'River': '#4169E1',
        'SeaLake': '#1E90FF'
    }

class EuroSATVisualizer:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.class_colors = get_class_colors()
        plt.style.use('default')

    def plot_class_distribution(self, counts: Dict[str, int], title: str = "Class Distribution"):
        plt.figure(figsize=(12, 6))
        colors = [self.class_colors.get(cls, '#3498db') for cls in counts.keys()]
        bars = plt.bar(counts.keys(), counts.values(), color=colors)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt

    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = True):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   square=True, cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return plt

    def plot_training_history(self, history: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        if 'val_f1' in history:
            axes[1, 0].plot(history['val_f1'], label='Val F1', color='green')
            axes[1, 0].set_title('Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate plot (if available)
        axes[1, 1].text(0.5, 0.5, 'Additional Metrics\nCould be Added Here',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Additional Metrics')
        
        plt.tight_layout()
        return fig

    def plot_prediction_results(self, image_path: str, prediction: Dict, actual: Optional[str] = None):
        try:
            img = Image.open(image_path)
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display image
            axes[0].imshow(img)
            axes.axis('off')
            
            title = f"Predicted: {prediction['predicted_class']} ({prediction['confidence']*100:.1f}%)"
            if actual:
                title += f"\nActual: {actual}"
                if actual != prediction['predicted_class']:
                    title += " ❌"
                else:
                    title += " ✅"
            axes[0].set_title(title, fontsize=12)
            
            # Display probabilities
            classes = list(prediction['all_probabilities'].keys())
            probs = list(prediction['all_probabilities'].values())
            colors = [self.class_colors.get(cls, '#3498db') for cls in classes]
            
            bars = axes[1].barh(classes, probs, color=colors)
            axes[9].set_xlim(0, 1)
            axes[9].set_xlabel('Probability')
            axes[9].set_title('Class Probabilities', fontsize=12)
            
            # Highlight predicted class
            pred_idx = classes.index(prediction['predicted_class'])
            bars[pred_idx].set_edgecolor('red')
            bars[pred_idx].set_linewidth(3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.error(f"Failed to create prediction visualization: {str(e)}")
            return None

    def save_plot(self, plot, save_path: str, dpi: int = 300):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plot.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plot.close()
            logging.info(f"Plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot: {str(e)}")
