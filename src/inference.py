import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Optional
from pathlib import Path

from configs.default import Config

class EuroSATClassifier:
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.classes = Config.CLASSES
        
        # Configure threading
        torch.set_num_threads(Config.TORCH_THREADS)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logging.info(f"EuroSAT Classifier initialized on {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model with correct architecture matching training"""
        try:
            # Create model with same architecture as training
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
            in_features = model.classifier[0].in_features
            
            # Use same classifier architecture as training (simple dropout + linear)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, len(self.classes))
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
            
            # Load with flexibility for missing/unexpected keys
            incompatible = model.load_state_dict(state_dict, strict=False)
            
            if incompatible.missing_keys:
                logging.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logging.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            
            model = model.to(self.device)
            model.eval()
            
            logging.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict_image(self, image_path: str, use_tta: bool = False, return_gradcam: bool = False) -> Dict:
        """Predict land cover class for an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs[0], dim=0).cpu().numpy()
            
            # Get top prediction
            top_idx = np.argmax(probabilities)
            top_prob = probabilities[top_idx]
            predicted_class = self.classes[top_idx]
            
            # Create probability dictionary
            all_probabilities = dict(zip(self.classes, probabilities.tolist()))
            
            # Get top 5 predictions
            top_classes = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
            
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': float(top_prob),
                'all_probabilities': all_probabilities,
                'top_classes': top_classes,
                'filename': Path(image_path).name
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Prediction failed for {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'filename': Path(image_path).name
            }

    def export_to_onnx(self, output_path: str) -> bool:
        """Export model to ONNX format"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logging.info(f"Model exported to ONNX: {output_path}")
            return True
        except Exception as e:
            logging.error(f"ONNX export failed: {str(e)}")
            return False

    def export_to_torchscript(self, output_path: str) -> bool:
        """Export model to TorchScript format"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(output_path)
            logging.info(f"Model exported to TorchScript: {output_path}")
            return True
        except Exception as e:
            logging.error(f"TorchScript export failed: {str(e)}")
            return False