import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from configs.default import Config
from src.utils import seed_everything, get_class_weights, mixup_data, cutmix_data

class EuroSATDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = sorted([d.name for d in (Path(root_dir) / 'train').iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                logging.warning(f"Class directory not found: {cls_dir}")
                continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append((str(cls_dir / img_name), self.class_to_idx[cls]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {str(e)}")
            placeholder = torch.zeros(3, *Config.IMAGE_SIZE) if self.transform else Image.new('RGB', Config.IMAGE_SIZE)
            return placeholder, 0

class MobileNetV3Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        seed_everything(Config.SEED)
        
        self.device = self._get_device()
        self.best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        if torch.cuda.is_available() and Config.USE_AMP:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        if not torch.cuda.is_available():
            logging.warning("CUDA not available - disabling mixed precision training")
        
        self._setup_system()
        self._create_log_dirs()
        self._setup()
        
        # FIXED: Added missing 'val_accuracy' key
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_accuracy': [],  # FIXED: Added this missing key
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            mem_info = torch.cuda.mem_get_info()
            if (mem_info[0] / mem_info[1]) < 0.2:
                logging.warning("Low GPU memory available, falling back to CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        return torch.device('cpu')

    def _setup_system(self) -> None:
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(Config.TORCH_THREADS)
        os.environ['OMP_NUM_THREADS'] = str(Config.TORCH_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(Config.TORCH_THREADS)

    def _create_log_dirs(self) -> None:
        self.tensorboard_dir = Config.LOGS_DIR / 'tensorboard' / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def _setup(self) -> None:
        try:
            if not Path(self.config['data_path']).exists():
                raise FileNotFoundError(f"Data path not found: {self.config['data_path']}")
            
            metadata_path = Path(self.config['data_path']) / 'metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            
            # Use ImageNet normalization to prevent NaN issues
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(Config.IMAGE_SIZE[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            
            self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(Config.IMAGE_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            
            self._init_datasets()
            self._init_model()
            self._init_optimizer()
            
        except Exception as e:
            logging.error(f"Setup failed: {str(e)}")
            raise

    def _init_datasets(self) -> None:
        try:
            self.train_dataset = EuroSATDataset(
                self.config['data_path'], 'train', self.train_transform)
            self.val_dataset = EuroSATDataset(
                self.config['data_path'], 'val', self.val_transform)
            
            logging.info(f"Training samples: {len(self.train_dataset)}")
            logging.info(f"Validation samples: {len(self.val_dataset)}")
            
            loader_args = {
                'batch_size': self.config['batch_size'],
                'num_workers': 2,
                'pin_memory': self.device.type == 'cuda'
            }
            
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_args)
            self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_args)
            
        except Exception as e:
            logging.error(f"Failed to initialize datasets: {str(e)}")
            raise

    def _init_model(self) -> None:
        try:
            self.model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
            in_features = self.model.classifier[0].in_features
            
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, len(self.train_dataset.classes))
            )
            
            for m in self.model.classifier:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
            
            self.model = self.model.to(self.device)
            logging.info(f"Model initialized on {self.device}")
            
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise

    def _init_optimizer(self) -> None:
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            with autocast(enabled=Config.USE_AMP and self.scaler is not None):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at batch {batch_idx}, skipping")
                continue
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            if batch_idx % 100 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with autocast(enabled=Config.USE_AMP and self.scaler is not None):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                if not torch.isnan(loss):
                    running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
        epoch_acc = 100. * correct / total
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # FIXED: Better handling of history keys
        for key, value in metrics.items():
            if key != 'loss':
                hist_key = f"val_{key}"
                if hist_key not in self.history:
                    self.history[hist_key] = []
                self.history[hist_key].append(value)
                self.writer.add_scalar(f'{key.capitalize()}/val', value, epoch)
            else:
                self.history['val_loss'].append(value)
                self.writer.add_scalar('Loss/val', value, epoch)
        
        return metrics

    def train(self) -> float:
        best_f1 = 0.0
        early_stop_counter = 0
        
        logging.info("Starting training...")
        
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                train_loss, train_acc = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)
                
                self.scheduler.step(val_metrics['f1'])
                
                logging.info(
                    f"Epoch {epoch}/{self.config['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
                
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    early_stop_counter = 0
                    
                    self.best_metrics = {
                        'accuracy': val_metrics['accuracy'],
                        'precision': val_metrics['precision'],
                        'recall': val_metrics['recall'],
                        'f1': val_metrics['f1']
                    }
                    
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.config['patience']:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % self.config['checkpoint_interval'] == 0:
                    self._save_checkpoint(epoch)
                
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.writer.close()
        
        return best_f1

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config,
            'history': self.history,
            'classes': self.train_dataset.classes
        }
        
        if self.scaler is not None:
            state['scaler_state'] = self.scaler.state_dict()
        
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        save_path = save_dir / filename
        
        try:
            torch.save(state, save_path)
            logging.info(f"Saved checkpoint to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")
            raise