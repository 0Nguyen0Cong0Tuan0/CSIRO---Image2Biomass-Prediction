"""
CSIRO Image2Biomass Prediction - Complete Solution
Main training script implementing DINOv2, Tweedie Loss, and Hierarchical Constraints
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import GroupKFold

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    TRAIN_IMG_DIR = "train"
    TEST_IMG_DIR = "test"
    OUTPUT_DIR = "outputs"
    
    # Model
    BACKBONE = "vit_small_patch14_dinov2.lvd142m"  # DINOv2 Small
    IMG_SIZE = 518  # DINOv2 optimal size
    NUM_COMPONENTS = 5  # 5 biomass targets
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    TWEEDIE_P = 1.5  # Optimal for compound Poisson-Gamma
    
    # Cross-validation
    N_FOLDS = 5
    FOLD_TO_TRAIN = 0  # Set to None to train all folds
    
    # Competition weights
    TARGET_WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5
    }
    
    TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def get_transforms(img_size: int, mode: str = 'train'):
    """Advanced augmentation pipeline for vegetation imagery"""
    
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, 
                              scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            
            # Photometric (cautious with hue to avoid confusing green/dead)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, 
                               val_shift_limit=20, p=0.3),
            
            # Noise and regularization
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                          fill_value=0, p=0.3),
            
            # Normalization (DINOv2 uses ImageNet stats)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class BiomassDataset(Dataset):
    """Dataset for CSIRO biomass prediction with pivoting from long to wide format"""
    
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, mode: str = 'train'):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode
        
        if mode == 'train':
            # Pivot from long to wide format
            self.df = df.pivot_table(
                index=['image_path', 'Sampling_Date', 'State', 'Species', 
                       'Pre_GSHH_NDVI', 'Height_Ave_cm'],
                columns='target_name',
                values='target'
            ).reset_index()
            
            # Metadata normalization
            self.df['Height_Ave_cm'] = self.df['Height_Ave_cm'].fillna(0)
            self.height_mean = self.df['Height_Ave_cm'].mean()
            self.height_std = self.df['Height_Ave_cm'].std()
            
            self.df['Pre_GSHH_NDVI'] = self.df['Pre_GSHH_NDVI'].fillna(0)
            self.ndvi_mean = self.df['Pre_GSHH_NDVI'].mean()
            self.ndvi_std = self.df['Pre_GSHH_NDVI'].std()
        else:
            # Test set: drop duplicates, keep unique images
            self.df = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.img_dir / row['image_path']
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.mode == 'train':
            # Extract targets in correct order
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Dead_g'],
                row['Dry_Clover_g'],
                row['GDM_g'],
                row['Dry_Total_g']
            ], dtype=torch.float32)
            
            # Metadata features
            date = pd.to_datetime(row['Sampling_Date'])
            day_of_year = date.dayofyear
            sin_date = np.sin(2 * np.pi * day_of_year / 365.0)
            cos_date = np.cos(2 * np.pi * day_of_year / 365.0)
            
            height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-6)
            ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-6)
            
            meta = torch.tensor([sin_date, cos_date, height_norm, ndvi_norm], 
                               dtype=torch.float32)
            
            return image, meta, targets
        else:
            return image, str(row['image_path'])


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class TweedieLoss(nn.Module):
    """Tweedie Loss for zero-inflated continuous data (1 < p < 2)"""
    
    def __init__(self, p: float = 1.5, epsilon: float = 1e-8):
        super().__init__()
        assert 1 < p < 2, "Tweedie power p must be in (1, 2)"
        self.p = p
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred + self.epsilon  # Ensure positivity
        
        # Tweedie deviance
        term1 = -target * torch.pow(pred, 1 - self.p) / (1 - self.p)
        term2 = torch.pow(pred, 2 - self.p) / (2 - self.p)
        
        loss = term1 + term2
        return loss.mean()


class HierarchicalBiomassHead(nn.Module):
    """
    Ratio-constrained regression head ensuring sum(components) = total
    Predicts total biomass + component ratios, then multiplies to get components
    """
    
    def __init__(self, in_dim: int, num_components: int = 5, hidden_dim: int = 512):
        super().__init__()
        self.num_components = num_components
        
        # Common feature extractor
        self.common = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Total biomass branch (scalar)
        self.total_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensure positivity
        )
        
        # Component ratio branch (vector)
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_components)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.common(x)
        
        # Predict total biomass
        total_biomass = self.total_head(feat)  # (batch, 1)
        
        # Predict component ratios
        ratio_logits = self.ratio_head(feat)
        ratios = F.softmax(ratio_logits, dim=1)  # (batch, num_components), sums to 1
        
        # Derive component biomass
        component_biomass = total_biomass * ratios  # (batch, num_components)
        
        return component_biomass, total_biomass, ratios


class DinoBiomassModel(pl.LightningModule):
    """Main model with DINOv2 backbone and hierarchical constraints"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Load DINOv2 backbone
        self.backbone = timm.create_model(
            config.BACKBONE, 
            pretrained=True, 
            num_classes=0  # Remove classification head
        )
        self.embed_dim = self.backbone.num_features
        
        # Freeze early layers (layer-wise freezing strategy)
        self._freeze_early_layers()
        
        # Hierarchical regression head
        self.head = HierarchicalBiomassHead(
            in_dim=self.embed_dim,
            num_components=config.NUM_COMPONENTS,
            hidden_dim=config.HIDDEN_DIM
        )
        
        # Loss function
        self.criterion = TweedieLoss(p=config.TWEEDIE_P)
        
        # Competition weights
        self.register_buffer('weights', torch.tensor([
            config.TARGET_WEIGHTS['Dry_Green_g'],
            config.TARGET_WEIGHTS['Dry_Dead_g'],
            config.TARGET_WEIGHTS['Dry_Clover_g'],
            config.TARGET_WEIGHTS['GDM_g'],
            config.TARGET_WEIGHTS['Dry_Total_g']
        ]))
        
        # Metrics storage
        self.validation_step_outputs = []
    
    def _freeze_early_layers(self):
        """Freeze patch embedding and early transformer blocks"""
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2-3 blocks for fine-tuning
        for name, param in self.backbone.named_parameters():
            # For ViT-Small (12 blocks), unfreeze blocks 10-11
            if any(x in name for x in ['blocks.10', 'blocks.11', 'norm']):
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # DINOv2 feature extraction
        features = self.backbone(x)
        
        # Hierarchical head
        components, total_pred, ratios = self.head(features)
        
        return components, total_pred
    
    def training_step(self, batch, batch_idx):
        images, meta, targets = batch
        
        # Forward pass
        preds, total_pred = self(images)
        
        # Weighted Tweedie loss
        loss = 0
        for i in range(5):
            component_loss = self.criterion(preds[:, i], targets[:, i])
            loss += component_loss * self.weights[i]
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, meta, targets = batch
        preds, total_pred = self(images)
        
        # Weighted Tweedie loss
        loss = 0
        for i in range(5):
            component_loss = self.criterion(preds[:, i], targets[:, i])
            loss += component_loss * self.weights[i]
        
        # MSE for monitoring
        mse_loss = F.mse_loss(preds, targets)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mse', mse_loss, prog_bar=True, on_epoch=True)
        
        # Store predictions for R² calculation
        self.validation_step_outputs.append({
            'preds': preds.detach().cpu(),
            'targets': targets.detach().cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate weighted R² at end of validation epoch"""
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)
        
        # Calculate weighted R²
        r2_score = self._calculate_weighted_r2(all_preds, all_targets)
        self.log('val_r2', r2_score, prog_bar=True)
        
        self.validation_step_outputs.clear()
    
    def _calculate_weighted_r2(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate globally weighted R² as per competition metric"""
        # Flatten and apply weights
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        weights_expanded = self.weights.repeat(preds.size(0))
        
        # Weighted mean
        weighted_mean = (weights_expanded * targets_flat).sum() / weights_expanded.sum()
        
        # Weighted SS_res and SS_tot
        ss_res = (weights_expanded * (targets_flat - preds_flat) ** 2).sum()
        ss_tot = (weights_expanded * (targets_flat - weighted_mean) ** 2).sum()
        
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2.item()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def prepare_data(config: Config):
    """Load and prepare data with stratified group k-fold"""
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Create groups based on location/date to prevent leakage
    train_df['group'] = train_df['image_path'].str.extract(r'(ID\d+)')[0]
    
    # Stratify by total biomass bins
    train_df_pivot = train_df.pivot_table(
        index=['image_path', 'group'],
        columns='target_name',
        values='target'
    ).reset_index()
    
    # Create biomass bins for stratification
    total_biomass = train_df_pivot['Dry_Total_g']
    train_df_pivot['biomass_bin'] = pd.qcut(total_biomass, q=5, labels=False, duplicates='drop')
    
    # Group K-Fold
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    train_df_pivot['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(
        train_df_pivot, 
        train_df_pivot['biomass_bin'], 
        groups=train_df_pivot['group']
    )):
        train_df_pivot.loc[val_idx, 'fold'] = fold
    
    # Merge fold info back to original dataframe
    fold_map = dict(zip(train_df_pivot['image_path'], train_df_pivot['fold']))
    train_df['fold'] = train_df['image_path'].map(fold_map)
    
    return train_df


def train_fold(train_df: pd.DataFrame, fold: int, config: Config):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}\n")
    
    # Split data
    train_fold_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_fold_df = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"Train samples: {len(train_fold_df)}")
    print(f"Val samples: {len(val_fold_df)}")
    
    # Create datasets
    train_dataset = BiomassDataset(
        train_fold_df, 
        config.TRAIN_IMG_DIR,
        transform=get_transforms(config.IMG_SIZE, mode='train'),
        mode='train'
    )
    
    val_dataset = BiomassDataset(
        val_fold_df,
        config.TRAIN_IMG_DIR,
        transform=get_transforms(config.IMG_SIZE, mode='val'),
        mode='train'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = DinoBiomassModel(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.OUTPUT_DIR}/fold_{fold}",
        filename='best-{epoch:02d}-{val_r2:.4f}',
        monitor='val_r2',
        mode='max',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_r2',
        patience=10,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=False
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return checkpoint_callback.best_model_path


# ============================================================================
# INFERENCE
# ============================================================================

def predict(model_path: str, config: Config):
    """Generate predictions for test set"""
    print(f"\n{'='*60}")
    print("Generating Predictions")
    print(f"{'='*60}\n")
    
    # Load model
    model = DinoBiomassModel.load_from_checkpoint(model_path, config=config)
    model.eval()
    model.cuda()
    
    # Load test data
    test_df = pd.read_csv(config.TEST_CSV)
    test_dataset = BiomassDataset(
        test_df,
        config.TEST_IMG_DIR,
        transform=get_transforms(config.IMG_SIZE, mode='val'),
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Predict
    predictions = []
    image_paths = []
    
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.cuda()
            preds, _ = model(images)
            predictions.append(preds.cpu().numpy())
            image_paths.extend(paths)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Create submission
    submission_rows = []
    for idx, img_path in enumerate(image_paths):
        img_id = Path(img_path).stem
        for i, target_name in enumerate(config.TARGET_NAMES):
            sample_id = f"{img_id}__{target_name}"
            target_value = predictions[idx, i]
            submission_rows.append({
                'sample_id': sample_id,
                'target': target_value
            })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(f"{config.OUTPUT_DIR}/submission.csv", index=False)
    print(f"Submission saved to {config.OUTPUT_DIR}/submission.csv")
    
    return submission_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data
    train_df = prepare_data(config)
    
    # Train
    if config.FOLD_TO_TRAIN is not None:
        # Train single fold
        best_model_path = train_fold(train_df, config.FOLD_TO_TRAIN, config)
        # Generate predictions
        predict(best_model_path, config)
    else:
        # Train all folds
        best_models = []
        for fold in range(config.N_FOLDS):
            best_model_path = train_fold(train_df, fold, config)
            best_models.append(best_model_path)
        
        print("\nAll folds trained. Best models:")
        for i, path in enumerate(best_models):
            print(f"Fold {i}: {path}")


if __name__ == "__main__":
    main()