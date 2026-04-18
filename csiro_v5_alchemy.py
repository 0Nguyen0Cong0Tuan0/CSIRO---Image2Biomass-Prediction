import os
import gc
import math
import random
import warnings
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import cv2
import timm
import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

# Scikit-learn / ML imports for Stage 1 Baseline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION & SEEDING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    TRAIN_IMG_DIR: Path = DATA_PATH / "train"
    TEST_IMG_DIR: Path = DATA_PATH / "test"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42
    NUM_WORKERS: int = 4

@dataclass
class AlchemyCFG:
    """🔮 Alchemical Hyperparameters from Research Codex"""
    # Architecture (Sec 2.3 - Compositional-Total)
    USE_COMPOSITIONAL_HEAD: bool = True
    
    # Loss Functions (Sec 5)
    HCL_LAMBDA: float = 0.15          # Hierarchical Constraint Loss weight
    TWEEDIE_POWER: float = 1.5        # For zero-inflated Dead biomass
    TOTAL_LOSS_WEIGHT: float = 0.5    # Match competition weight
    
    # Metadata Fusion (Sec 4.1 - FiLM)
    FILM_HIDDEN_DIM: int = 128
    USE_ENHANCED_FILM: bool = True
    
    # DINOv2 Settings (Sec 3.2)
    # Using 'reg4' variant to handle artifacts
    DINO_VARIANT: str = "vit_large_patch14_reg4_dinov2.lvd142m" 
    FREEZE_BACKBONE: bool = True      # Only train heads to prevent overfitting small data
    
    # Training
    N_FOLDS: int = 5
    BATCH_SIZE: int = 8               # Small batch size for large resolution
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 10                  # Short training due to pre-trained backbone
    IMG_SIZE: int = 518               # Native DINOv2 resolution

TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
# Competition Weights: Total is king (0.5)
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 
                  'GDM_g': 0.2, 'Dry_Total_g': 0.5}

cfg = Config()
alchemy_cfg = AlchemyCFG()

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seeding(cfg.SEED)

# ═══════════════════════════════════════════════════════════════════
# 📊 DATA UTILITIES (PIVOT/MELT/METADATA)
# ═══════════════════════════════════════════════════════════════════

def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Converts long format (competition) to wide format (one row per image)."""
    if 'target' in df.columns:
        index_cols = ['image_path', 'Sampling_Date', 'State', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
        # Filter only existing columns
        index_cols = [c for c in index_cols if c in df.columns]
        
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index=index_cols, 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    else:
        # Test set structure
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index='image_path', 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame, test_long: pd.DataFrame) -> pd.DataFrame:
    """Converts wide predictions back to long format for submission."""
    # Ensure all target columns exist
    for col in TARGET_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    
    # Merge with sample_id from original test file
    submission = test_long[['sample_id', 'image_path', 'target_name']].merge(
        melted,
        on=['image_path', 'target_name'],
        how='left'
    )[['sample_id', 'target']]
    
    return submission

# ═══════════════════════════════════════════════════════════════════
# 🧠 IDEA #1 & #3: COMPOSITIONAL HEAD & DINOv2 REGISTERS (Sec 2.3, 3.2)
# ═══════════════════════════════════════════════════════════════════

class CompositionalHead(nn.Module):
    """
    🎯 Compositional-Total Head (Research Sec 2.3.1)
    
    Theory:
    Instead of predicting 5 independent values, predict:
      1. Total Magnitude (scalar) - Primary task (0.5 weight)
      2. Component Ratios (3-simplex) - Species composition
    Then reconstruct targets deterministically.
    """
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(64, input_dim // 4)
        
        # Magnitude Branch: Predict Total Biomass
        self.magnitude_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(beta=1.0)  # Enforce Non-negative Total
        )
        
        # Ratio Branch: Predict Species Composition (Green, Clover, Dead)
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)  # Logits for [Green, Clover, Dead]
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Step 1: Predict magnitude and ratios
        total = self.magnitude_head(features)  # (B, 1)
        ratio_logits = self.ratio_head(features)  # (B, 3)
        ratios = F.softmax(ratio_logits, dim=-1)  # Sum to 1
        
        # Step 2: Deterministic reconstruction (Eq 3 from research)
        green = total * ratios[:, 0:1]
        clover = total * ratios[:, 1:2]
        dead = total * ratios[:, 2:3]
        
        # Hierarchical Constraints
        gdm = green + clover       # GDM = Green + Clover
        # total_recon = gdm + dead # Guaranteed by algebra: T*(p1+p2+p3) = T*1 = T
        
        return {
            'total': total,
            'ratios': ratios,
            'green': green,
            'clover': clover, 
            'dead': dead,
            'gdm': gdm
        }

def extract_clean_features(model, x, has_registers=True):
    """
    🧹 Extract DINOv2 features with register token filtering (Sec 3.2).
    
    Standard DINOv2 w/ reg: [CLS, REG1, REG2, REG3, REG4, PATCH_1, ..., PATCH_N]
    We must discard registers before spatial pooling to avoid artifacts.
    """
    # Get full sequence from backbone
    features = model.forward_features(x)  # (B, N_tokens, D)
    
    if has_registers:
        cls_token = features[:, 0]      # Global context
        # Skip CLS (idx 0) + 4 registers (idx 1-4) -> Start at 5
        patch_tokens = features[:, 5:]  
        
        # Spatial pooling on clean patches only
        spatial_features = patch_tokens.mean(dim=1)
        
        # Concatenate for rich representation (Global + Local Density)
        return torch.cat([cls_token, spatial_features], dim=-1)
    else:
        return features[:, 0]

# ═══════════════════════════════════════════════════════════════════
# 🎛️ IDEA #2: METADATA FiLM FUSION (Sec 4.1)
# ═══════════════════════════════════════════════════════════════════

class MetadataFiLMGenerator(nn.Module):
    """
    🎚️ Generates FiLM parameters (gamma, beta) from metadata.
    Formula: FiLM(F | z) = γ(z) · F + β(z)
    """
    def __init__(self, feature_dim: int, num_states: int = 6, num_seasons: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        # Learnable embeddings
        self.state_embed = nn.Embedding(num_states, 8)
        self.season_embed = nn.Embedding(num_seasons, 8)
        
        # Metadata input dimension: 
        # 1 (NDVI) + 1 (Height) + 8 (State) + 8 (Season) = 18
        metadata_input_dim = 18 
        
        self.mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2)  # Output gamma and beta
        )
        
    def forward(self, ndvi, height, state_idx, season_idx):
        # Normalize scalars (Robust scaling logic assumed here or in dataset)
        # Assuming inputs are already roughly scaled or we apply simple normalization:
        ndvi_norm = (ndvi - 0.5) / 0.2
        height_norm = (height - 10.0) / 10.0
        
        state_emb = self.state_embed(state_idx)
        season_emb = self.season_embed(season_idx)
        
        # Concatenate
        z = torch.cat([ndvi_norm.unsqueeze(1), height_norm.unsqueeze(1), state_emb, season_emb], dim=1)
        
        # Generate params
        params = self.mlp(z)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta

class FiLMLayer(nn.Module):
    def forward(self, features, gamma, beta):
        # Broadcast gamma/beta (B, D) to features (B, D)
        return features * (1.0 + gamma) + beta

# ═══════════════════════════════════════════════════════════════════
# 🏛️ MODEL: ALCHEMICAL DINOv2
# ═══════════════════════════════════════════════════════════════════

class AlchemicalDINOv2(nn.Module):
    def __init__(self, backbone_name=alchemy_cfg.DINO_VARIANT, freeze_backbone=True):
        super().__init__()
        
        # 1. Vision Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0
        )
        self.feature_dim = self.backbone.num_features
        self.has_registers = 'reg' in backbone_name
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 2. Metadata Conditioning
        # Combined dim = CLS + Spatial Mean = 2 * feature_dim
        self.combined_dim = self.feature_dim * 2
        
        self.film_generator = MetadataFiLMGenerator(
            feature_dim=self.combined_dim,
            hidden_dim=alchemy_cfg.FILM_HIDDEN_DIM
        )
        self.film_layer = FiLMLayer()
        
        # 3. Compositional Head
        self.head = CompositionalHead(self.combined_dim, dropout=0.2)
        
    def forward(self, images, metadata):
        # Extract visual features
        visual_features = extract_clean_features(
            self.backbone, images, has_registers=self.has_registers
        )
        
        # Generate FiLM params
        gamma, beta = self.film_generator(
            metadata['ndvi'], metadata['height'], 
            metadata['state_idx'], metadata['season_idx']
        )
        
        # Modulate
        features = self.film_layer(visual_features, gamma, beta)
        
        # Predict
        return self.head(features)

# ═══════════════════════════════════════════════════════════════════
# 🔥 LOSS FUNCTIONS (Sec 5)
# ═══════════════════════════════════════════════════════════════════

class TweedieLoss(nn.Module):
    """💀 Tweedie loss for zero-inflated biomass (Sec 5.1)."""
    def __init__(self, power=1.5):
        super().__init__()
        self.p = power
        
    def forward(self, y_pred, y_true):
        # Epsilon to avoid log(0)
        epsilon = 1e-6
        y_pred = torch.clamp(y_pred, min=epsilon)
        
        a = y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        loss = -a + b
        return loss.mean()

class CombinedBiomassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.tweedie = TweedieLoss(power=alchemy_cfg.TWEEDIE_POWER)
        
    def forward(self, pred_dict, targets):
        # targets: [Green, Dead, Clover, GDM, Total]
        
        # 1. Weighted MSE for all components
        loss_green = self.mse(pred_dict['green'], targets[:, 0:1]) * TARGET_WEIGHTS['Dry_Green_g']
        loss_dead = self.mse(pred_dict['dead'], targets[:, 1:2]) * TARGET_WEIGHTS['Dry_Dead_g']
        loss_clover = self.mse(pred_dict['clover'], targets[:, 2:3]) * TARGET_WEIGHTS['Dry_Clover_g']
        loss_gdm = self.mse(pred_dict['gdm'], targets[:, 3:4]) * TARGET_WEIGHTS['GDM_g']
        loss_total = self.mse(pred_dict['total'], targets[:, 4:5]) * TARGET_WEIGHTS['Dry_Total_g']
        
        weighted_mse = (loss_green + loss_dead + loss_clover + loss_gdm + loss_total).mean()
        
        # 2. Tweedie Loss for Dead Biomass (Zero-inflated handling)
        tweedie_term = self.tweedie(pred_dict['dead'], targets[:, 1:2])
        
        # 3. Hierarchical Consistency (Soft Penalty)
        # Should be near zero by design, but gradients helps stability
        cons_err = torch.abs(pred_dict['gdm'] - (pred_dict['green'] + pred_dict['clover'])).mean()
        
        total_loss = weighted_mse + 0.1 * tweedie_term + alchemy_cfg.HCL_LAMBDA * cons_err
        return total_loss

# ═══════════════════════════════════════════════════════════════════
# 📦 DATASET & TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

class BiomassDataset(Dataset):
    def __init__(self, df, transform, img_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Label Encoding (Simple manual mapping for consistency)
        self.states = {'NSW': 0, 'WA': 1, 'VIC': 2, 'SA': 3, 'TAS': 4, 'QLD': 5}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image Loading
        img_name = os.path.basename(row['image_path'])
        path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(path)
        if img is None: # Safety for missing images
            img = np.zeros((518, 518, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            img = self.transform(image=img)['image']
            
        # Metadata Processing
        state_str = row.get('State', 'NSW')
        state_idx = self.states.get(state_str, 0)
        
        # Heuristic Season from Date (if available) or random default
        # Assuming date format YYYY-MM-DD
        try:
            month = int(str(row.get('Sampling_Date', '2020-01-01')).split('-')[1])
            if month in [9, 10, 11]: season_idx = 0 # Spring
            elif month in [12, 1, 2]: season_idx = 1 # Summer
            elif month in [3, 4, 5]: season_idx = 2 # Autumn
            else: season_idx = 3 # Winter
        except:
            season_idx = 0
            
        metadata = {
            'ndvi': torch.tensor(row.get('Pre_GSHH_NDVI', 0.5), dtype=torch.float32),
            'height': torch.tensor(row.get('Height_Ave_cm', 10.0), dtype=torch.float32),
            'state_idx': torch.tensor(state_idx, dtype=torch.long),
            'season_idx': torch.tensor(season_idx, dtype=torch.long)
        }
        
        if self.is_train:
            # Targets: [Green, Dead, Clover, GDM, Total]
            targets = torch.tensor([
                row.get('Dry_Green_g', 0),
                row.get('Dry_Dead_g', 0),
                row.get('Dry_Clover_g', 0),
                row.get('GDM_g', 0),
                row.get('Dry_Total_g', 0)
            ], dtype=torch.float32)
            return img, metadata, targets
        
        return img, metadata

def train_dl_model(train_df, test_df):
    """Executes the Deep Learning Training and Inference Pipeline."""
    
    # Stratified K-Fold
    kf = KFold(n_splits=alchemy_cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    # Placeholders for OOF and Test Preds
    oof_preds = np.zeros((len(train_df), 5))
    test_preds = np.zeros((len(test_df), 5))
    
    # Transforms
    train_aug = A.Compose([
        A.Resize(alchemy_cfg.IMG_SIZE, alchemy_cfg.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_aug = A.Compose([
        A.Resize(alchemy_cfg.IMG_SIZE, alchemy_cfg.IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    
    criterion = CombinedBiomassLoss()
    
    print(f"\n🔮 Starting Alchemical DINOv2 Training ({alchemy_cfg.N_FOLDS} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"   > Fold {fold+1}/{alchemy_cfg.N_FOLDS}")
        
        # Subsets
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Datasets
        train_ds = BiomassDataset(fold_train, train_aug, cfg.TRAIN_IMG_DIR, is_train=True)
        val_ds = BiomassDataset(fold_val, val_aug, cfg.TRAIN_IMG_DIR, is_train=True)
        test_ds = BiomassDataset(test_df, val_aug, cfg.TEST_IMG_DIR, is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=alchemy_cfg.BATCH_SIZE, shuffle=True, 
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=alchemy_cfg.BATCH_SIZE, shuffle=False, 
                                num_workers=cfg.NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=alchemy_cfg.BATCH_SIZE, shuffle=False, 
                                 num_workers=cfg.NUM_WORKERS)
        
        # Model & Opt
        model = AlchemicalDINOv2().to(cfg.DEVICE)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                      lr=alchemy_cfg.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()
        
        # Training Loop
        for epoch in range(alchemy_cfg.EPOCHS):
            model.train()
            # Basic training loop (simplified for brevity)
            for imgs, metas, tgts in train_loader:
                imgs, tgts = imgs.to(cfg.DEVICE), tgts.to(cfg.DEVICE)
                metas = {k: v.to(cfg.DEVICE) for k, v in metas.items()}
                
                with torch.cuda.amp.autocast():
                    preds = model(imgs, metas)
                    loss = criterion(preds, tgts)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Validation Inference
        model.eval()
        val_fold_preds = []
        with torch.no_grad():
            for imgs, metas, _ in val_loader:
                imgs = imgs.to(cfg.DEVICE)
                metas = {k: v.to(cfg.DEVICE) for k, v in metas.items()}
                res = model(imgs, metas)
                # Stack: [Green, Dead, Clover, GDM, Total]
                batch_p = torch.stack([res['green'], res['dead'], res['clover'], res['gdm'], res['total']], dim=1)
                val_fold_preds.append(batch_p.squeeze(-1).cpu().numpy())
        
        oof_preds[val_idx] = np.concatenate(val_fold_preds)
        
        # Test Inference
        test_fold_preds = []
        with torch.no_grad():
            for imgs, metas in test_loader:
                imgs = imgs.to(cfg.DEVICE)
                metas = {k: v.to(cfg.DEVICE) for k, v in metas.items()}
                res = model(imgs, metas)
                batch_p = torch.stack([res['green'], res['dead'], res['clover'], res['gdm'], res['total']], dim=1)
                test_fold_preds.append(batch_p.squeeze(-1).cpu().numpy())
        
        test_preds += np.concatenate(test_fold_preds) / alchemy_cfg.N_FOLDS
        
        # Clean up
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

    return oof_preds, test_preds

# ═══════════════════════════════════════════════════════════════════
# 📉 STAGE 1: ML BASELINE (SIMULATED FOR EXECUTION)
# ═══════════════════════════════════════════════════════════════════

def run_ml_baseline(train_df, test_df):
    """
    Runs a Gradient Boosting Regressor baseline.
    In a real run, this would use SigLIP embeddings. 
    Here we use Metadata + simple features to ensure code runs.
    """
    print("\n📊 Stage 1: Running ML Baseline...")
    
    # Feature Engineering
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    target_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    # Handle missing features
    for df in [train_df, test_df]:
        df['Pre_GSHH_NDVI'] = df['Pre_GSHH_NDVI'].fillna(0.5)
        df['Height_Ave_cm'] = df['Height_Ave_cm'].fillna(10.0)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    X_test = test_df[feature_cols].values
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    ml_preds = np.zeros((len(test_df), 5))
    
    for i, target_name in enumerate(target_cols):
        # Train one regressor per target
        model = HistGradientBoostingRegressor(random_state=cfg.SEED)
        
        # Simple averaging over folds
        target_preds = np.zeros(len(test_df))
        for train_idx, _ in kf.split(X_train):
            model.fit(X_train[train_idx], y_train[train_idx, i])
            target_preds += model.predict(X_test) / 5
            
        ml_preds[:, i] = target_preds
        
    return ml_preds

# ═══════════════════════════════════════════════════════════════════
# 🧪 POST-PROCESSING & SUBMISSION
# ═══════════════════════════════════════════════════════════════════

def post_process_biomass_v5(predictions):
    """Safe post-processing with hard constraints."""
    # 1. Clip negative values
    predictions = np.maximum(predictions, 0.0)
    
    # 2. Zero-clamping (Research thresholds)
    # idx: 0=Green, 1=Dead, 2=Clover, 3=GDM, 4=Total
    CLOVER_THRESHOLD = 1.25
    DEAD_THRESHOLD = 1.0
    
    predictions[:, 2] = np.where(predictions[:, 2] < CLOVER_THRESHOLD, 0.0, predictions[:, 2])
    predictions[:, 1] = np.where(predictions[:, 1] < DEAD_THRESHOLD, 0.0, predictions[:, 1])
    
    # 3. Enforce Mass Balance Hierarchy
    # GDM = Green + Clover
    predictions[:, 3] = predictions[:, 0] + predictions[:, 2]
    # Total = GDM + Dead
    predictions[:, 4] = predictions[:, 3] + predictions[:, 1]
    
    return predictions

def main():
    print("="*70)
    print("🔮 CSIRO BIOMASS ALCHEMY v5: COMPLETE PIPELINE")
    print("="*70)
    
    # 1. Load Data
    # In a Kaggle kernel, these paths exist. Locally, adjust cfg.DATA_PATH.
    if not cfg.TRAIN_IMG_DIR.exists():
        print("⚠️ Data paths not found. Creating dummy data for testing structure...")
        os.makedirs(cfg.TRAIN_IMG_DIR, exist_ok=True)
        os.makedirs(cfg.TEST_IMG_DIR, exist_ok=True)
        # Create dummy CSVs
        train_df = pd.DataFrame({
            'sample_id': [f'ID_{i}' for i in range(10)],
            'image_path': [f'train/img_{i}.jpg' for i in range(10)],
            'Pre_GSHH_NDVI': np.random.rand(10),
            'Height_Ave_cm': np.random.rand(10)*20,
            'target_name': 'Dry_Total_g',
            'target': np.random.rand(10)*100
        }) # Simplified logic for dummy creation
        # Real logic expects wide format after pivot, so we mock pivot result directly for train
        train_wide = pd.DataFrame({
            'image_path': [f'train/img_{i}.jpg' for i in range(20)],
            'Pre_GSHH_NDVI': np.random.rand(20),
            'Height_Ave_cm': np.random.rand(20)*20,
            'Dry_Green_g': np.random.rand(20)*50,
            'Dry_Dead_g': np.random.rand(20)*50,
            'Dry_Clover_g': np.random.rand(20)*10,
            'GDM_g': np.random.rand(20)*60,
            'Dry_Total_g': np.random.rand(20)*110
        })
        test_long = pd.DataFrame({
            'sample_id': [f'ID_TEST_{i}_{t}' for i in range(5) for t in TARGET_NAMES],
            'image_path': [f'test/img_{i}.jpg' for i in range(5) for t in TARGET_NAMES],
            'target_name': [t for i in range(5) for t in TARGET_NAMES]
        })
    else:
        # Real Loading
        train_df_raw = pd.read_csv(cfg.DATA_PATH / 'train.csv')
        test_df_raw = pd.read_csv(cfg.DATA_PATH / 'test.csv')
        
        # Pivot Train to Wide
        train_wide = pivot_table(train_df_raw)
        
        # Test Long is needed for submission structure
        test_long = test_df_raw.copy()
        # Pivot Test for Inference
        test_wide = pivot_table(test_df_raw)

    print(f"✅ Loaded: {len(train_wide)} training samples, {len(test_wide if 'test_wide' in locals() else 0)} test samples.")

    # 2. Stage 1: ML Baseline
    try:
        ml_preds = run_ml_baseline(train_wide, test_wide)
    except Exception as e:
        print(f"⚠️ ML Baseline failed: {e}. Using zeros.")
        ml_preds = np.zeros((len(test_wide), 5))

    # 3. Stage 3: Alchemical DL Training & Inference
    try:
        # Check if we can run DL (GPU check or small dummy run)
        if torch.cuda.is_available() or cfg.DEVICE == 'cpu':
            _, dl_preds = train_dl_model(train_wide, test_wide)
        else:
            print("⚠️ No GPU. Skipping DL training.")
            dl_preds = ml_preds # Fallback
    except Exception as e:
        print(f"⚠️ DL Pipeline failed: {e}. Falling back to ML.")
        dl_preds = ml_preds

    # 4. Ensemble & Post-Process
    print("\n⚖️ Ensembling: 0.3 ML + 0.7 DL")
    final_preds_raw = 0.3 * ml_preds + 0.7 * dl_preds
    
    final_preds_proc = post_process_biomass_v5(final_preds_raw)
    
    # 5. Create Submission
    sub_df = melt_table(pd.DataFrame(final_preds_proc, columns=TARGET_NAMES).assign(image_path=test_wide['image_path']), 
                        test_long)
    
    sub_df.to_csv("submission.csv", index=False)
    print("✅ Saved submission.csv")
    print("Sample:\n", sub_df.head())

if __name__ == "__main__":
    main()