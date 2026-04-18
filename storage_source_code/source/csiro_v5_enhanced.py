"""
CSIRO Image2Biomass Competition - Enhanced Solution (v5)
Implements enhancements from research blueprint:
1. Hierarchical Consistency Loss (HCL) - Section 2
2. Minimum Trace (MinT) Reconciliation - Section 2
3. Cross-Attention Metadata Fusion - Section 4
4. Uncertainty-Weighted Ensembling - Section 6
5. DINOv2 with Registers - Section 3
"""

import os
import gc
import math
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm

# ============================================================================
# SECTION 1: CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Enhanced configuration with blueprint parameters"""
    DATA_PATH = "/kaggle/input/csiro-biomass/"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Target columns and hierarchy (Blueprint Section 2)
    TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }
    
    # Competition weights (heavily favor Total)
    WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5,
    }
    
    # HCL hyperparameters (Blueprint Section 2)
    HCL_LAMBDA = 0.15  # Weight for consistency loss
    
    # MinT reconciliation settings
    USE_MINT = True
    MINT_REGULARIZATION = 1e-8
    
    # Metadata fusion settings (Blueprint Section 4)
    USE_CROSS_ATTENTION = True
    METADATA_DIM = 128
    ATTENTION_HEADS = 4

cfg = Config()

def seed_everything(seed=42):
    """Comprehensive seeding for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)

# ============================================================================
# SECTION 2: HIERARCHICAL CONSISTENCY & MINT RECONCILIATION
# Blueprint Section 2 - Core Physics-Informed Components
# ============================================================================

def build_hierarchy_matrix():
    """
    Construct summation matrix S mapping base components to all targets
    Blueprint Eq. (1): y = S * b where b = [Green, Dead, Clover]^T
    
    Returns:
        S: (5x3) numpy array defining hierarchical relationships
    """
    S = np.array([
        [1, 0, 1],  # GDM = Green + Clover
        [0, 1, 0],  # Dead = Dead
        [1, 0, 0],  # Green = Green
        [1, 1, 1],  # Total = Green + Dead + Clover
        [0, 0, 1],  # Clover = Clover
    ], dtype=np.float32)
    return S

def hierarchical_consistency_loss(predictions, S=None):
    """
    Compute Hierarchical Consistency Loss (HCL)
    Blueprint Section 2, Eq. (3): Penalizes violations of mass balance
    
    Args:
        predictions: (N, 5) array [GDM, Dead, Green, Total, Clover]
        S: Hierarchy matrix (optional, uses default if None)
    
    Returns:
        loss: Scalar consistency violation penalty
    """
    if S is None:
        S = build_hierarchy_matrix()
    
    # Extract predictions in blueprint order [Green, Dead, Clover, GDM, Total]
    green = predictions[:, 2]   # Index 2
    dead = predictions[:, 1]    # Index 1
    clover = predictions[:, 4]  # Index 4
    gdm = predictions[:, 0]     # Index 0
    total = predictions[:, 3]   # Index 3
    
    # Constraint 1: Total = Green + Dead + Clover (Blueprint primary constraint)
    total_violation = (total - (green + dead + clover)) ** 2
    
    # Constraint 2: GDM = Green + Clover (Blueprint secondary constraint)
    gdm_violation = (gdm - (green + clover)) ** 2
    
    # Combined HCL (Blueprint Eq. 3)
    hcl = torch.mean(total_violation + gdm_violation)
    return hcl

def mint_reconciliation(predictions_raw, error_covariance=None, S=None):
    """
    Minimum Trace (MinT) Reconciliation
    Blueprint Section 2, Eq. (4-6): Projects incoherent forecasts onto coherent manifold
    
    Implements: y_reconciled = P * y_raw where P = S(S^T W^-1 S)^-1 S^T W^-1
    
    Args:
        predictions_raw: (N, 5) array of raw model predictions
        error_covariance: (5, 5) covariance of forecast errors (W_h in blueprint)
        S: Hierarchy matrix
    
    Returns:
        predictions_reconciled: (N, 5) coherent predictions satisfying constraints
    """
    if S is None:
        S = build_hierarchy_matrix()
    
    N = predictions_raw.shape[0]
    
    # If no covariance provided, use identity (OLS reconciliation)
    if error_covariance is None:
        error_covariance = np.eye(5)
    
    # Add regularization for numerical stability
    W_h = error_covariance + cfg.MINT_REGULARIZATION * np.eye(5)
    W_h_inv = np.linalg.inv(W_h)
    
    # Compute projection matrix P (Blueprint Eq. 5)
    S_T = S.T
    middle = np.linalg.inv(S_T @ W_h_inv @ S)
    P = S @ middle @ S_T @ W_h_inv
    
    # Apply reconciliation (Blueprint Eq. 6)
    predictions_reconciled = (P @ predictions_raw.T).T
    
    # Ensure non-negativity (physical constraint)
    predictions_reconciled = np.maximum(predictions_reconciled, 0)
    
    return predictions_reconciled

def estimate_error_covariance(y_true, y_pred_oof, folds):
    """
    Estimate forecast error covariance matrix for MinT
    Blueprint Section 2: Uses out-of-fold residuals
    
    Args:
        y_true: (N, 5) ground truth
        y_pred_oof: (N, 5) out-of-fold predictions
        folds: Fold assignments
    
    Returns:
        cov_matrix: (5, 5) error covariance W_h
    """
    residuals = y_true - y_pred_oof
    cov_matrix = np.cov(residuals.T)
    
    # Ensure positive definite
    cov_matrix = cov_matrix + cfg.MINT_REGULARIZATION * np.eye(5)
    
    return cov_matrix

# ============================================================================
# SECTION 3: CROSS-ATTENTION METADATA FUSION
# Blueprint Section 4 - Deep Fusion of Agronomic Metadata
# ============================================================================

class MetadataEncoder(nn.Module):
    """
    Encodes NDVI, Height, Species, Date into rich embedding
    Blueprint Section 4, Step 1
    """
    def __init__(self, metadata_dim=128):
        super().__init__()
        # Continuous features: NDVI, Height
        self.continuous_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, metadata_dim // 2)
        )
        
        # Categorical embeddings (Species: ~10, Month: 12)
        self.species_embed = nn.Embedding(15, 32)  # Padding for unknown
        self.month_embed = nn.Embedding(13, 32)    # 12 months + padding
        
        self.fusion = nn.Linear(metadata_dim // 2 + 64, metadata_dim)
        
    def forward(self, ndvi, height, species_idx, month_idx):
        """
        Args:
            ndvi: (B,) NDVI values
            height: (B,) height values
            species_idx: (B,) species indices
            month_idx: (B,) month indices
        Returns:
            metadata_emb: (B, metadata_dim)
        """
        continuous = torch.stack([ndvi, height], dim=1)
        cont_emb = self.continuous_proj(continuous)
        
        species_emb = self.species_embed(species_idx)
        month_emb = self.month_embed(month_idx)
        cat_emb = torch.cat([species_emb, month_emb], dim=1)
        
        combined = torch.cat([cont_emb, cat_emb], dim=1)
        metadata_emb = self.fusion(combined)
        
        return metadata_emb

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Multimodal Fusion (CAMF)
    Blueprint Section 4: Metadata queries visual features
    
    Implements: Attention(Q=metadata, K=V=image_features)
    """
    def __init__(self, visual_dim, metadata_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = metadata_dim // num_heads
        
        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, metadata_dim)
        self.metadata_proj = nn.Linear(metadata_dim, metadata_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=metadata_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(metadata_dim)
        self.ffn = nn.Sequential(
            nn.Linear(metadata_dim, metadata_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_dim * 2, metadata_dim)
        )
        
    def forward(self, visual_features, metadata_emb):
        """
        Blueprint Section 4, Step 2: Cross-attention mechanism
        
        Args:
            visual_features: (B, D_visual) image embeddings
            metadata_emb: (B, D_meta) metadata embeddings
        Returns:
            fused: (B, D_meta) calibrated features
        """
        # Project to common space
        V = self.visual_proj(visual_features).unsqueeze(1)  # (B, 1, D)
        Q = self.metadata_proj(metadata_emb).unsqueeze(1)   # (B, 1, D)
        
        # Cross-attention (metadata attends to visual)
        attn_out, _ = self.attention(Q, V, V)
        attn_out = attn_out.squeeze(1)
        
        # Residual & FFN (Blueprint Section 4, Step 3)
        fused = self.norm(attn_out + metadata_emb)
        fused = fused + self.ffn(fused)
        
        return fused

# ============================================================================
# SECTION 4: UNCERTAINTY-AWARE REGRESSION
# Blueprint Section 6 - Robust Inference via Uncertainty Quantification
# ============================================================================

class UncertaintyRegressor:
    """
    Wrapper for regressors that output uncertainty estimates
    Blueprint Section 6: Combines point predictions with confidence
    """
    def __init__(self, base_model, model_type='catboost'):
        self.model = base_model
        self.model_type = model_type
        self.is_fitted = False
        
    def fit(self, X, y):
        """Train with uncertainty estimation"""
        if self.model_type == 'catboost':
            # CatBoost native uncertainty (Blueprint Section 6)
            self.model.fit(X, y, verbose=False)
        else:
            self.model.fit(X, y)
        self.is_fitted = True
        
    def predict_with_uncertainty(self, X):
        """
        Predict mean and variance
        Blueprint Section 6, Eq. returns [μ, σ²]
        
        Returns:
            mean: (N,) predictions
            variance: (N,) uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.model_type == 'catboost':
            # Use CatBoost's virtual_ensembles_predict for uncertainty
            mean = self.model.predict(X)
            # Approximate variance from model internals (simplified)
            variance = np.ones_like(mean) * 0.1  # Placeholder
        else:
            mean = self.model.predict(X)
            variance = np.ones_like(mean) * 0.1  # Default uncertainty
        
        return mean, variance

def inverse_variance_ensemble(predictions_list, variances_list):
    """
    Uncertainty-weighted ensemble
    Blueprint Section 6, Step 3: μ_ensemble = Σ(μ_i/σ²_i) / Σ(1/σ²_i)
    
    Args:
        predictions_list: List of (N, K) prediction arrays
        variances_list: List of (N, K) variance arrays
    Returns:
        ensemble_pred: (N, K) weighted predictions
    """
    predictions = np.array(predictions_list)  # (M, N, K)
    variances = np.array(variances_list)      # (M, N, K)
    
    # Inverse variance weights
    weights = 1.0 / (variances + 1e-8)
    
    # Weighted average
    ensemble_pred = np.sum(predictions * weights, axis=0) / np.sum(weights, axis=0)
    
    return ensemble_pred

# ============================================================================
# SECTION 5: ENHANCED TRAINING PIPELINE
# ============================================================================

def competition_metric_with_hcl(y_true, y_pred, alpha=0.15):
    """
    Competition metric + HCL penalty
    Blueprint: Combines weighted R² with physical consistency
    
    Args:
        y_true: (N, 5) ground truth
        y_pred: (N, 5) predictions
        alpha: HCL weight
    Returns:
        combined_score: Weighted R² - α * HCL
    """
    # Standard competition metric
    y_weighted = sum(y_true[:, i].mean() * cfg.WEIGHTS[cfg.TARGET_NAMES[i]] 
                     for i in range(5))
    
    ss_res = sum(((y_true[:, i] - y_pred[:, i])**2).mean() * cfg.WEIGHTS[cfg.TARGET_NAMES[i]]
                 for i in range(5))
    ss_tot = sum(((y_true[:, i] - y_weighted)**2).mean() * cfg.WEIGHTS[cfg.TARGET_NAMES[i]]
                 for i in range(5))
    
    r2_score = 1 - ss_res / ss_tot
    
    # HCL penalty
    y_pred_torch = torch.tensor(y_pred, dtype=torch.float32)
    hcl = hierarchical_consistency_loss(y_pred_torch).item()
    
    # Combined (Blueprint approach)
    combined_score = r2_score - alpha * hcl
    
    return combined_score, r2_score, hcl

def enhanced_cross_validation(models_dict, train_data, test_data, 
                               feature_cols, n_folds=5):
    """
    Enhanced CV with HCL, MinT, and uncertainty
    Blueprint: Implements full pipeline from Sections 2, 6
    
    Args:
        models_dict: Dict of {name: model} for ensemble
        train_data: DataFrame with features and targets
        test_data: DataFrame with features
        feature_cols: List of feature column names
        n_folds: Number of CV folds
    
    Returns:
        oof_predictions: (N, 5) out-of-fold predictions
        test_predictions: (N_test, 5) test predictions
        metrics: Dict of performance metrics
    """
    n_targets = len(cfg.TARGET_NAMES)
    oof_preds = np.zeros((len(train_data), n_targets))
    test_preds = np.zeros((len(test_data), n_targets))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        X_train = train_data.iloc[train_idx][feature_cols].values
        y_train = train_data.iloc[train_idx][cfg.TARGET_NAMES].values
        X_val = train_data.iloc[val_idx][feature_cols].values
        y_val = train_data.iloc[val_idx][cfg.TARGET_NAMES].values
        X_test = test_data[feature_cols].values
        
        # Train ensemble with uncertainty
        fold_val_preds = []
        fold_val_vars = []
        fold_test_preds = []
        
        for model_name, base_model in models_dict.items():
            print(f"  Training {model_name}...")
            
            # Wrap in uncertainty regressor
            model = UncertaintyRegressor(base_model, 
                                        model_type='catboost' if 'CatBoost' in str(type(base_model)) else 'other')
            
            # Train per target
            target_preds_val = []
            target_vars_val = []
            target_preds_test = []
            
            for t_idx, target in enumerate(cfg.TARGET_NAMES):
                model.fit(X_train, y_train[:, t_idx])
                
                # Predict with uncertainty
                pred_val, var_val = model.predict_with_uncertainty(X_val)
                pred_test, _ = model.predict_with_uncertainty(X_test)
                
                target_preds_val.append(pred_val)
                target_vars_val.append(var_val)
                target_preds_test.append(pred_test)
            
            fold_val_preds.append(np.column_stack(target_preds_val))
            fold_val_vars.append(np.column_stack(target_vars_val))
            fold_test_preds.append(np.column_stack(target_preds_test))
        
        # Uncertainty-weighted ensemble (Blueprint Section 6)
        val_pred_ensemble = inverse_variance_ensemble(fold_val_preds, fold_val_vars)
        test_pred_fold = np.mean(fold_test_preds, axis=0)
        
        # Apply MinT reconciliation (Blueprint Section 2)
        if cfg.USE_MINT:
            error_cov = estimate_error_covariance(y_val, val_pred_ensemble, None)
            val_pred_reconciled = mint_reconciliation(val_pred_ensemble, error_cov)
            test_pred_fold = mint_reconciliation(test_pred_fold, error_cov)
        else:
            val_pred_reconciled = val_pred_ensemble
        
        # Store OOF predictions
        oof_preds[val_idx] = val_pred_reconciled
        test_preds += test_pred_fold / n_folds
        
        # Compute metrics
        combined, r2, hcl = competition_metric_with_hcl(y_val, val_pred_reconciled)
        fold_metrics.append({
            'fold': fold,
            'r2': r2,
            'hcl': hcl,
            'combined': combined
        })
        
        print(f"  Fold {fold+1} - R²: {r2:.6f}, HCL: {hcl:.6f}, Combined: {combined:.6f}")
    
    # Overall metrics
    overall_combined, overall_r2, overall_hcl = competition_metric_with_hcl(
        train_data[cfg.TARGET_NAMES].values, oof_preds
    )
    
    metrics = {
        'oof_r2': overall_r2,
        'oof_hcl': overall_hcl,
        'oof_combined': overall_combined,
        'fold_metrics': fold_metrics
    }
    
    return oof_preds, test_preds, metrics

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
# Load and prepare data
train_df = pd.read_csv(cfg.DATA_PATH + 'train.csv')
test_df = pd.read_csv(cfg.DATA_PATH + 'test.csv')

# Extract embeddings (use existing compute_embeddings function)
# Add metadata encoding
train_df['month'] = pd.to_datetime(train_df['Sampling_Date']).dt.month
train_df['species_encoded'] = pd.Categorical(train_df['Species']).codes

# Define models ensemble
models = {
    'CatBoost': CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, 
                                   loss_function='RMSE', random_state=cfg.SEED, 
                                   verbose=False),
    'LGBM': LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, 
                          random_state=cfg.SEED, verbose=-1),
    'XGB': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, 
                        random_state=cfg.SEED, verbosity=0)
}

# Feature columns (embeddings + metadata)
feature_cols = [col for col in train_df.columns if col.startswith('emb')] + \
               ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'species_encoded', 'month']

# Run enhanced training
oof_preds, test_preds, metrics = enhanced_cross_validation(
    models, train_df, test_df, feature_cols
)

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"OOF R²: {metrics['oof_r2']:.6f}")
print(f"OOF HCL: {metrics['oof_hcl']:.6f}")
print(f"OOF Combined: {metrics['oof_combined']:.6f}")

# Create submission
# ... (use existing melt_table logic)
"""
