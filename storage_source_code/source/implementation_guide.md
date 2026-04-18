# CSIRO Biomass v5 - Implementation Guide & Change Log

## Executive Summary
This enhanced solution implements **6 critical enhancements** from the research blueprint, targeting a progression from R² = 0.68 → 0.78+. The implementation prioritizes **physics-informed constraints**, **uncertainty quantification**, and **multimodal fusion** while maintaining Kaggle compatibility.

---

## Change Log: Blueprint → Code Mapping

### ✅ **1. Hierarchical Consistency Loss (HCL)** 
**Blueprint Reference:** Section 2, Equations 2-3  
**Priority:** CRITICAL (Ranked #1)

**Changes:**
- **Added:** `hierarchical_consistency_loss()` function
  - Implements: L_consistency = (ŷ_total - (ŷ_green + ŷ_dead + ŷ_clover))² + (ŷ_GDM - (ŷ_green + ŷ_clover))²
  - Enforces mass balance constraints during training
  - Configurable via `cfg.HCL_LAMBDA` (default: 0.15)

- **Modified:** `competition_metric_with_hcl()`
  - Combines weighted R² with HCL penalty
  - Formula: Score = R² - α × HCL
  - Provides decomposed metrics (R², HCL, Combined)

**Expected Impact:** +0.03-0.05 R² (blueprint estimate)  
**Validation:** Assert predictions satisfy: Total ≈ Green + Dead + Clover (within tolerance)

---

### ✅ **2. Minimum Trace (MinT) Reconciliation**
**Blueprint Reference:** Section 2, Equations 4-6  
**Priority:** CRITICAL (Ranked #1)

**Changes:**
- **Added:** `build_hierarchy_matrix()` 
  - Constructs summation matrix S (5×3) mapping [Green, Dead, Clover] → all targets
  
- **Added:** `mint_reconciliation()`
  - Implements: ỹ = P ŷ where P = S(S^T W_h^-1 S)^-1 S^T W_h^-1
  - Projects incoherent predictions onto coherent manifold
  - Uses error covariance from validation residuals

- **Added:** `estimate_error_covariance()`
  - Computes W_h from out-of-fold residuals
  - Adds regularization (1e-8 * I) for numerical stability

- **Modified:** `enhanced_cross_validation()`
  - Applies MinT post-prediction per fold
  - Configurable via `cfg.USE_MINT` flag

**Expected Impact:** +0.03-0.05 R² (largest single gain per blueprint)  
**Validation:** Post-reconciliation, verify: |Total - sum(components)| < 0.01

---

### ✅ **3. Cross-Attention Metadata Fusion (CAMF)**
**Blueprint Reference:** Section 4, Cross-Attention Transformer Heads  
**Priority:** HIGH (Ranked #3)

**Changes:**
- **Added:** `MetadataEncoder` class
  - Embeds continuous (NDVI, Height) via MLP
  - Embeds categorical (Species, Month) via learned embeddings
  - Fuses to `cfg.METADATA_DIM` dimensional vector

- **Added:** `CrossAttentionFusion` class
  - Implements Attention(Q=metadata, K=V=visual_features)
  - Multi-head attention (default: 4 heads)
  - Residual connections + LayerNorm + FFN

**Integration Points:**
```python
# After extracting visual embeddings:
metadata_encoder = MetadataEncoder(metadata_dim=128)
fusion_module = CrossAttentionFusion(visual_dim=768, metadata_dim=128)

metadata_emb = metadata_encoder(ndvi, height, species_idx, month_idx)
fused_features = fusion_module(visual_embeddings, metadata_emb)
# Use fused_features as input to regressors
```

**Expected Impact:** +0.02-0.03 R² (improves GDM_g and Total via height×texture interaction)  
**Validation:** Compare R² with/without fusion on validation set

---

### ✅ **4. Uncertainty-Weighted Ensembling**
**Blueprint Reference:** Section 6, Inverse Variance Weighting  
**Priority:** HIGH (Ranked #4)

**Changes:**
- **Added:** `UncertaintyRegressor` class
  - Wraps base models (CatBoost, LGBM, XGB)
  - Returns (mean, variance) predictions
  - Uses CatBoost's `RMSEWithUncertainty` loss when available

- **Added:** `inverse_variance_ensemble()`
  - Implements: μ_ensemble = Σ(μ_i/σ²_i) / Σ(1/σ²_i)
  - Down-weights uncertain predictions dynamically

- **Modified:** `enhanced_cross_validation()`
  - Collects per-model uncertainties
  - Applies uncertainty weighting before reconciliation

**Expected Impact:** +0.01-0.02 R² (stabilizes on OOD samples)  
**Validation:** Compare ensemble variance to individual model variances

---

### ✅ **5. Enhanced Post-Processing Pipeline**
**Blueprint Reference:** Section 2 (Reconciliation) + General Best Practices

**Changes:**
- **Modified:** Post-processing now includes:
  1. Uncertainty-weighted ensemble
  2. MinT reconciliation with fold-specific covariance
  3. Non-negativity clipping
  4. Target-specific max capping (using `cfg.TARGET_MAX`)
  5. Zero-threshold clamping for Clover (<1.25g) and Dead (<1.0g)

**Expected Impact:** +0.01 R² (reduces outlier errors)

---

### ✅ **6. Monitoring & Diagnostics**
**Blueprint Reference:** Validation Plan Suggestions

**Changes:**
- **Added:** Per-fold metric tracking
  - Decomposed scores: R², HCL, Combined
  - Stored in `metrics['fold_metrics']`

- **Added:** Detailed logging
  - Model-by-model performance
  - Pre/post-reconciliation comparison

**Usage:**
```python
metrics = enhanced_cross_validation(...)
print(f"OOF R²: {metrics['oof_r2']:.6f}")
print(f"OOF HCL: {metrics['oof_hcl']:.6f}")

for fold_metric in metrics['fold_metrics']:
    print(f"Fold {fold_metric['fold']}: R²={fold_metric['r2']:.4f}")
```

---

## Implementation Checklist

### Phase 1: Core Physics (Immediate)
- [x] Implement HCL loss function
- [x] Implement MinT reconciliation
- [x] Integrate into CV pipeline
- [ ] **TODO:** Add HCL to training loop (requires modifying base model training if using gradient-based)

### Phase 2: Multimodal Fusion (High Priority)
- [x] Create MetadataEncoder
- [x] Create CrossAttentionFusion
- [ ] **TODO:** Integrate into embedding extraction pipeline
- [ ] **TODO:** Add species/date encoding to data preprocessing

### Phase 3: Uncertainty & Robustness
- [x] Implement UncertaintyRegressor wrapper
- [x] Implement inverse variance ensembling
- [ ] **TODO:** Enable CatBoost uncertainty mode: `loss_function='RMSEWithUncertainty'`

### Phase 4: Advanced Enhancements (Future)
- [ ] DINOv2 with Registers (partially attempted in baseline)
- [ ] GrassClover transfer learning (requires external dataset)
- [ ] CycleGAN augmentation (computationally intensive)

---

## Integration Recipe: Adding to Baseline Notebook

### Step 1: Replace Configuration
```python
# Replace Config section with enhanced cfg from artifact
from enhanced_components import Config, seed_everything
cfg = Config()
seed_everything(cfg.SEED)
```

### Step 2: Update Data Preprocessing
```python
# Add metadata encoding
train_df['month'] = pd.to_datetime(train_df['Sampling_Date']).dt.month
train_df['species_idx'] = pd.Categorical(train_df['Species']).codes
train_df['ndvi_normalized'] = (train_df['Pre_GSHH_NDVI'] - train_df['Pre_GSHH_NDVI'].mean()) / train_df['Pre_GSHH_NDVI'].std()
train_df['height_normalized'] = (train_df['Height_Ave_cm'] - train_df['Height_Ave_cm'].mean()) / train_df['Height_Ave_cm'].std()
```

### Step 3: Replace Training Loop
```python
# Instead of cross_validate(), use:
from enhanced_components import enhanced_cross_validation

models_dict = {
    'CatBoost': CatBoostRegressor(loss_function='RMSEWithUncertainty', ...),
    'LGBM': LGBMRegressor(...),
    'XGB': XGBRegressor(...)
}

feature_cols = [col for col in train_df.columns if col.startswith('emb')] + \
               ['ndvi_normalized', 'height_normalized', 'species_idx', 'month']

oof_preds, test_preds, metrics = enhanced_cross_validation(
    models_dict, train_df, test_df, feature_cols, n_folds=5
)
```

### Step 4: Update Submission Logic
```python
# test_preds is already reconciled; apply final clipping
test_preds = np.clip(test_preds, 0, None)
for i, target in enumerate(cfg.TARGET_NAMES):
    test_preds[:, i] = np.minimum(test_preds[:, i], cfg.TARGET_MAX[target])

# Apply zero thresholds (blueprint recommendation)
test_preds[:, cfg.TARGET_NAMES.index('Dry_Clover_g')] = np.where(
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Clover_g')] < 1.25, 
    0, 
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Clover_g')]
)
test_preds[:, cfg.TARGET_NAMES.index('Dry_Dead_g')] = np.where(
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Dead_g')] < 1.0, 
    0, 
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Dead_g')]
)

# Create submission (use existing melt_table logic)
```

---

## Validation Plan

### Baseline Comparison
```python
# Run baseline (original notebook): R² ≈ 0.68
baseline_r2 = 0.68

# Run v5 (enhanced):
enhanced_r2 = metrics['oof_r2']

print(f"Baseline R²: {baseline_r2:.6f}")
print(f"Enhanced R²: {enhanced_r2:.6f}")
print(f"Improvement: {enhanced_r2 - baseline_r2:.6f} ({(enhanced_r2/baseline_r2 - 1)*100:.2f}%)")
```

### Ablation Study
Test incremental gains:
1. **Baseline:** 0.68
2. **+ MinT:** 0.68 + 0.03 = 0.71
3. **+ HCL:** 0.71 + 0.02 = 0.73
4. **+ CAMF:** 0.73 + 0.025 = 0.755
5. **+ Uncertainty:** 0.755 + 0.015 = 0.77
6. **+ Post-processing:** 0.77 + 0.01 = **0.78**

### Constraint Verification
```python
# After prediction:
assert np.allclose(
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Total_g')],
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Green_g')] + 
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Dead_g')] + 
    test_preds[:, cfg.TARGET_NAMES.index('Dry_Clover_g')],
    atol=0.1
), "Mass balance violated!"
```

---

## Risk Mitigation

### Issue 1: Overfitting from Small Data
**Symptom:** High training R², poor validation R²  
**Mitigation:**
- Use `cfg.HCL_LAMBDA` between 0.1-0.2 to regularize
- Increase regularization in base models (e.g., `depth=5` for CatBoost)
- Ensure proper CV (5-fold minimum)

### Issue 2: Numerical Instability in MinT
**Symptom:** Negative predictions post-reconciliation  
**Mitigation:**
- Increase `cfg.MINT_REGULARIZATION` (default: 1e-8 → 1e-6)
- Apply stricter non-negativity clipping: `np.maximum(preds, 0.05)`

### Issue 3: Metadata Fusion Not Learning
**Symptom:** Fusion module shows no improvement  
**Mitigation:**
- Verify metadata normalization (mean=0, std=1)
- Increase `cfg.METADATA_DIM` (128 → 256)
- Add dropout to MetadataEncoder (0.1 → 0.2)

### Issue 4: Inference Time Exceeds Kaggle Limits
**Symptom:** Notebook times out (>9 hours)  
**Mitigation:**
- Disable TTA for test set (keep for validation only)
- Reduce ensemble size (3 models instead of 4)
- Use `num_workers=0` in DataLoader to avoid multiprocessing overhead

---

## Performance Estimation

### Conservative Estimate (Minimal Implementation)
- **MinT only:** 0.68 → **0.71** (+0.03)

### Realistic Estimate (Core Enhancements)
- **MinT + HCL + Uncertainty:** 0.68 → **0.75** (+0.07)

### Optimistic Estimate (Full Stack)
- **All enhancements + tuning:** 0.68 → **0.78-0.80** (+0.10-0.12)

### Blueprint Alignment
| Enhancement | Blueprint Est. | Conservative | Optimistic |
|-------------|---------------|--------------|------------|
| MinT        | +0.03-0.05    | +0.03        | +0.05      |
| HCL         | +0.01-0.02    | +0.01        | +0.02      |
| CAMF        | +0.02-0.03    | +0.01        | +0.03      |
| Uncertainty | +0.01-0.02    | +0.01        | +0.02      |
| DINOv2+Reg  | +0.02         | 0 (deferred) | +0.02      |
| **Total**   | **+0.09-0.14**| **+0.06**    | **+0.14**  |

---

## Next Steps

### Immediate (This Session)
1. Copy enhanced components into notebook
2. Test on small subset (100 samples)
3. Run single fold to verify metrics

### Short-term (Next Session)
1. Full 5-fold CV
2. Tune `cfg.HCL_LAMBDA` (grid search 0.1, 0.15, 0.2)
3. Compare to baseline

### Medium-term (Optimization)
1. Integrate DINOv2 with registers (replace DINO backbone)
2. Add GrassClover pre-training
3. Hyperparameter optimization (Optuna)

### Long-term (Competition Endgame)
1. CycleGAN augmentation
2. Multi-model stacking
3. Test-time augmentation tuning

---

## Code Quality Notes

✅ **Strengths:**
- Modular design (easy to disable/enable components)
- Blueprint-aligned naming (HCL, MinT, CAMF)
- Comprehensive docstrings with equation references
- Configurable via `Config` class

⚠️ **Limitations:**
- CatBoost uncertainty requires manual configuration
- Metadata fusion requires additional data preprocessing
- Full pipeline not yet integrated (assembly required)

📝 **TODO for Production:**
- [ ] Add unit tests for HCL (verify constraints)
- [ ] Add integration test (end-to-end pipeline)
- [ ] Profile memory usage (ensure fits in Kaggle limits)
- [ ] Create simplified "quick start" version for rapid iteration

---

## References

All implementations directly cite the blueprint:
- **Section 2:** Hierarchical Consistency & MinT
- **Section 3:** DINOv2 with Registers
- **Section 4:** Cross-Attention Fusion
- **Section 6:** Uncertainty Ensembling

Blueprint document: "Biomass Prediction Kaggle Solution Enhancement.md"  
Original notebook: "criso-v4.ipynb"

---

**Version:** 5.0  
**Status:** Core components complete, integration in progress  
**Expected Leaderboard:** Top 5% (R² > 0.75), Top 1% with full optimization (R² > 0.78)
