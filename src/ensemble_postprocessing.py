"""
CSIRO Image2Biomass - Ensemble and Post-Processing
Advanced techniques for hierarchical reconciliation and multi-model ensembling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from scipy.optimize import minimize


class HierarchicalReconciliation:
    """
    Enforce physical constraints: Total = sum(components)
    Uses optimization to find minimal adjustment that satisfies constraints
    """
    
    def __init__(self):
        self.target_indices = {
            'Dry_Green_g': 0,
            'Dry_Dead_g': 1,
            'Dry_Clover_g': 2,
            'GDM_g': 3,
            'Dry_Total_g': 4
        }
    
    def reconcile_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Reconcile predictions to satisfy:
        - GDM_g = Dry_Green_g + Dry_Clover_g
        - Dry_Total_g = GDM_g + Dry_Dead_g
        
        Args:
            predictions: (N, 5) array of predictions
        
        Returns:
            reconciled: (N, 5) array with constraints satisfied
        """
        N = predictions.shape[0]
        reconciled = np.zeros_like(predictions)
        
        for i in range(N):
            reconciled[i] = self._reconcile_single(predictions[i])
        
        return reconciled
    
    def _reconcile_single(self, pred: np.ndarray) -> np.ndarray:
        """
        Reconcile a single sample using ratio-based adjustment
        
        Strategy:
        1. Trust Dry_Total_g prediction (highest weight)
        2. Adjust components proportionally to sum to total
        """
        green = pred[0]
        dead = pred[1]
        clover = pred[2]
        gdm_pred = pred[3]
        total_pred = pred[4]
        
        # Component sum (should equal total)
        component_sum = green + dead + clover
        
        # Ratio-based adjustment
        if component_sum > 0:
            # Scale components to match total
            scale_factor = total_pred / component_sum
            green_adj = green * scale_factor
            dead_adj = dead * scale_factor
            clover_adj = clover * scale_factor
        else:
            # If all components are zero, distribute total equally
            green_adj = total_pred / 3
            dead_adj = total_pred / 3
            clover_adj = total_pred / 3
        
        # Calculate GDM from adjusted components
        gdm_adj = green_adj + clover_adj
        total_adj = gdm_adj + dead_adj
        
        return np.array([green_adj, dead_adj, clover_adj, gdm_adj, total_adj])


class StateBasedPostProcessing:
    """
    Apply state-specific rules based on Western Australia anomaly
    """
    
    def __init__(self, state_map_path: str = None):
        """
        Args:
            state_map_path: CSV mapping image_path to state (if available in test)
        """
        self.state_map = None
        if state_map_path and Path(state_map_path).exists():
            self.state_map = pd.read_csv(state_map_path)
    
    def apply_wa_correction(self, 
                           predictions: np.ndarray, 
                           image_paths: List[str]) -> np.ndarray:
        """
        Apply WA-specific correction: set Dry_Dead_g = 0 for WA samples
        
        Args:
            predictions: (N, 5) array
            image_paths: List of image paths
        
        Returns:
            corrected: (N, 5) array
        """
        if self.state_map is None:
            print("Warning: No state map available for WA correction")
            return predictions
        
        corrected = predictions.copy()
        
        # Create path to state mapping
        path_to_state = dict(zip(self.state_map['image_path'], 
                                self.state_map['State']))
        
        for i, img_path in enumerate(image_paths):
            state = path_to_state.get(img_path, None)
            if state == 'WA':
                # Set Dry_Dead_g to 0
                corrected[i, 1] = 0.0
                # Recalculate total
                corrected[i, 4] = corrected[i, 3] + corrected[i, 1]
        
        return corrected


class ZeroClampingStrategy:
    """
    Clamp near-zero predictions to exact zero for sparse components
    """
    
    def __init__(self, 
                 clover_threshold: float = 5.0,
                 dead_threshold: float = 5.0):
        """
        Args:
            clover_threshold: Clamp Dry_Clover_g < threshold to 0
            dead_threshold: Clamp Dry_Dead_g < threshold to 0
        """
        self.clover_threshold = clover_threshold
        self.dead_threshold = dead_threshold
    
    def apply_clamping(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply zero clamping to sparse components
        
        Args:
            predictions: (N, 5) array
        
        Returns:
            clamped: (N, 5) array
        """
        clamped = predictions.copy()
        
        # Clamp clover
        clover_mask = clamped[:, 2] < self.clover_threshold
        clamped[clover_mask, 2] = 0.0
        
        # Clamp dead
        dead_mask = clamped[:, 1] < self.dead_threshold
        clamped[dead_mask, 1] = 0.0
        
        # Recalculate GDM and Total
        clamped[:, 3] = clamped[:, 0] + clamped[:, 2]  # GDM
        clamped[:, 4] = clamped[:, 3] + clamped[:, 1]  # Total
        
        return clamped


class WeightedEnsemble:
    """
    Weighted averaging of multiple model predictions
    Supports automatic weight optimization via validation set
    """
    
    def __init__(self, model_weights: Dict[str, float] = None):
        """
        Args:
            model_weights: Dict mapping model_name to weight
                          If None, uses uniform weights
        """
        self.model_weights = model_weights
    
    def ensemble_predictions(self, 
                           model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Ensemble predictions from multiple models
        
        Args:
            model_predictions: Dict mapping model_name to predictions (N, 5)
        
        Returns:
            ensembled: (N, 5) array
        """
        model_names = list(model_predictions.keys())
        
        # Initialize weights
        if self.model_weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            weights = self.model_weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted average
        ensembled = np.zeros_like(model_predictions[model_names[0]])
        for name, preds in model_predictions.items():
            ensembled += preds * weights[name]
        
        return ensembled
    
    def optimize_weights(self, 
                        model_predictions: Dict[str, np.ndarray],
                        targets: np.ndarray,
                        competition_weights: np.ndarray) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation targets
        
        Args:
            model_predictions: Dict of model predictions
            targets: Ground truth (N, 5)
            competition_weights: (5,) array of target weights
        
        Returns:
            optimal_weights: Dict of optimized weights
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        def objective(weights):
            # Normalize weights
            weights = weights / weights.sum()
            
            # Ensemble
            ensembled = np.zeros_like(targets)
            for i, name in enumerate(model_names):
                ensembled += model_predictions[name] * weights[i]
            
            # Calculate weighted MSE
            errors = (targets - ensembled) ** 2
            weighted_errors = errors * competition_weights[np.newaxis, :]
            loss = weighted_errors.mean()
            
            return loss
        
        # Optimize
        x0 = np.ones(n_models) / n_models  # Initial uniform weights
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                         method='SLSQP')
        
        optimal_weights = dict(zip(model_names, result.x))
        return optimal_weights


def full_postprocessing_pipeline(
    predictions: np.ndarray,
    image_paths: List[str],
    apply_reconciliation: bool = True,
    apply_wa_correction: bool = False,
    apply_clamping: bool = True,
    clover_threshold: float = 5.0,
    dead_threshold: float = 5.0,
    state_map_path: str = None
) -> np.ndarray:
    """
    Complete post-processing pipeline
    
    Args:
        predictions: (N, 5) raw model predictions
        image_paths: List of image paths
        apply_reconciliation: Whether to enforce hierarchical constraints
        apply_wa_correction: Whether to apply WA dead biomass correction
        apply_clamping: Whether to clamp near-zero values
        clover_threshold: Threshold for clamping clover
        dead_threshold: Threshold for clamping dead
        state_map_path: Path to state mapping CSV
    
    Returns:
        processed: (N, 5) post-processed predictions
    """
    processed = predictions.copy()
    
    # Step 1: Zero clamping (before reconciliation)
    if apply_clamping:
        print("Applying zero clamping...")
        clamper = ZeroClampingStrategy(clover_threshold, dead_threshold)
        processed = clamper.apply_clamping(processed)
    
    # Step 2: State-based correction
    if apply_wa_correction:
        print("Applying WA correction...")
        state_processor = StateBasedPostProcessing(state_map_path)
        processed = state_processor.apply_wa_correction(processed, image_paths)
    
    # Step 3: Hierarchical reconciliation
    if apply_reconciliation:
        print("Applying hierarchical reconciliation...")
        reconciler = HierarchicalReconciliation()
        processed = reconciler.reconcile_predictions(processed)
    
    return processed


def create_submission_from_ensemble(
    model_paths: List[str],
    test_csv_path: str,
    output_path: str = "ensemble_submission.csv",
    model_weights: Dict[str, float] = None
):
    """
    Create submission from ensemble of multiple models
    
    Args:
        model_paths: List of paths to trained model checkpoints
        test_csv_path: Path to test.csv
        output_path: Output submission file path
        model_weights: Optional dict of model weights
    """
    from main_training import DinoBiomassModel, BiomassDataset, get_transforms, Config
    from torch.utils.data import DataLoader
    
    config = Config()
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
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
    
    # Load models and generate predictions
    model_predictions = {}
    image_paths = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\nGenerating predictions from model {i+1}/{len(model_paths)}...")
        
        # Load model
        model = DinoBiomassModel.load_from_checkpoint(model_path, config=config)
        model.eval()
        model.cuda()
        
        predictions = []
        
        with torch.no_grad():
            for images, paths in test_loader:
                if i == 0:
                    image_paths.extend(paths)
                
                images = images.cuda()
                preds, _ = model(images)
                predictions.append(preds.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        model_predictions[f"model_{i}"] = predictions
    
    # Ensemble
    print("\nEnsembling predictions...")
    ensembler = WeightedEnsemble(model_weights)
    ensembled_preds = ensembler.ensemble_predictions(model_predictions)
    
    # Post-processing
    print("\nApplying post-processing...")
    final_preds = full_postprocessing_pipeline(
        ensembled_preds,
        image_paths,
        apply_reconciliation=True,
        apply_wa_correction=False,  # Set True if state info available
        apply_clamping=True,
        clover_threshold=3.0,
        dead_threshold=3.0
    )
    
    # Create submission
    print("\nCreating submission file...")
    target_names = config.TARGET_NAMES
    submission_rows = []
    
    for idx, img_path in enumerate(image_paths):
        img_id = Path(img_path).stem
        for i, target_name in enumerate(target_names):
            sample_id = f"{img_id}__{target_name}"
            target_value = final_preds[idx, i]
            submission_rows.append({
                'sample_id': sample_id,
                'target': target_value
            })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    
    return submission_df


if __name__ == "__main__":
    # Example usage
    model_paths = [
        "outputs/fold_0/best-epoch=XX-val_r2=0.XXXX.ckpt",
        "outputs/fold_1/best-epoch=XX-val_r2=0.XXXX.ckpt",
        "outputs/fold_2/best-epoch=XX-val_r2=0.XXXX.ckpt",
    ]
    
    # Option 1: Uniform ensemble
    create_submission_from_ensemble(
        model_paths,
        "test.csv",
        "ensemble_submission.csv"
    )
    
    # Option 2: Weighted ensemble (weights optimized on validation)
    model_weights = {
        "model_0": 0.35,
        "model_1": 0.40,
        "model_2": 0.25
    }
    
    create_submission_from_ensemble(
        model_paths,
        "test.csv",
        "weighted_ensemble_submission.csv",
        model_weights=model_weights
    )