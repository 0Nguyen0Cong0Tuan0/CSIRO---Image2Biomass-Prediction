# **Advanced Biomass Estimation via DINOv2 Adaptation, Tweedie Regression, and Hierarchical Constraints: A Comprehensive Research Report**

## **1\. Executive Summary**

The transition from qualitative pastoral observation to quantitative biomass estimation represents a pivotal shift in modern precision agriculture. Accurate, scalable prediction of pasture biomass—specifically disjoint components such as Green Dry Matter (GDM) and Senescent (Dead) material—enables data-driven grazing management, optimized stocking rates, and precise yield forecasting. However, the computational realization of this goal is fraught with complexities that standard computer vision methodologies often fail to address. The "Biomass Prediction Research Plan" analyzed in this report targets these specific failures, proposing a sophisticated synthesis of self-supervised learning, probabilistic regression, and physically constrained neural architectures.

This report serves as an exhaustive technical blueprint for implementing a State-of-the-Art (SOTA) biomass prediction pipeline, specifically tailored to the nuances of the CSIRO Image2Biomass challenge. The analysis proceeds from a foundational critique of the dataset's statistical anomalies—most notably zero-inflation and heteroscedasticity—to the architectural implementation of a **DINOv2-based regressor**. Unlike traditional Convolutional Neural Networks (CNNs) trained on ImageNet, DINOv2’s self-supervised Vision Transformer (ViT) backbone offers a distinct advantage in capturing the high-frequency textural features required to distinguish specific vegetation species (e.g., Clover vs. Ryegrass) independent of semantic object labels.1

Furthermore, this report provides the mathematical and practical justification for abandoning the standard Mean Squared Error (MSE) loss function in favor of the **Tweedie distribution**.3 The biomass data, characterized by a compound Poisson-Gamma distribution (where exact zeros are common, but positive values are continuous and skewed), requires a loss landscape that explicitly models the probability mass at zero. This is coupled with a **Hierarchical Constraint** mechanism, architected directly into the neural network's output heads, to ensure that the predicted components (Green, Dead, etc.) physically sum to the predicted Total biomass—a constraint often violated by independent regression heads.5

Finally, the report details a novel application of **Knowledge Distillation (KD)** under the Learning Using Privileged Information (LUPI) framework.7 By utilizing a "Teacher" model trained on rich tabular metadata (seasonality, location, weather) to guide a "Student" image model, we embed environmental priors into the visual feature extraction process. The following sections dismantle these components into rigorous theoretical discussions and production-ready PyTorch implementations, offering a complete roadmap for execution.

## ---

**2\. Problem Space Analysis: The Agronomic and Statistical Challenge**

To engineer an effective solution, one must first rigorously define the problem space. The task of predicting pasture biomass from aerial imagery is not merely a regression problem; it is a task of interpreting high-variance biological signals from unstructured visual data, constrained by the realities of data collection.

### **2.1 The "Clip and Weigh" Ground Truth and Data Noise**

The ground truth for such datasets is typically derived from the "clip and weigh" method: a quadrant of pasture is physically cut, sorted by species, dried, and weighed.8 This process, while the gold standard, introduces inescapable noise. Human error in sorting species, variations in drying times, and the precise alignment of the physical quadrant with the captured image create a "noisy label" regime.

Crucially, the relationship between the visual appearance of pasture and its mass is non-linear and species-dependent. A dense mat of clover may weigh significantly more than an equally tall stand of ryegrass due to water content and structural density.9 Therefore, models relying solely on "height" proxies (like simple photogrammetry) often fail. The model must learn **density**, a feature inferred from subtle textural cues—shadowing between leaves, leaf width, and color gradients—that standard encoders like ResNet-50, trained on object-centric ImageNet classes, struggle to prioritize.10

### **2.2 Statistical Anomalies: Zero-Inflation and Heteroscedasticity**

The distribution of biomass targets—*Dry\_Green\_g*, *Dry\_Dead\_g*, *Dry\_Clover\_g*, *GDM\_g*, and *Dry\_Total\_g*—exhibits properties that violate the assumptions of standard Gaussian regression.

Zero-Inflation:  
In many samples, specific components are entirely absent. For example, a lush, irrigated paddock in spring may have exactly 0.0g of Dry\_Dead\_g biomass. Similarly, a pure ryegrass field will have 0.0g of Dry\_Clover\_g.9 This is not a "small number" or a "missing value"; it is an exact zero. Standard regression models trained with MSE loss tend to "hedge their bets" in these scenarios, predicting small positive values (e.g., 2.5g) to minimize average error, thereby failing to capture the binary presence/absence nature of the biological reality.  
Heteroscedasticity:  
The variance of biomass measurements increases with the mean. A prediction error of ±50g is negligible in a paddock with 2000g of biomass but catastrophic in a paddock with 100g. This scale-dependent variance means that simple $L2$ losses (MSE) are dominated by high-biomass outliers, causing the model to neglect the fine-grained accuracy required for low-biomass samples.4

### **2.3 The "Dead Biomass" Conundrum**

Exploratory data analysis (EDA) of the CSIRO dataset reveals that *Dry\_Dead\_g* is consistently the most difficult target to predict, showing the weakest correlation with simple metadata features like height.6

* **Visual Ambiguity:** Dead biomass often sits at the base of the sward, occluded by green canopy (the "green-on-brown" problem).  
* **Spectral Confusion:** Dry dead grass can be spectrally similar to bare soil, confounding simple RGB analysis.  
* **Ratio Dependency:** Strong correlations exist between *Dead* biomass and the *Total* biomass ratio.6 Dead matter is rarely an independent variable; it is a function of the total growth cycle and seasonal decay.

Implication for Research Plan:  
These factors dictate that independent regression of each target is suboptimal. The architecture must explicitly model the dependencies between targets (e.g., Total \= Green \+ Dead) and utilize loss functions capable of handling zero-inflated distributions.

## ---

**3\. Theoretical Framework: DINOv2 and Probabilistic Regression**

This section establishes the theoretical underpinnings of the proposed solution, justifying the selection of DINOv2 and Tweedie Loss over conventional alternatives.

### **3.1 Self-Supervised Learning: The DINOv2 Advantage**

The choice of DINOv2 (Vision Transformer) as the backbone is strategic. Traditional supervised pre-training (e.g., EfficientNet on ImageNet) teaches a model to be invariant to intra-class variance (e.g., ignoring the color of a car to classify it as a "car"). However, in biomass estimation, that "intra-class variance" (texture, color, leaf shape) is the signal itself.1

Mechanism of Action:  
DINOv2 employs a discriminative self-distillation approach. A "Student" network attempts to predict the output of a "Teacher" network. Crucially, the teacher is a momentum-averaged version of the student. The training objective combines:

1. **DINO Loss:** A cross-entropy loss on the class token output, encouraging global semantic understanding.  
2. **iBOT Loss:** A masked image modeling loss applied to patch tokens. The network must reconstruct masked patches using context from visible patches.11

This local-to-global correspondence forces DINOv2 to learn rich, high-frequency textural descriptors. For a pasture image, DINOv2 effectively learns a "vocabulary" of grass textures—dense clover, sparse fescue, senescent stalks—without ever seeing a biomass label. When transferred to the regression task, these features are significantly more robust to illumination changes and "nuisance" variability (like shadows) than features derived from supervised classification.2

### **3.2 The Tweedie Distribution Family**

To address the zero-inflation and heteroscedasticity described in Section 2.2, we turn to the Tweedie distribution. The Tweedie family is a subset of Exponential Dispersion Models (EDM) defined by a variance function $V(\mu) = \phi \mu^p$, where $\mu$ is the mean, $\phi$ is the dispersion, and $p$ is the power parameter.3

The power parameter $p$ dictates the nature of the distribution:

* $p=0$: Normal distribution (Variance is constant).  
* $p=1$: Poisson distribution (Variance equals mean).  
* $p=2$: Gamma distribution (Variance scales with square of mean).  
* **$1 < p < 2$**: **Compound Poisson-Gamma Distribution**.

Why $1 < p < 2$ is the "Sweet Spot":  
In this interval, the Tweedie distribution models the variable as a Poisson sum of Gamma distributed variables.

$$Y = \sum_{i=1}^{N} X_i, \quad N \sim \text{Poisson}(\lambda), \quad X_i \sim \text{Gamma}(\alpha, \beta)$$

* If $N=0$ (no Poisson events), then $Y=0$. This models the **exact zeros** in the data (e.g., no clover present).  
* If $N \> 0$, $Y$ is a sum of continuous positive variables, resulting in a **skewed, continuous positive distribution**.

This mathematically mirrors the biological reality of biomass: "biomass clumps" (Poisson) each having a certain weight (Gamma). By optimizing the Negative Log-Likelihood (NLL) of the Tweedie distribution, the model naturally learns to predict exact zeros without the need for complex two-stage "Hurdle" models (which separate classification of zero vs. non-zero from regression).12

### **3.3 Hierarchical Constraints in Neural Networks**

Neural networks are universal function approximators, but they do not inherently respect physical laws. In the CSIRO dataset, the targets are physically coupled:

$$\text{Dry\_Total\_g} \approx \text{Dry\_Green\_g} + \text{Dry\_Dead\_g}$$

(Note: GDM and Clover are subsets or related metrics, but Total is an aggregate).  
Standard multi-output regression treats these as independent tasks. If a model predicts Green=100, Dead=50, but Total=120, it has violated physical consistency. This inconsistency creates gradient conflict: the "Total" head might be pushing gradients to increase features related to density, while the "Green" head pushes to decrease them.

Architectural Solution:  
We propose a Ratio-Based Constraint. Instead of predicting absolute values for components, the network predicts:

1. A scalar $\hat{Y}_{Total}$.  
2. A vector of proportions $\hat{p} = [p_{green}, p_{dead}, \dots]$ such that $\sum \hat{p} = 1$ (via Softmax).  
3. The final component predictions are derived: $\hat{Y}_{component} = \hat{Y}_{Total} \times p_{component}$.

This guarantees that $\sum \hat{Y}_{component} \equiv \hat{Y}_{Total}$. It also simplifies the learning task: the "Total" head focuses on *density* (how much stuff?), while the "Ratio" head focuses on *composition* (what kind of stuff?).6

## ---

**4\. Data Engineering Pipeline Implementation**

The following section details the implementation of the data pipeline, translating the theoretical requirements into PyTorch and Albumentations code. This pipeline addresses the "Wide vs. Long" format discrepancies and implements aggressive geometric augmentation.

### **4.1 The Biomass Dataset Class**

The dataset class must handle the ingestion of the CSV metadata and images. The CSIRO dataset often presents in a "long" format (multiple rows per image), which we must pivot to "wide" for efficient training.9

```Python
import os  
import pandas as pd  
import numpy as np  
import torch  
from torch.utils.data import Dataset  
from PIL import Image  
import albumentations as A  
from albumentations.pytorch import ToTensorV2

class BiomassDataset(Dataset):  
    def __init__(self, csv_file, img_dir, transform=None, mode='train'):  
        """  
        Args:  
            csv_file (string): Path to the csv file with annotations.  
            img_dir (string): Directory with all the images.  
            transform (callable, optional): Albumentations transform pipeline.  
            mode (string): 'train' or 'test'.  
        """  
        self.df = pd.read_csv(csv_file)  
        self.img_dir = img_dir  
        self.transform = transform  
        self.mode = mode  
          
        # PIVOT STRATEGY:  
        # Convert Long format (one row per target) to Wide (one row per image).  
        # We group by 'image_path' and aggregate targets.  
        if self.mode == 'train':  
            self.targets =  
            # Pivot table to get one row per image with columns for each target  
            self.wide_df = self.df.pivot_table(  
                index=,   
                columns='target_name',   
                values='target'  
            ).reset_index()  
              
            # Metadata normalization (Example: Height)  
            self.wide_df['Height_Ave_cm'] = self.wide_df['Height_Ave_cm'].fillna(0)  
            self.height_mean = self.wide_df['Height_Ave_cm'].mean()  
            self.height_std = self.wide_df['Height_Ave_cm'].std()  
        else:  
            # Test set has no targets, just unique images  
            self.wide_df = self.df.drop_duplicates()

    def __len__(self):  
        return len(self.wide\_df)

    def __getitem__(self, idx):  
        row = self.wide_df.iloc[idx]  
        img_name = os.path.join(self.img_dir, row['image_path'])  
          
        # Load Image  
        try:  
            image = np.array(Image.open(img_name).convert('RGB'))  
        except (IOError, FileNotFoundError):  
            # Fallback for missing images or corruption  
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply Augmentations  
        if self.transform:  
            augmented = self.transform(image=image)  
            image = augmented['image']

        # Process Metadata (Tabular features)  
        # 1. Cyclical Date Encoding  
        date = pd.to_datetime(row)  
        day_of_year = date.dayofyear  
        sin_date = np.sin(2 * np.pi * day_of_year / 365.0)  
        cos_date = np.cos(2 * np.pi * day_of_year / 365.0)  
          
        # 2. State Encoding (Simple mapping for this example)  
        state_map = {'Tas': 0, 'WA': 1, 'NSW': 2, 'Vic': 3, 'SA': 4, 'QLD': 5}  # Example map  
        state_idx = state_map.get(row.get('State', 'Tas'), 0)  # Default to 0  
          
        # 3. Height (Normalized)  
        height = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-6) if self.mode == 'train' else 0.0

        meta_features = torch.tensor([sin_date, cos_date, height], dtype=torch.float32)

        if self.mode == 'train':  
            # Extract Targets in specific order  
            target_vals = row[self.targets].values.astype(np.float32)  
            # Targets:  
            return image, meta_features, torch.tensor(target_vals), torch.tensor(state_idx, dtype=torch.long)  
        else:  
            return image, meta_features, row['image_path']
```
### **4.2 Advanced Augmentation Strategy**

Vegetation imagery presents specific challenges: it is texture-heavy and orientation-invariant (grass has no "up" or "down" semantically), but spectrally sensitive (green vs. yellow is a critical distinction).16

***Geometric Transforms (High Probability)***:  
We employ aggressive geometric transforms. RandomResizedCrop is the most critical augmentation.18 It simulates varying drone altitudes (Ground Sampling Distances) and forces the model to learn density independent of scale. Rotations and flips exploit the inherent symmetry of pasture.  

***Photometric Transforms (Low/Moderate Probability)***:  
We must be cautious with color jitter. Excessive hue shifting can confuse the model (turning green grass yellow mimics dead biomass). We prioritize RandomBrightnessContrast to simulate cloud cover or exposure differences, but keep HueSaturationValue subtle.  

***Regularization***:  
CoarseDropout (similar to Cutout) is used to mask out square regions of the image. This forces the DINOv2 backbone to attend to global context rather than overfitting to specific local features (like a single distinctive weed).17

```Python
def get_transforms(cfg):  
    img_size = cfg.IMG_SIZE  
      
    train_transform = A.Compose()

    valid_transform = A.Compose()  
      
    return train_transform, valid_transform
```
## ---

**5\. Model Architecture: The "DinoBiomass" Regressor**

The core innovation of this plan is the DinoBiomass architecture. It fuses a pre-trained DINOv2 backbone with a hierarchical regression head that enforces physical constraints.

### **5.1 Backbone Integration and LoRA (Low-Rank Adaptation)**

While full fine-tuning of ViT backbones can yield high performance, it is computationally expensive and prone to overfitting on small datasets. Linear probing (freezing the backbone) is often too rigid. We recommend **Low-Rank Adaptation (LoRA)** or a **Layer-Wise Freezing** strategy.19

For this implementation, we will use a **Layer-Wise Freezing** approach, which is simpler to implement in standard PyTorch and highly effective. We will freeze the initial patch embedding and early transformer blocks (layers 0-8 of ViT-Small), allowing only the final blocks (9-11) and the regression head to train. This preserves the generic texture features learned by DINOv2 while adapting the high-level semantic features to the biomass domain.

### **5.2 Ratio-Constrained Head Design**

As discussed in Section 3.3, the head separates the prediction of *Total* biomass (Magnitude) from the *Component Ratios* (Distribution).

* **Total Branch:** Predicts a single scalar value. Uses Softplus activation to ensure positivity.  
* **Ratio Branch:** Predicts $N$ logits. Uses Softmax to ensure they sum to 1\.  
* **Fusion:** Element-wise multiplication of Total and Ratios.

### **5.3 Complete PyTorch Lightning Module**

```Python

import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
import pytorch_lightning as pl  
import timm

# -------------------------------------------------------------------  
# Custom Tweedie Loss Module  
# -------------------------------------------------------------------  
class TweedieLoss(nn.Module):  
    """  
    Tweedie Loss for Zero-Inflated continuous data (1 \< p \< 2).  
    Minimizing Negative Log-Likelihood.  
    """  
    def __init__(self, p=1.5, reduction='mean', epsilon=1e-8):  
        super().__init__()  
        assert 1 < p < 2, "Tweedie power p must be in (1, 2)"  
        self.p = p  
        self.reduction = reduction  
        self.epsilon = epsilon

    def forward(self, pred, target):  
        # Pred must be strictly positive.   
        # We assume pred comes from Softplus or Exp, but add epsilon for safety.  
        pred = pred + self.epsilon  
          
        # Calculation of Unit Deviance for Tweedie  
        # term1 = -y * pred^(1-p) / (1-p)  
        # term2 = + pred^(2-p) / (2-p)  
          
        term1 = -target * torch.pow(pred, 1 - self.p) / (1 - self.p)  
        term2 = torch.pow(pred, 2 - self.p) / (2 - self.p)  
          
        loss = term1 + term2  
          
        if self.reduction == 'mean':  
            return loss.mean()  
        elif self.reduction == 'sum':  
            return loss.sum()  
        return loss

# -------------------------------------------------------------------  
# Ratio-Constrained Regression Head  
# -------------------------------------------------------------------  
class HierarchicalBiomassHead(nn.Module):  
    def __init__(self, in_dim, num_components=5, hidden_dim=512):  
        super().__init__()  
          
        # Common feature extractor from backbone features  
        self.common = nn.Sequential(  
            nn.Linear(in_dim, hidden_dim),  
            nn.LayerNorm(hidden_dim),  
            nn.GELU(),  
            nn.Dropout(0.3)  
        )  
          
        # Branch 1: Total Biomass (Scalar)  
        self.total_head = nn.Sequential(  
            nn.Linear(hidden_dim, 256),  
            nn.GELU(),  
            nn.Linear(256, 1),  
            nn.Softplus() # Enforce positive mass  
        )  
          
        # Branch 2: Ratios (Vector)  
        self.ratio_head = nn.Sequential(  
            nn.Linear(hidden_dim, 256),  
            nn.GELU(),  
            nn.Linear(256, num_components)  
            # Softmax applied in forward  
        )

    def forward(self, x):  
        feat = self.common(x)  
          
        # Predict Total  
        total_biomass = self.total_head(feat)  
          
        # Predict Ratios  
        ratio_logits = self.ratio_head(feat)  
        ratios = F.softmax(ratio_logits, dim=1)  #, sums to 1  
          
        # Derive Component Biomass  
        # Broadcast total across ratios: * -> (batch_size, num_components)
        component_biomass = total_biomass * ratios  
          
        return component_biomass, total_biomass, ratios

# -------------------------------------------------------------------  
# Main Lightning Module  
# -------------------------------------------------------------------  
class DinoBiomassModel(pl.LightningModule):  
    def __init__(self, learning_rate=1e-4, weight_decay=1e-3, tweedie_p=1.5):  
        super().__init__()  
        self.save_hyperparameters()  
          
        # 1. Load Backbone (DINOv2)  
        # We use timm's implementation which hosts DINOv2 weights  
        self.backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)  
        self.embed_dim = self.backbone.num_features  
          
        # 2. Freeze Early Layers (Layer-Wise Freezing Strategy)  
        # Freeze everything first  
        for param in self.backbone.parameters():  
            param.requires_grad = False  
              
        # Unfreeze last 2 blocks for fine-tuning semantic features  
        # Note: Implementation depends on specific TIMM block naming  
        for name, param in self.backbone.named_parameters():  
            if 'blocks.10' in name or 'blocks.11' in name or 'norm' in name:  
                param.requires_grad = True  
          
        # 3. Hierarchical Head  
        # Targets: -> 5 Components  
        self.head = HierarchicalBiomassHead(in_dim=self.embed_dim, num_components=5)  
          
        # 4. Loss Function  
        self.criterion = TweedieLoss(p=tweedie_p)  
          
        # Weights for competition metric (from Kaggle Overview)  
        # Green, Dead, Clover, GDM, Total  
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
    def forward(self, x):  
        # DINOv2 Forward  
        features = self.backbone(x)  
          
        # Head Forward  
        components, total_pred, ratios = self.head(features)  
        return components, total_pred

    def training_step(self, batch, batch_idx):  
        images, meta, targets, _ = batch  
          
        # Forward pass  
        # components:  
        preds, total_pred = self(images)  
          
        # Calculate Weighted Tweedie Loss  
        # We compute loss per component, then weighted sum  
        loss = 0  
        for i in range(5):  
            # Apply Tweedie to each component individually  
            l = self.criterion(preds[:, i], targets[:, i])  
            loss += l * self.weights[i]  
              
        # Logging  
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  
        return loss

    def validation_step(self, batch, batch_idx):  
        images, meta, targets, _ = batch  
        preds, total_pred = self(images)  
          
        # Calculate Weighted R2 (Metric Proxy)  
        # Note: Actual R2 calculation requires global stats, approximated here via MSE for monitoring  
        mse_loss = F.mse_loss(preds, targets)  
        self.log('val_mse', mse_loss, prog_bar=True)  
          
        # Validation Tweedie Loss  
        loss = 0  
        for i in range(5):  
            loss += self.criterion(preds[:, i], targets[:, i]) * self.weights[i]  
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):  
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),   
                                lr=self.hparams.learning_rate,   
                                weight_decay=self.hparams.weight_decay)  
          
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(  
            optimizer, T_0=10, T_mult=2, eta_min=1e-6  
        )  
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
```
## ---

**6\. Knowledge Distillation: Learning Using Privileged Information (LUPI)**

This section details the optional but highly recommended "Teacher-Student" distillation module. In agronomy, metadata (weather, soil) is a strong predictor of biomass. However, at inference time (e.g., a drone flight), we may only have the image. We can use the metadata during training to "teach" the image model.7

### **6.1 The Teacher: Tabular-Visual Fusion**

The teacher model has access to "Privileged Information" (tabular metadata). It fuses the visual embedding (frozen DINOv2) with the metadata embedding to produce a superior prediction.

Distillation Logic:  
The Student (our DinoBiomassModel above) tries to minimize two losses:

1. **Task Loss:** Discrepancy between Student Prediction and Ground Truth (Tweedie).  
2. **Distillation Loss:** Discrepancy between Student Prediction and Teacher Prediction (KL Divergence or MSE).

Crucially, because the Teacher uses metadata (e.g., "It is Winter"), it will confidently predict low "Dead" biomass. The Student, seeing only a brown image, might be confused. The distillation loss forces the Student to find visual cues (e.g., specific lighting or texture) that correlate with "Winter" to match the Teacher's confidence, effectively "baking in" the metadata knowledge.7

### **6.2 Implementation of the Distillation Loop**

We modify the training\_step of the Lightning Module to support distillation.

```Python

    def training_step_with_distillation(self, batch, batch_idx):  
        images, meta, targets, _ = batch  
          
        # 1. Student Forward (Image only)  
        s_components, s_total, s_ratios = self(images)  
          
        # 2. Teacher Forward (Image + Metadata) - inference mode, no grad  
        with torch.no_grad():  
            t_components, t_total, t_ratios = self.teacher_model(images, meta)  
              
        # 3. Task Loss (Tweedie vs Ground Truth)  
        task_loss = 0  
        for i in range(5):  
            task_loss += self.criterion(s_components[:, i], targets[:, i]) * self.weights[i]  
              
        # 4. Distillation Loss  
        # a. Logits/Response Distillation (KL Div on Ratios)  
        # Teacher ratios are "soft targets" for Student ratios  
        distill_loss_ratios = F.kl_div(torch.log(s_ratios + 1e-8), t_ratios, reduction='batchmean')  
          
        # b. Feature/Regression Distillation (MSE on Total)  
        # Student should match Teacher's estimate of total biomass  
        distill_loss_total = F.mse_loss(s_total, t_total)  
          
        # Total Loss  
        # Alpha balances Task vs Distillation  
        alpha = 0.7   
        total_loss = alpha * task_loss + (1 - alpha) * (distill_loss_ratios + distill_loss_total)  
          
        self.log('distill_loss', total_loss)  
        return total_loss
```
## ---

**7\. Strategic Recommendations and Roadmap**

### **7.1 Hyperparameter Tuning**

* **Tweedie Power ($p$):** This is the most sensitive parameter. We recommend a grid search over $p \in [1.1, 1.3, 1.5, 1.7, 1.9]$. Literature suggests values closer to 1.5 are generally robust for compound Poisson-Gamma processes.3  
* **Hierarchical Constraints:** The "Ratio-Based" architecture is theoretically superior but can suffer from "gradient vanishing" if the Total prediction effectively reaches zero (killing gradients for the Ratio head). The Softplus activation \+ epsilon in the Total head is critical to prevent this.

### **7.2 Ensemble Strategy**

Given the high variance of the "Dead" biomass target 6, single models often exhibit high variance. An ensemble of 3-5 models is recommended:

1. **Model A:** DINOv2-Small, Tweedie $p=1.3$.  
2. **Model B:** DINOv2-Base, Tweedie $p=1.7$.  
3. **Model C:** ResNet50 (Baseline check), MSE Loss.

The final prediction should be a weighted average, potentially giving lower weight to Model C if validation scores confirm DINOv2's superiority.

### **7.3 Conclusion**

This research plan represents a significant leap from standard "ImageNet-Transfer" baselines. By respecting the statistical nature of the data (via Tweedie Loss), the physical laws of the domain (via Hierarchical Constraints), and the texture-centric nature of the imagery (via DINOv2), this pipeline is engineered to capture the subtle signals of pasture biomass that elude conventional architectures. The inclusion of LUPI-based distillation further robustifies the model against the sparse data regime, ensuring that available metadata maximally benefits the visual inference process.

### ---

**Citations**

    9 Kaggle CSIRO biomass prediction solution code github  
    10 Kaggle CSIRO biomass prediction solution code \- Model list  
    3 PyTorch Forecasting \- TweedieLoss  
    4 Tweedie loss function explanation  
    5 AAAI Article \- Enforcing constraints in NN  
    14 StackOverflow \- Neural network sum constraints  
    1 Kili Tech \- DINOv2 Fine-tuning  
    11 GitHub \- Meta DINOv2  
    19 DebuggerCafe \- DINOv2 Fine-tuning vs Transfer  
    23 Knowledge Distillation \- Time Series/Image  
    22 OpenReview \- Tabular to Image Knowledge Transfer  
    16 Albumentations \- Pipelines  
    18 Albumentations \- Choosing Augmentations  
    17 Albumentations \- XYMasking/Cutout  
    6 Kaggle CSIRO discussion \- Metric analysis  
    8 Kaggle CSIRO Overview \- Scoring  
    15 Kaggle Code \- EfficientNet baseline  
    9 Kaggle Code \- Lightning/TIMM baseline  
    7 Arxiv \- Learning Using Privileged Information  
    12 Zero-Inflated Models Book  
    13 GitHub \- Count Models (Hurdle/ZINB)  
    20 EmergentMind \- DINOv2 Approaches  
    2 EmergentMind \- Frozen DINOv2 Encoder

#### **Works cited**

1. DinoV2 Fine-Tuning Tutorial: How to Maximize Accuracy for Computer Vision Tasks, accessed December 26, 2025, [https://kili-technology.com/blog/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks](https://kili-technology.com/blog/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks)  
2. Frozen DINOv2 Encoder \- Emergent Mind, accessed December 26, 2025, [https://www.emergentmind.com/topics/frozen-dinov2-encoder](https://www.emergentmind.com/topics/frozen-dinov2-encoder)  
3. TweedieLoss \- PyTorch Forecasting Documentation, accessed December 26, 2025, [https://pytorch-forecasting.readthedocs.io/en/v1.3.0/api/pytorch\_forecasting.metrics.point.TweedieLoss.html](https://pytorch-forecasting.readthedocs.io/en/v1.3.0/api/pytorch_forecasting.metrics.point.TweedieLoss.html)  
4. Tweedie Loss Function. An example: Insurance pricing | by Sathesan Thavabalasingam, accessed December 26, 2025, [https://sathesant.medium.com/tweedie-loss-function-395d96883f0b](https://sathesant.medium.com/tweedie-loss-function-395d96883f0b)  
5. DeepSaDe: Learning Neural Networks That Guarantee Domain Constraint Satisfaction, accessed December 26, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/29109/30097](https://ojs.aaai.org/index.php/AAAI/article/view/29109/30097)  
6. Analysis: Height\_Ave\_cm and Dead Biomass Prediction \- Testing the Host's Hypothesis \- CSIRO \- Image2Biomass Prediction | Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/discussion/650736](https://www.kaggle.com/competitions/csiro-biomass/discussion/650736)  
7. Learning Using Generated Privileged Information by Text-to-Image Diffusion Models \- arXiv, accessed December 26, 2025, [https://arxiv.org/html/2309.15238v2](https://arxiv.org/html/2309.15238v2)  
8. CSIRO \- Image2Biomass Prediction | Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/overview/description](https://www.kaggle.com/competitions/csiro-biomass/overview/description)  
9. CSIRO Img2Bio: EDA & baseline Lit TIMM regression \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/code/jirkaborovec/csiro-img2bio-eda-baseline-lit-timm-regression](https://www.kaggle.com/code/jirkaborovec/csiro-img2bio-eda-baseline-lit-timm-regression)  
10. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/code](https://www.kaggle.com/competitions/csiro-biomass/code)  
11. NKI-AI/meta-dinov2: PyTorch code and models for the DINOv2 self-supervised learning method. \- GitHub, accessed December 26, 2025, [https://github.com/NKI-AI/meta-dinov2](https://github.com/NKI-AI/meta-dinov2)  
12. Zero-Inflated and Hurdle Models \- John Della Rosa, accessed December 26, 2025, [https://johndellarosa.github.io/projects/distribution-book/zero-inflated](https://johndellarosa.github.io/projects/distribution-book/zero-inflated)  
13. wanglab-georgetown/countmodels: Compare Hurdle and Zero-inflated Count Models \- GitHub, accessed December 26, 2025, [https://github.com/wanglab-georgetown/countmodels](https://github.com/wanglab-georgetown/countmodels)  
14. Restrict the sum of outputs in a neural network regression (Keras) \- Stack Overflow, accessed December 26, 2025, [https://stackoverflow.com/questions/60003673/restrict-the-sum-of-outputs-in-a-neural-network-regression-keras](https://stackoverflow.com/questions/60003673/restrict-the-sum-of-outputs-in-a-neural-network-regression-keras)  
15. CSIRO Biomass Prediction using EfficientNet-B3 \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/code/barkataliarbab/csiro-biomass-prediction-using-efficientnet-b3](https://www.kaggle.com/code/barkataliarbab/csiro-biomass-prediction-using-efficientnet-b3)  
16. Pipelines (Compose) \- Albumentations, accessed December 26, 2025, [https://albumentations.ai/docs/2-core-concepts/pipelines/](https://albumentations.ai/docs/2-core-concepts/pipelines/)  
17. Albumentations: XYMasking. Exploring the Depths of Image… | by Vladimir Iglovikov | Medium, accessed December 26, 2025, [https://medium.com/@iglovikov/albumentations-xymasking-71bdc7925c55](https://medium.com/@iglovikov/albumentations-xymasking-71bdc7925c55)  
18. Choosing Augmentations for Model Generalization \- Albumentations, accessed December 26, 2025, [https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/)  
19. DINOv2 for Image Classification: Fine-Tuning vs Transfer Learning \- DebuggerCafe, accessed December 26, 2025, [https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/)  
20. DINOv2-Based Approaches Overview \- Emergent Mind, accessed December 26, 2025, [https://www.emergentmind.com/topics/dinov2-based-approaches](https://www.emergentmind.com/topics/dinov2-based-approaches)  
21. Freezing Layers in Deep Learning and Transfer Learning | Exxact Blog, accessed December 26, 2025, [https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models](https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models)  
22. On Transferring Expert Knowledge from Tabular Data to Images \- OpenReview, accessed December 26, 2025, [https://openreview.net/forum?id=B1VWS7ZRm6](https://openreview.net/forum?id=B1VWS7ZRm6)  
23. Image Representation-Driven Knowledge Distillation for Improved Time-Series Interpretation on Wearable Sensor Data \- NIH, accessed December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12567629/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12567629/)