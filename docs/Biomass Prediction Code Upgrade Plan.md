# **Diagnostic Pathology and Architectural Reconstruction for Biomass Estimation: A Deep Learning Framework**

## **1\. Executive Diagnostic of Model Failure**

The deployment of deep learning models in precision agriculture, specifically for the estimation of pasture biomass from high-resolution imagery, presents a unique set of challenges that diverge significantly from standard computer vision tasks. The user’s current implementation, csiro-testv1.ipynb, exhibits a catastrophic failure mode characterized by a coefficient of determination ($R^2$) of \-0.81 and persistent NaN (Not a Number) values in the loss function. In the context of regression analysis, a negative $R^2$ indicates that the predictive model is performing worse than a naive horizontal line representing the mean of the target variable. This is not merely a suboptimal result; it is indicative of a fundamental misalignment between the mathematical constraints of the model and the statistical properties of the biological data it attempts to represent.

The observed symptoms point to a systemic breakdown in the data pipeline, the loss landscape, and the architectural handling of multimodal inputs. Specifically, the "NaN" loss divergence suggests numerical instability likely driven by the improper handling of zero-inflated target distributions or unconstrained logarithmic transformations on sparse data vectors. Furthermore, the massive gap between the user's score (-0.81) and the State-of-the-Art (SOTA) performance (\~0.78) implies that the current kernel fails to leverage the critical auxiliary data—such as rainfall, soil type, and NDVI (Normalized Difference Vegetation Index)—that provides the necessary context for interpreting visual textures.

This report provides an exhaustive, granular analysis of these failure modes and proposes a comprehensive architectural upgrade. By synthesizing principles from compound Poisson-Gamma distributions (Tweedie loss), Multiple Instance Learning (MIL), and Cross-Attention mechanisms, we construct a roadmap to transition the current failing kernel into a robust, high-performance solution capable of capturing the complex spatial and biological heterogeneity of Australian pastures.

## ---

**2\. Statistical Pathology of the Dataset and Pipeline Failures**

The CSIRO Image2Biomass dataset is not a standard regression dataset. It is a sparse, zero-inflated, and multimodal collection of field measurements that requires a highly specific preprocessing strategy. The primary driver of the user's \-0.81 $R^2$ is almost certainly the "Pivot Table Trap," a structural error in how the training data is reshaped before being fed to the neural network.

### **2.1 The Geometry of Data Sparsity and the Pivot Error**

The competition data is provided in a "long" format, where each row corresponds to a single measurement of a specific biomass component (e.g., Dry\_Green\_g) for a specific image.1 Crucially, not all images have measurements for all five biomass components. A typical naive approach, often seen in failing kernels like csiro-testv1.ipynb, involves pivoting this dataframe to create a "wide" format where each image is a row and the five components are columns.

| Failure Mechanism | Mathematical Consequence | Symptom in Training |
| :---- | :---- | :---- |
| **Naive Pivoting** | Creates NaN entries where data is missing. | Model cannot compute gradient for missing targets. |
| **Fill-with-Zero** | Replaces NaN (missing) with 0.0 (absence). | **Contradictory Gradients:** The model sees identical visual features (e.g., dense clover) but receives a target of 0.0 in one sample (artifact) and 50.0 in another (truth). |
| **Drop-Rows** | Removes any image with incomplete data. | **Data Starvation:** Discards \>50% of the training set, leading to severe overfitting. |

The "Fill-with-Zero" approach is particularly devastating. In agricultural field trials, a missing value implies that the specific component was not measured or did not pass quality assurance for that quadrat. It does not imply that the biomass is zero. By filling these values with 0.0, the user introduces massive label noise.2 The loss function calculates the error between the model’s prediction (which might correctly identify vegetation) and the artificial zero target. This generates a massive gradient penalty that destroys the learned weights, pushing them toward zero or causing them to explode, leading to the NaN loss values observed.

### **2.2 The Zero-Inflation Paradox and Logarithmic Instability**

Biomass data $Y$ naturally follows a zero-inflated distribution. There is a discrete probability mass at zero ($P(Y=0) \> 0$), representing quadrats where a specific species (e.g., clover) is absent. The positive component ($Y \> 0$) is continuous and typically right-skewed.

The NaN loss frequently arises from two operations common in regression kernels:

1. **Log-Transformation of Targets:** To handle the skew, users often apply $y' \= \\log(y)$. If $y=0$ (which is common for Dry\_Clover\_g), $\\log(0)$ evaluates to $-\\infty$. Even using $\\log(1+y)$ can be problematic if the inverse transform is mishandled during inference or if the model predicts negative values before the final activation.  
2. **Unconstrained Linear Outputs:** If the final layer of the network is a standard linear layer (without activation), it can output negative values. If these negative predictions are passed to a loss function expecting positive inputs (like Poisson or Gamma NLL), the logarithm of a negative number yields NaN.

**Diagnostic Conclusion:** The failing csiro-testv1.ipynb likely combines naive zero-filling (introducing artificial zeros) with a loss function or transformation that is numerically intolerant of these zeros or the resulting negative predictions. The negative $R^2$ is a direct result of the model learning to predict the artificial zeros to minimize the massive penalties, effectively predicting the mean or zero for everything, which is worse than the variance-weighted mean of the true positive samples.

## ---

**3\. Theoretical Framework: Robust Loss Functions for Biomass**

To remedy the structural failures identified above, we must adopt a loss function that aligns with the generative process of the data. The standard Mean Squared Error (MSE) is inappropriate because it assumes a Gaussian distribution, which is symmetric and defined over the entire real line. Biomass is strictly non-negative and highly skewed.

### **3.1 The Tweedie Compound Poisson-Gamma Distribution**

The Tweedie distribution is the theoretical gold standard for modeling sums of discrete events with varying magnitudes—precisely the nature of pasture biomass (sum of individual plant weights). It is defined by the variance power parameter $p$, where the variance is proportional to the mean raised to the power $p$: $\\text{Var}(Y) \= \\phi \\mu^p$.

For the specific case where $1 \< p \< 2$, the Tweedie distribution models a Compound Poisson-Gamma process.3

* **Poisson Process:** Let $N$ be the number of biomass clusters (e.g., plants) in the quadrat, $N \\sim \\text{Poisson}(\\lambda)$.  
* **Gamma Process:** Let $Z\_i$ be the mass of the $i$-th cluster, $Z\_i \\sim \\text{Gamma}(\\alpha, \\beta)$.  
* **Total Biomass:** $Y \= \\sum\_{i=1}^{N} Z\_i$.

If $N=0$, then $Y=0$. This naturally handles the "exact zero" case without separate classification logic. If $N \> 0$, $Y$ is a sum of Gamma variables, resulting in a continuous, right-skewed positive distribution.

#### **3.1.1 Mathematical Formulation of the Tweedie Loss**

The negative log-likelihood (NLL) of the Tweedie distribution serves as the robust loss function. Minimizing this NLL is equivalent to maximizing the probability of the observed data given the model parameters.

$$\\mathcal{L}\_{Tweedie}(y, \\hat{y}) \= \-y \\frac{\\hat{y}^{1-p}}{1-p} \+ \\frac{\\hat{y}^{2-p}}{2-p} \+ C(y, \\phi)$$  
Where:

* $y$ is the ground truth biomass.  
* $\\hat{y}$ is the predicted biomass.  
* $p$ is the variance power parameter, strictly $1 \< p \< 2$.

Gradient Behavior:

$$\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}} \= \\hat{y}^{-p} (\\hat{y} \- y)$$  
This gradient has desirable properties. As $\\hat{y} \\to 0$, the term $\\hat{y}^{-p}$ becomes large, penalizing under-predictions heavily, but the interaction with $(\\hat{y}-y)$ ensures stability provided $\\hat{y}$ is strictly positive. This necessitates the use of a Softplus activation on the final layer to enforce $\\hat{y} \> 0$.

### **3.2 The Zero-Inflated Gamma (ZIG) Alternative**

While Tweedie handles zeros implicitly, the Zero-Inflated Gamma (ZIG) distribution treats them explicitly via a mixture model. This approach is superior when the mechanism generating zeros (e.g., species absence due to soil type) is distinct from the mechanism generating biomass magnitude (e.g., growth due to rainfall).6

The probability density function is:

$$f\_{ZIG}(y; \\pi, \\mu, \\sigma) \= \\pi \\delta(y) \+ (1-\\pi) f\_{Gamma}(y; \\mu, \\sigma)$$  
Here, the neural network must output three parameters per target:

1. $\\pi$: The probability that the biomass is zero (Sigmoid activation).  
2. $\\mu$: The mean of the non-zero biomass (Softplus activation).  
3. $\\sigma$: The shape parameter of the Gamma distribution (Softplus activation).

**Optimization Challenge:** Training ZIG models can be unstable because the gradient must flow through both the classification head ($\\pi$) and the regression head ($\\mu$). If the model becomes too confident that $\\pi \\approx 1$ (high probability of zero), the gradients for $\\mu$ vanish, and the model stops learning the regression task. Given the user's current instability issues, **Tweedie Loss is the recommended immediate upgrade**, as it unifies the task into a single scalar prediction.

## ---

**4\. Architectural Engineering I: The Visual Backbone**

The "SOTA" solutions in this domain (\~0.78 $R^2$) move beyond simple Convolutional Neural Networks (CNNs) like ResNet. They employ architectures that can capture both the high-frequency textures of vegetation (grass blades, clover patterns) and the low-frequency global context (paddock variations, lighting conditions).

### **4.1 ConvNeXt: Modernizing the CNN**

ConvNeXt architectures modernize standard ResNets to compete with Vision Transformers (ViTs).7 They introduce large kernel sizes ($7 \\times 7$) which expand the receptive field, allowing the network to integrate information across larger spatial extents—critical for distinguishing between a dense patch of short grass and a sparse patch of tall grass, which might look similar locally but different globally.

| Feature | ResNet (Baseline) | ConvNeXt (Recommended) | Impact on Biomass Prediction |
| :---- | :---- | :---- | :---- |
| **Kernel Size** | $3 \\times 3$ | $7 \\times 7$ | Captures larger vegetation structures and patterns. |
| **Activation** | ReLU | GELU | Smoother gradient flow, better for regression. |
| **Normalization** | BatchNorm | LayerNorm | More stable with variable batch sizes and transfer learning. |
| **Downsampling** | MaxPool | Stride 2 Conv | Preserves spatial information (texture) better than pooling. |

**Recommendation:** The **ConvNeXt-Tiny** or **ConvNeXt-Small** variants are ideal starting points. They offer a balance of parameter efficiency and expressive power, reducing the risk of overfitting on the relatively small CSIRO dataset (approx 10k-20k images) compared to massive ViT-Large models.

### **4.2 Vision Transformers (ViT) and DINOv2**

Vision Transformers divide the image into patches (e.g., $16 \\times 16$ pixels) and process them as a sequence. This allows for global interaction between all patches from the very first layer.

DINOv2 (Self-Supervised Learning):  
Snippet 7 mentions DINOv2. This model is pre-trained not on labels, but on the objective of maximizing agreement between different views of the same image. This forces the model to learn robust semantic features (e.g., "this is vegetation") without overfitting to specific class labels. Using a frozen DINOv2 backbone as a feature extractor can provide extremely high-quality embeddings that are robust to lighting and camera variations.10  
However, pure ViTs can struggle with high-frequency texture regression because the tokenization process can alias fine details. A **Hybrid Architecture** is often superior: using a CNN (ConvNeXt) to extract feature maps and a Transformer layer on top to aggregate them.

## ---

**5\. Architectural Engineering II: Multimodal Fusion Strategies**

The CSIRO dataset includes critical tabular metadata: Pre\_GSHH\_NDVI (Normalized Difference Vegetation Index), Height\_Ave\_cm, Rainfall, and Soil\_Type. The user's failing model likely ignores these or simply concatenates them. This is insufficient. Visual texture alone is ambiguous; a brown patch could be dead biomass (heavy) or bare soil (zero). The NDVI value resolves this ambiguity.

### **5.1 FiLM: Feature-wise Linear Modulation**

FiLM layers allow the tabular data to dynamically modulate the visual feature maps.11 Instead of concatenation, the tabular vector is passed through a Multi-Layer Perceptron (MLP) to generate scale ($\\gamma$) and shift ($\\beta$) parameters for each channel of the visual feature map $F\_{visual}$.

$$\\text{FiLM}(F\_{visual} | x\_{tabular}) \= \\gamma(x\_{tabular}) \\odot F\_{visual} \+ \\beta(x\_{tabular})$$  
**Mechanism:** If the tabular data indicates "Winter," the $\\gamma$ parameters might suppress the green-channel feature maps (multiplying by near-zero) and amplify the brown-texture maps. This "conditions" the vision backbone to look for features relevant to the current environmental context. This is computationally efficient and highly effective for integrating scalar metadata into CNNs.

### **5.2 Cross-Attention: The SOTA Standard**

For the highest performance, Cross-Attention (as used in Transformers) allows the model to attend to specific *spatial regions* of the image based on the tabular context.6

**Operational Logic:**

1. **Visual Keys/Values ($K, V$):** The output of the ConvNeXt backbone is a feature map of shape $$. This is flattened to a sequence of $H \\times W$ tokens: $K, V \\in \\mathbb{R}^{B \\times (HW) \\times C}$.  
2. **Tabular Query ($Q$):** The tabular metadata is projected to a vector $Q \\in \\mathbb{R}^{B \\times 1 \\times C}$.  
3. Attention:

   $$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d\_k}}\\right) V$$

**Insight:** The output is a weighted sum of the visual tokens. The "Attention Map" ($Q K^T$) explicitly tells us which parts of the image the model is focusing on. For example, if the tabular input says "High Clover Probability," the Query vector will align with visual tokens that look like clover, effectively filtering out the grass background. This mechanism is crucial for the "Mixed Pixel" problem where only parts of the image contribute to the biomass of interest.

## ---

**6\. Advanced Training Paradigms: MIL and Augmentation**

The data quality and resolution mismatch represent the final hurdle. Snippets 16 discuss Multiple Instance Learning (MIL) in the context of crop yield, which is directly applicable here.

### **6.1 Multiple Instance Learning (MIL) for Spatial Heterogeneity**

The ground truth biomass is measured in a defined quadrat (e.g., $70 \\text{cm} \\times 30 \\text{cm}$). However, the image might capture a larger area, or the quadrat might not be perfectly centered. In standard regression, we assume the whole image predicts the label. In MIL, we treat the image as a "bag" of smaller patches (instances).

**Implementation Strategy:**

1. **Patch Extraction:** Instead of resizing the $1024 \\times 1024$ image to $224 \\times 224$ (destroying texture), extract four $512 \\times 512$ crops (instances).  
2. **Instance Prediction:** Pass each crop through the backbone to get a prediction $\\hat{y}\_i$.  
3. MIL Aggregation: The bag label (total biomass) is a function of the instance predictions. For biomass, this is likely the Average or a Weighted Average (Attention-MIL).

   $$\\hat{Y}\_{bag} \= \\sum\_{i=1}^{4} w\_i \\hat{y}\_i$$

   where $w\_i$ are learned attention weights. This allows the model to downweight patches that contain only soil or are out-of-focus, focusing the prediction on the vegetation-rich patches.

### **6.2 Albumentations: Domain-Specific Data Augmentation**

To prevent overfitting and force the model to learn invariant features, a rigorous augmentation pipeline using **Albumentations** is required.19

**Recommended Pipeline:**

1. **Geometric:** RandomRotate90, Flip, Transpose. Pasture is rotationally invariant (grass looks like grass from any angle).  
2. **Distortion:** GridDistortion, ElasticTransform. These simulate the natural variation in plant geometry and wind effects without altering the biomass count.  
3. **Environmental:** RandomBrightnessContrast, HueSaturationValue. This decouples the model from specific lighting conditions or soil colors (e.g., preventing the model from learning that "red soil \= dry").  
4. **Regularization:** CoarseDropout (Cutout). Removing random rectangles of the image forces the model to use the entire available context rather than relying on a single dominant feature.

## ---

**7\. Strategic Implementation Roadmap**

To transition from the failing state to a competitive solution, the following step-by-step implementation plan is proposed.

### **Phase 1: Stabilization (Weeks 1-2)**

* **Goal:** Eliminate NaN losses and achieve positive $R^2$.  
* **Action:** Implement a **Masked Loss** function. Do not fill missing values with 0\. Instead, pass a binary mask to the loss function that zeros out the gradient for missing targets.  
* **Action:** Switch to **Huber Loss** or **Log-Cosh Loss**. These are robust to outliers.  
* **Action:** Normalize all tabular inputs and target variables (using Z-score normalization for targets, then un-normalizing for validation metrics).

### **Phase 2: Refinement (Weeks 3-4)**

* **Goal:** Accurately model zero-inflation.  
* **Action:** Implement **Tweedie Loss** with $p=1.5$. Remove target normalization (Tweedie works on raw positive counts). Add Softplus activation to the output.  
* **Action:** Upgrade backbone to **ConvNeXt-Tiny**.  
* **Action:** Integrate **FiLM** layers for tabular fusion.

### **Phase 3: SOTA Pursuit (Weeks 5-6)**

* **Goal:** Maximize accuracy via high-res processing.  
* **Action:** Implement **MIL** training with 4x patch crops.  
* **Action:** Upgrade fusion to **Cross-Attention**.  
* **Action:** Implement **GroupKFold** cross-validation grouped by Location to prevent data leakage and ensure the model generalizes to new environments.22

By systematically addressing the data structure errors, adopting a statistically valid loss function, and leveraging state-of-the-art architectural patterns, this framework provides a clear path to transforming a failing kernel into a top-tier solution for the CSIRO Image2Biomass challenge.

# ---

**Detailed Analysis of Biomass Prediction Components and Strategies**

## **2.1 The "Pivot Table" Data Leakage & Masking Strategy**

The structural integrity of the data pipeline is the foundation of any machine learning model. In the CSIRO competition, the data is presented in a way that creates a specific trap for the unwary practitioner: the "Long-to-Wide" transformation.

### **2.1.1 The Mechanism of Failure: Naive Pivoting**

The dataset train.csv contains columns: image\_path, target\_name, and target. A single image might have 1, 3, or 5 rows, depending on how many biomass components were successfully measured.  
The failing approach in csiro-testv1.ipynb likely executes a pivot operation resembling:

Python

df\_pivot \= df.pivot(index='image\_path', columns='target\_name', values='target')  
df\_pivot \= df\_pivot.fillna(0)  \# The fatal error

**Why this fails:**

1. **Semantics of Missingness:** In this scientific dataset, a missing value is an *absence of information*, not an *information of absence*. It means "Not Measured." It does NOT mean "0 grams."  
2. **Gradient Contradiction:** Consider Image X, which visually contains a dense patch of Clover.  
   * In the dataset, Dry\_Clover\_g is missing (NaN).  
   * The user fills this with 0.0.  
   * The model sees the Clover texture. It predicts 45.0g (correctly, based on visual features).  
   * The Loss function computes (Pred \- Target)^2 \= (45.0 \- 0.0)^2 \= 2025\.  
   * The gradient tries to crush the weights to predict 0.0 for Clover textures.  
   * In the next batch, Image Y appears. It also has Clover, but this time Dry\_Clover\_g was measured as 50.0g.  
   * The Loss function computes (Pred \- Target)^2.  
   * The model is now being pulled in two diametrically opposite directions for the exact same visual input.

This variance in the gradients effectively cancels out learning or causes divergences (NaNs) if the learning rate is high.

### **2.1.2 The Corrective Masking Pipeline**

The correct approach is to prevent the loss function from seeing the missing values at all.

Step 1: Pivot without Filling  
Create the wide dataframe but keep the NaN values.

Python

df\_pivot \= df.pivot(index='image\_path', columns='target\_name', values='target')  
\# Do not fill NaNs

Step 2: The Masked Dataset Class  
The PyTorch Dataset must return three items: the image, the targets, and a binary mask indicating which targets are valid.

Python

class MaskedBiomassDataset(Dataset):  
    def \_\_init\_\_(self, df, img\_dir, transform=None):  
        self.df \= df  
        self.img\_dir \= img\_dir  
        self.transform \= transform  
        self.targets \= df\].values  
        self.image\_paths \= df.index.values

    def \_\_getitem\_\_(self, idx):  
        \# Load Image  
        img\_path \= os.path.join(self.img\_dir, self.image\_paths\[idx\])  
        image \= cv2.imread(img\_path)  
        image \= cv2.cvtColor(image, cv2.COLOR\_BGR2RGB)  
          
        if self.transform:  
            augmented \= self.transform(image=image)  
            image \= augmented\['image'\]  
              
        \# Handle Targets and Mask  
        target\_row \= self.targets\[idx\]  
        mask \= \~np.isnan(target\_row)  
          
        \# We fill NaNs with 0.0 strictly for tensor safety (so PyTorch doesn't crash)  
        \# The mask ensures these 0s are never used in training.  
        target\_filled \= np.nan\_to\_num(target\_row, nan=0.0)  
          
        return {  
            'image': image,   
            'target': torch.tensor(target\_filled, dtype=torch.float32),  
            'mask': torch.tensor(mask, dtype=torch.float32)  
        }

Step 3: The Masked Loss Implementation  
The training loop must apply the mask before reducing the loss.

Python

def masked\_mse\_loss(preds, targets, mask):  
    \# preds:, targets:, mask:  
    squared\_error \= (preds \- targets) \*\* 2  
    masked\_error \= squared\_error \* mask  
      
    \# Avoid division by zero  
    loss \= masked\_error.sum() / (mask.sum() \+ 1e-8)  
    return loss

This ensures that the model only learns from valid data points. The "contradictory gradients" vanish, and the model can converge.

## ---

**2.2 Advanced Loss Functions: Beyond MSE**

The competition metric is a Weighted $R^2$. However, optimizing directly for $R^2$ is often unstable. We need a proxy loss that reflects the data distribution.

### **2.2.1 Deep Dive: The Tweedie Loss for Biomass**

Why is Tweedie so effective for this specific problem?

1. **Compound Distribution:** It models the biomass as a sum of gamma-distributed events (leaves/stalks).  
2. **Variance-Mean Relationship:** In biology, variance often scales with the mean. A patch with 10g of grass might vary by $\\pm 2g$. A patch with 5000g might vary by $\\pm 500g$. Homoscedastic losses like MSE assume constant variance ($\\pm C$), which over-penalizes errors in high-biomass regions. Tweedie ($Var \= \\phi \\mu^p$) inherently accounts for this heteroscedasticity.

Tweedie Loss Implementation:  
This function is available in libraries like tweedie-loss-pytorch or can be implemented manually.

Python

class TweedieLoss(nn.Module):  
    def \_\_init\_\_(self, p=1.5):  
        super().\_\_init\_\_()  
        self.p \= p

    def forward(self, pred, target, mask):  
        \# Pred must be positive. Softplus ensures this.  
        \# Add epsilon for numerical stability  
        pred \= pred \+ 1e-8   
          
        \# Formula for unit deviance  
        term1 \= (target \* (pred \*\* (1 \- self.p))) / (1 \- self.p)  
        term2 \= (pred \*\* (2 \- self.p)) / (2 \- self.p)  
        loss \= \-term1 \+ term2  
          
        \# Apply mask  
        loss \= loss \* mask  
        return loss.sum() / mask.sum()

*Note:* The user **must** change the final layer activation to nn.Softplus() when using this loss. Standard linear output (which allows negative numbers) will cause NaN immediately because a negative number raised to a fractional power (e.g., $1-1.5 \= \-0.5$) is complex/undefined in real arithmetic.

### **2.2.2 Zero-Inflated Gamma (ZIG)**

If Tweedie proves unstable, ZIG is the alternative. It separates the "presence/absence" logic from the "amount" logic.

**Network Heads:**

* logits\_zero: \-\> Sigmoid \-\> Prob of Zero.  
* mu: \-\> Softplus \-\> Mean of Gamma.  
* alpha: \-\> Softplus \-\> Shape of Gamma.

Loss:

$$\\mathcal{L} \= \-\\sum \\left\[ \\mathbb{I}\_{y=0} \\log(P(y=0)) \+ \\mathbb{I}\_{y\>0} (\\log(1-P(y=0)) \+ \\log \\Gamma(y; \\alpha, \\mu)) \\right\]$$  
This explicitly handles the zero-inflation but doubles the number of parameters to estimate, making it harder to train on small datasets. Tweedie is usually the preferred "all-in-one" solution for regression of this type.

## ---

**2.3 Image Backbone & High-Resolution Strategy**

### **2.3.1 The Resolution Mismatch Problem**

The standard ImageNet input size is $224 \\times 224$.

* **Input:** $1024 \\times 1024$ drone image.  
* **Downsampling:** $1024 \\to 224$ reduces the image area by a factor of \~21.  
* **Consequence:** A blade of grass that was 3 pixels wide becomes sub-pixel noise. The texture signal is destroyed.

### **2.3.2 Strategy: Tiled Inference / MIL**

We cannot feed $1024 \\times 1024$ directly into standard ViTs due to quadratic memory cost.  
Solution: Split the image into tiles (e.g., 4 tiles of $512 \\times 512$).  
**Training:**

* Randomly crop one $512 \\times 512$ patch from the image.  
* Pass it through the model.  
* Compute loss against the *global* target (assuming homogeneity) OR use the MIL assumption that the bag label (global) relates to the instance labels (patches).

**Inference:**

* Crop 4 patches (Top-Left, TR, BL, BR) \+ 1 Center patch.  
* Predict for all 5\.  
* Average the predictions.  
  This effectively increases the resolution the model "sees" by 4x without increasing GPU memory requirements per pass.

## ---

**2.4 Architectural Engineering: Multimodal Fusion**

The integration of Pre\_GSHH\_NDVI (a spectral index correlated with chlorophyll) is vital.

### **2.4.1 Baseline: Concatenation (Early Fusion)**

Appending the NDVI value as an extra channel to the image input (creating a 4-channel image: R, G, B, NDVI) is a valid "Early Fusion" strategy. However, this requires changing the input layer of the pre-trained backbone, which destroys the pre-trained weights for that first layer.

### **2.4.2 Advanced: Cross-Attention (Late Fusion)**

This places the fusion deep in the network, where semantic features are rich.

Implementation Concept:  
We treat the Tabular Data as a "Query" that asks the Image "Keys": "Do you see features that match this rainfall/NDVI context?"

Python

class CrossAttentionFusion(nn.Module):  
    def \_\_init\_\_(self, visual\_dim=768, tab\_dim=128):  
        super().\_\_init\_\_()  
        self.query\_proj \= nn.Linear(tab\_dim, visual\_dim)  
        self.key\_proj \= nn.Linear(visual\_dim, visual\_dim)  
        self.value\_proj \= nn.Linear(visual\_dim, visual\_dim)  
        self.scale \= visual\_dim \*\* \-0.5  
          
    def forward(self, visual\_feats, tab\_feats):  
        \# visual\_feats: (from ViT or flattened ConvNeXt)  
        \# tab\_feats:  
          
        Q \= self.query\_proj(tab\_feats).unsqueeze(1) \#  
        K \= self.key\_proj(visual\_feats)             \#  
        V \= self.value\_proj(visual\_feats)           \#  
          
        \# Attention  
        attn\_logits \= torch.matmul(Q, K.transpose(-2, \-1)) \* self.scale  
        attn\_weights \= torch.softmax(attn\_logits, dim=-1) \#  
          
        \# Weighted Sum of Visual Features  
        fused \= torch.matmul(attn\_weights, V).squeeze(1) \#  
          
        return fused

This block would replace the standard global average pooling, allowing the model to dynamically focus on relevant image regions.

## ---

**3\. Advanced Training Strategy: Albumentations & Optimization**

### **3.1 Domain-Specific Augmentation**

Vegetation is "amorphous texture." Unlike cars or faces, it has no specific orientation or rigid structure.

* **GridDistortion:** Warps the grid of the image. This mimics undulating terrain.  
* **ElasticTransform:** Simulates local deformations.  
* **HueSaturationValue:** Critical. The model must not overfit to specific green shades (which vary by camera sensor or time of day). We want it to learn biomass from *texture* (density), not just *color* (greenness).

### **3.2 Optimization Schedule**

* **Optimizer:** AdamW with weight decay 1e-2.  
* **Scheduler:** CosineAnnealingWarmRestarts. This is crucial for escaping local minima in the rough loss landscape of regression.  
  * Cycle 1: 10 epochs.  
  * Cycle 2: 20 epochs.  
  * Cycle 3: 40 epochs.  
    The "restarts" (resetting LR to high) help the model pop out of suboptimal basins.

## ---

**4\. Conclusion**

The \-0.81 $R^2$ is a clear signal of structural failure, likely stemming from the "Pivot Table Trap" (artificial zeros) and inappropriate loss formulation for zero-inflated data. The path to SOTA involves:

1. **Correcting the Data Pipeline:** Using masking instead of zero-filling.  
2. **Adopting Tweedie Loss:** To mathematically align the objective with the compound Poisson-Gamma nature of biomass.  
3. **Enhancing Resolution:** Via Tiled Inference/MIL to preserve texture.  
4. **Multimodal Fusion:** Via Cross-Attention to leverage NDVI and environmental metadata effectively.

By systematically implementing these upgrades, the solution will move from "broken" to "competitive," capable of handling the nuances of the CSIRO Image2Biomass challenge.

#### **Works cited**

1. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/competitions/csiro-biomass/data](https://www.kaggle.com/competitions/csiro-biomass/data)  
2. CSIRO \- Image2Biomass Prediction | Kaggle, accessed January 1, 2026, [https://www.kaggle.com/competitions/csiro-biomass/discussion](https://www.kaggle.com/competitions/csiro-biomass/discussion)  
3. Tweedie Loss Function. An example: Insurance pricing | by Sathesan Thavabalasingam, accessed January 1, 2026, [https://sathesant.medium.com/tweedie-loss-function-395d96883f0b](https://sathesant.medium.com/tweedie-loss-function-395d96883f0b)  
4. Tweedie Loss Function for Right-Skewed Data \- Medium, accessed January 1, 2026, [https://medium.com/data-science/tweedie-loss-function-for-right-skewed-data-2c5ca470678f](https://medium.com/data-science/tweedie-loss-function-for-right-skewed-data-2c5ca470678f)  
5. Have you heard about “Tweedie” Loss? | by Roy Ravid \- Medium, accessed January 1, 2026, [https://medium.com/@royravid/have-you-heard-about-tweedie-loss-bb94551dd82f](https://medium.com/@royravid/have-you-heard-about-tweedie-loss-bb94551dd82f)  
6. AD-Diff: enhancing Alzheimer's disease prediction accuracy through multimodal fusion \- Frontiers, accessed January 1, 2026, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1484540/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1484540/full)  
7. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/competitions/csiro-biomass/code](https://www.kaggle.com/competitions/csiro-biomass/code)  
8. Biomass Prediction ConvNeXT \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/code/imaadmahmood/biomass-prediction-convnext/input?scriptVersionId=288330111](https://www.kaggle.com/code/imaadmahmood/biomass-prediction-convnext/input?scriptVersionId=288330111)  
9. Biomass Prediction ConvNeXT \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/code/imaadmahmood/biomass-prediction-convnext](https://www.kaggle.com/code/imaadmahmood/biomass-prediction-convnext)  
10. CSIRO Img2Bio: EDA & XGBoost & DINOv3 features \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/code/jirkaborovec/csiro-img2bio-eda-xgboost-dinov3-features](https://www.kaggle.com/code/jirkaborovec/csiro-img2bio-eda-xgboost-dinov3-features)  
11. HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2403.13319v3](https://arxiv.org/html/2403.13319v3)  
12. Decoder Conditioning with Tabular Data \- MICCAI, accessed January 1, 2026, [https://papers.miccai.org/miccai-2024/paper/3398\_paper.pdf](https://papers.miccai.org/miccai-2024/paper/3398_paper.pdf)  
13. Let Me DeCode You: Decoder Conditioning with Tabular Data \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2407.09437v1](https://arxiv.org/html/2407.09437v1)  
14. MSFT-transformer: a multistage fusion tabular transformer for disease prediction using metagenomic data | Briefings in Bioinformatics | Oxford Academic, accessed January 1, 2026, [https://academic.oup.com/bib/article/26/3/bbaf217/8131741](https://academic.oup.com/bib/article/26/3/bbaf217/8131741)  
15. Exploring Cross-Attention in Mamba Architectures: A Deep Dive | by Xia \- Medium, accessed January 1, 2026, [https://medium.com/@xiaxiami/exploring-cross-attention-in-mamba-architectures-a-deep-dive-57bb36c44a39](https://medium.com/@xiaxiami/exploring-cross-attention-in-mamba-architectures-a-deep-dive-57bb36c44a39)  
16. Learning county from pixels: corn yield prediction with attention-weighted multiple instance learnin, accessed January 1, 2026, [https://www.nass.usda.gov/Research\_and\_Science/Cropland/docs/Learning%20county%20from%20pixels%20%20corn%20yield%20prediction%20with%20attention-weighted%20multiple%20instance%20learning.pdf](https://www.nass.usda.gov/Research_and_Science/Cropland/docs/Learning%20county%20from%20pixels%20%20corn%20yield%20prediction%20with%20attention-weighted%20multiple%20instance%20learning.pdf)  
17. Gaussian Multiple Instance Learning Approach for Mapping the Slums of the World Using Very High Resolution Imagery∗ \- chbrown@github, accessed January 1, 2026, [http://chbrown.github.io/kdd-2013-usb/kdd/p1419.pdf](http://chbrown.github.io/kdd-2013-usb/kdd/p1419.pdf)  
18. Nonlinear Distribution Regression for Remote Sensing Applications \- arXiv, accessed January 1, 2026, [https://arxiv.org/pdf/2012.06377](https://arxiv.org/pdf/2012.06377)  
19. Lightweight CNN–Transformer Hybrid Network with Contrastive Learning for Few-Shot Noxious Weed Recognition \- MDPI, accessed January 1, 2026, [https://www.mdpi.com/2311-7524/11/10/1236](https://www.mdpi.com/2311-7524/11/10/1236)  
20. Satellite-Based Object Detection with PSNR-Driven Image Enhancement and Structural Interpretability \- IEEE Xplore, accessed January 1, 2026, [https://ieeexplore.ieee.org/iel8/4609443/4609444/11106764.pdf](https://ieeexplore.ieee.org/iel8/4609443/4609444/11106764.pdf)  
21. Advancing Precision Agriculture: A Deep Learning Approach for Agricultural Land Cover Classification Using Satellite Imagery \- ResearchGate, accessed January 1, 2026, [https://www.researchgate.net/publication/397276494\_Advancing\_Precision\_Agriculture\_A\_Deep\_Learning\_Approach\_for\_Agricultural\_Land\_Cover\_Classification\_Using\_Satellite\_Imagery](https://www.researchgate.net/publication/397276494_Advancing_Precision_Agriculture_A_Deep_Learning_Approach_for_Agricultural_Land_Cover_Classification_Using_Satellite_Imagery)  
22. A machine-learning model to harmonize brain volumetric data for quantitative neuro-radiological assessment of Alzheimer's disease \- CSIRO Research Publications Repository, accessed January 1, 2026, [https://publications.csiro.au/publications/\#publication/PIcsiro:EP2024-0510](https://publications.csiro.au/publications/#publication/PIcsiro:EP2024-0510)  
23. Selecting validation sets based on information entropy \- MSSANZ, accessed January 1, 2026, [https://mssanz.org.au/modsim2023/files/nguyen221.pdf](https://mssanz.org.au/modsim2023/files/nguyen221.pdf)