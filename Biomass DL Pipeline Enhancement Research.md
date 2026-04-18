# **Deep Research Expedition: Revolutionizing the Deep Learning Pipeline for Biomass Mastery in the CSIRO Image2Biomass Competition**

## **1\. Introduction: The Intersection of Agronomy and Advanced Computer Vision**

The CSIRO Image2Biomass Kaggle competition represents a pivotal challenge at the convergence of precision agriculture and high-dimensional regression. The objective—estimating pasture biomass from top-down RGB imagery—addresses a critical bottleneck in global food security and sustainable land management. Traditionally, biomass estimation relies on the "clip and weigh" method, a destructive, labor-intensive, and spatially sparse process that fails to capture the heterogeneity of vast grazing landscapes.1 The competition seeks to replace this manual standard with automated, non-destructive deep learning pipelines capable of predicting five distinct biomass components: Dry\_Green\_g, Dry\_Dead\_g, Dry\_Clover\_g, GDM\_g (Green Dry Matter), and Dry\_Total\_g.1

For data scientists and machine learning engineers, this task transcends standard computer vision classification. It is a multi-output regression problem constrained by strict physical and biological hierarchies, complicating the optimization landscape. While "Stage 2" solutions utilizing baseline Vision Transformers (ViTs) or Convolutional Neural Networks (CNNs) such as ResNet or EfficientNet typically plateau around a weighted coefficient of determination ($R^2$) of 0.68, achieving the target score of 0.78+ requires a fundamental architectural paradigm shift. This report asserts that the barrier to higher performance is not merely a lack of model capacity, but a misalignment between the model architecture and the physical laws governing the data.

This comprehensive analysis deconstructs the path from a baseline DINO-based solution to a state-of-the-art "Stage 3" pipeline. The proposed strategy addresses three primary bottlenecks identified in the competition literature and dataset analysis: the mismanagement of hierarchical sum constraints ($A \+ B \= C$), the inefficient fusion of critical metadata (NDVI, rainfall, seasonality), and the inherent limitations of training large-scale transformers on a small, high-variance dataset of only 1,162 images.3 By transitioning to a "Compositional-Total" architecture underpinned by DINOv2 with register tokens, Feature-wise Linear Modulation (FiLM), and chemically consistent loss functions, we establish a robust framework for biomass mastery.

### **1.1 The Competition Metric and Strategic Priorities**

Understanding the evaluation metric is the first step in optimization. The competition utilizes a globally weighted $R^2$ score, computed over all image-target pairs. Crucially, the weights are non-uniform, reflecting the agronomic value of specific components.1

**Table 1: Target Variable Definitions and Evaluation Weights**

| Target Variable | Definition | Evaluation Weight | Agronomic Significance |
| :---- | :---- | :---- | :---- |
| **Dry\_Total\_g** | The sum of all dry biomass components. | **0.5** | Primary indicator of feed availability. |
| **GDM\_g** | Green Dry Matter (Green \+ Clover). | **0.2** | Indicator of high-quality, digestible feed. |
| **Dry\_Green\_g** | Green grass (excluding clover). | 0.1 | Base pasture component. |
| **Dry\_Dead\_g** | Senescent/dead material. | 0.1 | Low nutritional value; indicator of waste. |
| **Dry\_Clover\_g** | Legume component. | 0.1 | High protein source; fixes nitrogen. |

The weighting scheme reveals a clear strategic imperative: 70% of the final score is derived from aggregate measures (Dry\_Total\_g and GDM\_g) rather than specific species identification.1 A model that excels at distinguishing clover from grass but fails to estimate the total biomass volume will perform poorly. Conversely, a model that accurately predicts total biomass provides a massive stable foundation for the score. This insight drives the central architectural proposal of this report: prioritizing the regression of Dry\_Total\_g as a primary task, while treating the sub-components (Green, Dead, Clover) as compositional ratios derived from that total. This aligns the model's inductive bias with the scoring metric's sensitivity.5

### **1.2 The Dataset Constraints: Small Data, High Variance**

The dataset comprises 1,162 images collected from 19 locations across Australia, covering variable seasons and pasture types.3 This sample size is exceptionally small for training modern Vision Transformers, which typically require millions of examples to converge without overfitting. Furthermore, the data exhibits extreme variance. The coefficient of variation (CV) for Dry\_Dead\_g exceeds 1.0, indicating that the standard deviation is larger than the mean.4 This "zero-inflated" nature—where dead biomass is often exactly zero in lush spring pastures but dominates in drought conditions—wreaks havoc on standard Mean Squared Error (MSE) loss functions, which tend to regress to the mean.

The combination of limited data and high variance necessitates aggressive regularization and the use of pre-trained backbones that possess robust "all-purpose" features. DINOv2, trained on 142 million images using self-supervision, offers a solution, but its adaptation to regression requires careful handling of attention artifacts, discussed in Section 3\.7

## ---

**2\. The Biological and Mathematical Hierarchy**

The most significant failure mode of "Stage 2" baselines is the treatment of the five target variables as independent regression tasks. In reality, the targets form a closed physical system governed by additive constraints. Ignoring these constraints leads to predictions that are physically impossible—such as the sum of green and dead grass exceeding the total biomass—which introduces unnecessary variance and degrades the weighted $R^2$ score.

### **2.1 Defining the Constraints**

The dataset documentation explicitly defines the relationships between the variables.3 These are not correlations; they are definitions.

1. **The Green Constraint:** $GDM\\\_g \= Dry\\\_Green\\\_g \+ Dry\\\_Clover\\\_g$.  
2. **The Total Constraint:** $Dry\\\_Total\\\_g \= GDM\\\_g \+ Dry\\\_Dead\\\_g$.  
3. **The Grand Sum:** $Dry\\\_Total\\\_g \= Dry\\\_Green\\\_g \+ Dry\\\_Clover\\\_g \+ Dry\\\_Dead\\\_g$.

In a standard multi-output neural network, the final layer typically consists of 5 neurons, each predicting one target independently. Let $\\hat{y} \\in \\mathbb{R}^5$ be the output vector. The optimizer minimizes $\\sum (y\_i \- \\hat{y}\_i)^2$. However, there is no guarantee that $\\hat{y}\_{total} \= \\sum \\hat{y}\_{components}$. In fact, due to the stochastic nature of gradient descent, they will almost certainly disagree. This disagreement is "free energy" in the error landscape—noise that lowers the score. Since Dry\_Total\_g carries a 0.5 weight, any error propagated into it from a noisy Dry\_Dead\_g prediction (via a bottom-up summation approach) is catastrophic. Conversely, predicting Dry\_Total\_g independently and ignoring its components leads to inconsistencies with the 0.2-weighted GDM\_g.1

### **2.2 Compositional Data Analysis (CoDa) Theory**

Pasture biomass composition is fundamentally a problem of proportions. Once the total biomass $T$ is known, the remaining task is to distribute $T$ among the three atomic components: Green, Dead, and Clover. This structure maps to the simplex space used in Compositional Data Analysis (CoDa).9

Standard regression operates in Euclidean real space ($\\mathbb{R}^n$), but compositional data exists on the simplex $\\mathbb{S}^d$, defined as:

$$\\mathbb{S}^d \= \\{ \\mathbf{x} \= \\in \\mathbb{R}^D \\mid x\_i \> 0, \\sum\_{i=1}^D x\_i \= \\kappa \\}$$

where $\\kappa$ is the constant sum (in this case, 1.0 or 100%).  
Research indicates that direct regression of compositional parts (e.g., predicting grams of clover directly) violates the scale invariance principle of CoDa. If a paddock has 1000g of grass and 10% is clover, predicting "100g of clover" is less robust than predicting "1000g Total" and "0.1 Clover Ratio" separately.4 This is because the visual features corresponding to "cloverness" (leaf shape, texture) are invariant to the total biomass density, while the "total biomass" features (occlusion, height, shadows) are invariant to the species mix. Decoupling these tasks allows the neural network to specialize.

### **2.3 The "Compositional-Total" Architecture**

To satisfy the hierarchical constraints and leverage CoDa principles, the "Stage 3" architecture must abandon the flat 5-neuron head in favor of a structured, branched output.

#### **2.3.1 Mathematical Formulation**

We define the output of the network not as 5 scalars, but as one scalar magnitude and one probability vector.  
Let the network backbone $f\_\\theta(x)$ produce a latent representation $\\mathbf{h}$. We construct two parallel heads:

1. Magnitude Head ($H\_{mag}$):

   $$\\hat{T} \= \\text{ReLU}(H\_{mag}(\\mathbf{h}))$$

   This predicts Dry\_Total\_g directly. We use ReLU (or Softplus) to enforce non-negativity.  
2. Ratio Head ($H\_{ratio}$):

   $$\\mathbf{z} \= H\_{ratio}(\\mathbf{h}) \\in \\mathbb{R}^3$$  
   $$\\mathbf{p} \= \\text{Softmax}(\\mathbf{z})$$

   where $\\mathbf{p} \= \[\\hat{p}\_{green}, \\hat{p}\_{clover}, \\hat{p}\_{dead}\]$ such that $\\sum \\mathbf{p} \= 1$.  
3. Reconstruction Layer (Deterministic):  
   The final 5 targets are reconstructed deterministically, enforcing the sum constraints by definition:  
   * $\\hat{y}\_{Total} \= \\hat{T}$  
   * $\\hat{y}\_{Green} \= \\hat{T} \\times \\hat{p}\_{green}$  
   * $\\hat{y}\_{Clover} \= \\hat{T} \\times \\hat{p}\_{clover}$  
   * $\\hat{y}\_{Dead} \= \\hat{T} \\times \\hat{p}\_{dead}$  
   * $\\hat{y}\_{GDM} \= \\hat{y}\_{Green} \+ \\hat{y}\_{Clover} \= \\hat{T} \\times (\\hat{p}\_{green} \+ \\hat{p}\_{clover})$

This architecture reduces the degrees of freedom from 5 to 4 (1 magnitude \+ 3 logits constrained to sum to 1). This reduction acts as a powerful regularizer, particularly for the Dry\_Dead\_g component. As noted in 4, Dead biomass has a weak correlation with height ($R^2 \\approx 0.0025$) but a strong correlation with the Dead/Total ratio ($r=0.71$). Predicting the ratio is inherently easier for the network than predicting the absolute mass of the dead material.

### **2.4 Hierarchical Loss Formulation**

While the architecture enforces the constraints on the *predictions*, the loss function must still guide the gradient descent effectively. We propose a composite loss function that optimizes both the total magnitude and the component distributions simultaneously.

$$\\mathcal{L} \= \\lambda\_{total} \\mathcal{L}\_{reg}(\\hat{T}, y\_{total}) \+ \\lambda\_{comp} \\mathcal{L}\_{div}(\\mathbf{p}, \\mathbf{p}\_{gt})$$  
Where:

* $\\mathcal{L}\_{reg}$ is a robust regression loss (e.g., Huber or Log-Cosh) for the total biomass.  
* $\\mathcal{L}\_{div}$ is a divergence metric (e.g., Kullback-Leibler Divergence or Cross-Entropy) comparing the predicted ratios $\\mathbf{p}$ to the ground truth ratios $\\mathbf{p}\_{gt}$.  
* $\\lambda\_{total}$ and $\\lambda\_{comp}$ are hyperparameters balancing the two objectives. Given the competition weights, prioritizing $\\lambda\_{total}$ is advisable.

This formulation effectively transforms the regression problem into a multi-task learning problem: "How much grass is there?" (Regression) and "What kind of grass is it?" (Classification/Segmentation). This separation of concerns allows the model to learn distinct feature sets for density and texture, addressing the root cause of the Stage 2 plateau.

## ---

**3\. Backbone Engineering: Mastering DINOv2**

The choice of the visual backbone is the single most critical decision in the pipeline. "Stage 2" solutions typically employ standard pre-trained models like ResNet50 or ViT-Base. However, these models are trained on ImageNet-1k, a dataset dominated by object classification (dogs, cars, chairs). Pasture biomass estimation is a *texture* and *density* problem, not an object classification problem. The semantic gap between "identifying a golden retriever" and "estimating grams of dry ryegrass" is substantial.

To bridge this gap, we employ **DINOv2** (Distillation with No Labels v2), a state-of-the-art vision transformer trained via self-supervision on 142 million curated images.7 DINOv2 is specifically designed to learn robust, "all-purpose" visual features that capture geometric and textural properties without relying on human annotations.

### **3.1 DINOv2 Mechanics and Suitability**

DINOv2 utilizes a joint objective combining a discriminative self-distillation loss (DINO) and a masked image modeling loss (iBOT).8

* **Self-Distillation:** Forces a "student" network to match the output of a "teacher" network (an exponential moving average of the student) on different crops of the same image. This enforces local-to-global consistency, crucial for understanding that a small patch of grass implies the properties of the whole paddock.  
* **Masked Image Modeling (iBOT):** Masks out patches of the image and forces the model to reconstruct the features of the missing patches. This forces the model to learn the spatial continuity of textures—exactly what is needed to infer biomass in occluded areas of the canopy.10

These properties make DINOv2 features significantly more discriminative for dense regression tasks than supervised baselines. However, naive implementation of DINOv2 leads to specific artifacts that must be addressed.

### **3.2 The Artifact Problem: Register Tokens**

Standard Vision Transformers, including DINOv1, suffer from "attention artifacts." Research has shown that these models tend to repurpose low-information background tokens to store global information (such as scene brightness or position).11 These tokens manifest as high-norm "outliers" in the feature maps, often appearing in random background patches.

In a classification task, these artifacts are harmless because the \`\` token aggregates information globally. However, for biomass estimation, we need to perform dense regression, often averaging features across the spatial grid to estimate density. If the feature map contains high-norm artifact tokens, they will disproportionately skew the average, leading to massive overestimation of biomass in sparse areas.

The Solution: Register Tokens.  
DINOv2 addresses this by introducing explicit "Register Tokens" (typically 4\) appended to the input sequence.8 These are learnable tokens that serve as designated "sinks" for global information. The model learns to discard artifacts into these registers, leaving the image patch tokens ($N\_{patches}$) clean and focused solely on local visual information.

* **Implementation Requirement:** Use the vit\_large\_patch14\_reg4\_dinov2 variant available in timm or transformers.12  
* Feature Extraction: During the forward pass, extract the sequence of tokens. Discard the 4 register tokens. Use the clean patch tokens for spatial pooling (Average Pooling) and the \`\` token for global context.

  $$\\mathbf{F}\_{spatial} \= \\frac{1}{N} \\sum\_{i=1}^{N} \\mathbf{t}\_{patch}^{(i)} \\quad (\\text{Excluding Registers})$$

### **3.3 Fine-Tuning Strategy: Low-Rank Adaptation (LoRA)**

Fine-tuning a ViT-Large (300M+ parameters) on 1,162 images is a recipe for catastrophic overfitting. Freezing the backbone entirely, however, prevents the model from adapting to the specific spectral characteristics of Australian pasture (e.g., distinguishing "dry clover" from "dead grass," which look spectrally similar).

LoRA (Low-Rank Adaptation) offers the optimal middle ground.14 LoRA freezes the pre-trained weights $W\_0 \\in \\mathbb{R}^{d \\times k}$ and injects two low-rank matrices $A \\in \\mathbb{R}^{d \\times r}$ and $B \\in \\mathbb{R}^{r \\times k}$ (where $r \\ll d$) into the specific layers (typically the Query and Value projections of the Attention mechanism).

$$W\_{new} \= W\_0 \+ \\Delta W \= W\_0 \+ B A$$

For this competition, setting a rank $r=8$ or $r=16$ reduces the trainable parameter count to less than 1% of the total model size. This allows the attention mechanism to re-orient itself towards pasture features (learning to attend to "clover patterns" rather than "dog ears") without destroying the robust general-purpose features learned during pre-training. This is essential for the "Stage 3" score jump, as it enables domain adaptation without the data requirements of full fine-tuning.14

### **3.4 Resolution Engineering**

The standard DINOv2 pre-training resolution is 518x518 pixels.13 Many baselines resize images to 224x224 for speed. In biomass estimation, texture is the primary signal. Resizing a 518px image of grass to 224px destroys the high-frequency information (leaf edges, stem thickness) required to distinguish species and estimate height visually.

* **Recommendation:** Maintain the native 518x518 resolution. This results in a $37 \\times 37$ grid of patch tokens (patch size 14). This dense grid provides 1,369 local density estimates per image, compared to only 256 estimates at 224x224. This 5x increase in spatial granularity is directly correlated with regression accuracy in texture-based tasks.15

## ---

**4\. Multimodal Fusion: Beyond Concatenation**

The Kaggle dataset provides critical metadata alongside the images: NDVI (Normalized Difference Vegetation Index), Height\_Ave\_cm, State, and Season.1 In "Stage 2" models, these are typically handled via "Late Fusion"—concatenating the metadata to the image features just before the final regression head.

**The Flaw of Late Fusion:** In Late Fusion, the visual encoder processes the image in isolation. It effectively "sees" the image without knowing the context. For example, a patch of brown pixels could be "dead grass" (biomass) or "bare soil" (no biomass). Without knowing the NDVI or Season *during* the visual processing, the encoder cannot disambiguate these textures, leading to ambiguous features.16 To reach 0.78+, we must implement "Early" or "Mid" fusion via **Conditioning**.

### **4.1 Feature-wise Linear Modulation (FiLM)**

FiLM is a mechanism that allows scalar and categorical metadata to modulate the internal feature maps of the neural network.17 Instead of simply appending metadata, FiLM uses it to scale and shift the visual features.  
Mathematically, for a feature map $\\mathbf{F}\_{c}$ (channel $c$), FiLM applies an affine transformation:

$$\\text{FiLM}(\\mathbf{F}\_{c} \\mid \\gamma\_c, \\beta\_c) \= \\gamma\_c(\\mathbf{z}) \\cdot \\mathbf{F}\_{c} \+ \\beta\_c(\\mathbf{z})$$

where $\\gamma\_c$ and $\\beta\_c$ are coefficients generated by a separate MLP (the "Conditioning Network") that takes the metadata vector $\\mathbf{z}$ as input.  
**Mechanism in Biomass:**

* **Scale ($\\gamma$):** If NDVI is high, the Conditioning Network might output a high $\\gamma$ for the "green texture" channels of the DINOv2 features, amplifying the signal of living biomass. If Season is "Summer," it might suppress the "green" channels and amplify "brown/dry" channels.  
* **Shift ($\\beta$):** Can act as a bias correction based on State (e.g., correcting for different soil background colors in WA vs NSW).

### **4.2 Adaptive Layer Normalization (AdaLN)**

A modern evolution of FiLM, popularized by Diffusion Transformers (DiT) and Generative models, is Adaptive Layer Normalization (AdaLN).18 AdaLN replaces the standard learnable affine parameters of LayerNorm with regressed values from the conditioning embedding.

$$\\text{AdaLN}(x, \\mathbf{c}) \= \\text{LayerNorm}(x) \\cdot (1 \+ \\gamma(\\mathbf{c})) \+ \\beta(\\mathbf{c})$$

Unlike post-hoc FiLM layers, AdaLN integrates the conditioning directly into the transformer blocks. For the "Stage 3" pipeline, inserting a custom AdaLN-modulated adapter block after the frozen DINOv2 backbone is the most efficient way to inject metadata. This allows the metadata to fundamentally alter the feature distribution before the regression head sees it.

### **4.3 Handling Categorical Metadata: Embeddings**

The variables State (e.g., NSW, WA) and Season (Winter, Spring) are categorical.

* **Problem with One-Hot:** One-hot encoding creates sparse, orthogonal vectors. It assumes "Winter" is as different from "Autumn" as it is from "Summer." It fails to capture the ordinal transitions or the geographic similarities between states (e.g., NSW and Victoria might be similar, while WA is distinct).21  
* Solution: Learnable Embeddings. Map each category to a dense vector (e.g., dimension $d=4$ or $8$).

  $$\\mathbf{e}\_{season} \= \\text{Embedding}(Season\_{idx})$$  
  $$\\mathbf{e}\_{state} \= \\text{Embedding}(State\_{idx})$$

  These embeddings are concatenated with the normalized scalar metadata (NDVI, Height) to form the conditioning vector $\\mathbf{z}$ passed to the FiLM/AdaLN generators. This allows the model to learn, for example, that the latent representation of "Autumn" lies between "Summer" and "Winter" in the embedding space, enabling better interpolation for transition periods.22

### **4.4 Robust Scaling for Scalars**

The scalar metadata (Height, NDVI) often contains outliers. Using StandardScaler (z-score) can be sensitive to these extremes.

* **Recommendation:** Use RobustScaler (from Scikit-Learn), which scales data based on the Interquartile Range (IQR). This ensures that the Conditioning Network is not destabilized by a few extreme height measurements, maintaining a stable $\\gamma$ and $\\beta$ modulation.23

## ---

**5\. Advanced Loss Functions and Optimization**

With the architecture defined, the final piece of the puzzle is the optimization strategy. The small dataset size and high target variance require specialized loss functions beyond MSE.

### **5.1 Tweedie Loss for Zero-Inflated Targets**

The Dry\_Dead\_g target is "zero-inflated." In many samples, it is exactly zero. In others, it is continuous. Standard MSE assumes a Gaussian distribution, which is symmetric and defined over $(-\\infty, \\infty)$. When trained on zero-inflated non-negative data, MSE models tend to "hedge" their bets, predicting a small positive constant (e.g., 5g) for true zeros to minimize average error. This ruins the accuracy for clean pastures.4

**Tweedie Loss** is derived from the Tweedie distribution (a compound Poisson-Gamma process) which naturally models non-negative data with a probability mass at zero.24

$$\\mathcal{L}\_{Tweedie}(y, \\hat{y}) \= \-y \\frac{\\hat{y}^{1-p}}{1-p} \+ \\frac{\\hat{y}^{2-p}}{2-p}$$

where $p \\in (1, 2)$ is the power parameter. As $p \\to 1$, it behaves like Poisson (count data). As $p \\to 2$, it behaves like Gamma (continuous data).

* **Application:** Apply Tweedie Loss (with $p \\approx 1.5$) specifically to the Dry\_Dead\_g (or Dead Ratio) output of the model. This encourages the model to predict *exact zeros* where appropriate, capturing the sparsity of the dead biomass distribution.

### **5.2 Manifold Mixup for Regularization**

Data augmentation is essential for small datasets. Standard "Input Mixup" (blending pixels of two images) is effective for classification but problematic for regression, as blended "ghost" textures may not map linearly to blended biomass values.  
Manifold Mixup applies the mixing operation in the latent feature space.26

1. Feed Image A and Image B into the DINOv2 backbone.  
2. Extract feature vectors $\\mathbf{f}\_A, \\mathbf{f}\_B$.  
3. Generate a random mixing coefficient $\\lambda \\sim \\text{Beta}(\\alpha, \\alpha)$.  
4. Create mixed feature: $\\mathbf{f}\_{mix} \= \\lambda \\mathbf{f}\_A \+ (1-\\lambda) \\mathbf{f}\_B$.  
5. Create mixed target: $y\_{mix} \= \\lambda y\_A \+ (1-\\lambda) y\_B$.  
6. Train the regression head on $(\\mathbf{f}\_{mix}, y\_{mix})$.

This technique forces the regression head to be linear in the semantic feature space. It fills the gaps in the sparse training manifold, effectively synthesizing infinite "virtual" pastures that lie semantically between the real training examples. This is significantly more robust than pixel-level mixup for regression tasks.28

### **5.3 Soft-Constraint Penalty (Lagrangian Relaxation)**

While the "Compositional-Total" architecture enforces hard constraints, we can further guide the early training dynamics using a soft-constraint penalty in the loss function.29 This is particularly useful if experimenting with non-compositional baselines.  
$$ \\mathcal{L}\_{cons} \= |  
| \\hat{y}{GDM} \- (\\hat{y}{Green} \+ \\hat{y}\_{Clover}) ||^2 $$  
However, as noted in research 31, soft constraints can sometimes conflict with the primary loss if the weighting is not tuned perfectly. The architectural constraint (Section 2.3) is the preferred method for the final "Stage 3" model.

## ---

**6\. The "Stage 3" Pipeline: Detailed Implementation**

Based on the preceding analysis, we define the complete "Stage 3" pipeline. This architecture is designed to directly address the 0.68 $\\to$ 0.78 gap.

### **6.1 Data Pipeline**

* **Preprocessing:** Images kept at 518x518. Metadata State/Season Label Encoded. Height/NDVI RobustScaled.  
* **Stratified Validation:** Stratified K-Fold (k=5) based on State and binned Dry\_Total\_g. This prevents "easy" folds where train/val distributions are identical, ensuring the model generalizes to new geographic regions.22  
* **Augmentation:** Horizontal/Vertical Flip, Random Rotate (90/180/270), Color Jitter (low intensity). **Manifold Mixup** enabled ($\\alpha=0.4$) during training.

### **6.2 Model Architecture Specification**

| Component | Specification | Rationale |
| :---- | :---- | :---- |
| **Backbone** | dinov2\_vit\_large\_patch14\_reg4 | SOTA texture features; Registers remove artifacts.13 |
| **Adaptation** | LoRA (Rank 8\) on Q, V proj | Fine-tunes attention without overfitting small data.14 |
| **Feature Pooling** | Concat(\`\`, AvgPool(Patches)) | Captures both global context and local density.8 |
| **Fusion Layer** | AdaLN-Zero Adapter | Injects metadata (NDVI, Height, State, Season) directly into feature stream.20 |
| **Head Architecture** | **Compositional-Total** | Separation of Magnitude (Total) and Proportion (Species) tasks. |
| **Magnitude Head** | MLP $\\to$ ReLU $\\to$ Dry\_Total\_g | Predicts the most stable, highest-weighted target (0.5). |
| **Ratio Head** | MLP $\\to$ Softmax $\\to$ $$ | Enforces $\\sum p \= 1$ constraint physically. |
| **Reconstruction** | Deterministic Math | $\\hat{y}\_{i} \= \\hat{T} \\times \\hat{p}\_{i}$. Guarantees consistency. |

### **6.3 Optimization Protocol**

* Loss Function:

  $$\\mathcal{L} \= 0.6 \\cdot \\text{Huber}(\\hat{T}, y\_{Total}) \+ 0.2 \\cdot \\text{KLDiv}(\\mathbf{p}, \\mathbf{p}\_{gt}) \+ 0.2 \\cdot \\text{Tweedie}(\\hat{y}\_{Dead}, y\_{Dead})$$  
  * *Note:* The Tweedie term on the reconstructed Dead mass ensures zero-inflation is handled, while the KL Divergence optimizes the species mixing ratios.  
* **Optimizer:** AdamW with Weight Decay 0.05 (decoupled).  
* **Learning Rate:** $1e-4$ for Head/Adapter, $5e-6$ for LoRA Backbone.  
* **Scheduler:** Cosine Annealing with Warmup (5 epochs).

### **6.4 Expected Performance Impact**

* **DINOv2 \+ Registers:** \+0.03-0.05 $R^2$ (Cleaner features, better texture resolution).  
* **Compositional Head:** \+0.03-0.04 $R^2$ (Eliminates impossible predictions, stabilizes Total).  
* **FiLM/AdaLN Metadata:** \+0.02-0.03 $R^2$ (Contextualizes visual ambiguity).  
* **Manifold Mixup:** \+0.01-0.02 $R^2$ (Generalization gap reduction).  
* **Total Projected Gain:** \~0.10 $R^2$ (0.68 $\\to$ 0.78).

## **7\. Conclusion**

The transition from a baseline 0.68 score to a winning 0.78+ in the CSIRO Image2Biomass competition is not achieved by simply training longer or adding more layers. It requires a holistic re-engineering of the pipeline to respect the physical reality of the data. By adopting the **Compositional-Total** architecture, we enforce the immutable hierarchical constraints of biomass accumulation. By integrating **DINOv2 with Register Tokens**, we leverage the world's most advanced self-supervised visual features while mitigating artifact noise. By utilizing **FiLM/AdaLN** and **Tweedie Loss**, we account for the critical metadata context and the zero-inflated statistical properties of dead biomass. This rigorous, physics-informed approach transforms the model from a black-box pattern matcher into a robust scientific instrument, capable of mastering the complex biological signals of Australian pastures.

#### **Works cited**

1. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed January 4, 2026, [https://www.kaggle.com/competitions/csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass)  
2. MLA, CSIRO & Google: Kaggle Competition \- Image2Biomass Prediction \- grow AG., accessed January 4, 2026, [https://www.growag.com/opportunity/mla-csiro-google-kaggle-competition-image2biomass-prediction](https://www.growag.com/opportunity/mla-csiro-google-kaggle-competition-image2biomass-prediction)  
3. Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture, accessed January 4, 2026, [https://arxiv.org/html/2510.22916v1](https://arxiv.org/html/2510.22916v1)  
4. Analysis: Height\_Ave\_cm and Dead Biomass Prediction \- Testing the Host's Hypothesis \- CSIRO \- Image2Biomass Prediction | Kaggle, accessed January 4, 2026, [https://www.kaggle.com/competitions/csiro-biomass/discussion/650736](https://www.kaggle.com/competitions/csiro-biomass/discussion/650736)  
5. Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture, accessed January 4, 2026, [https://www.researchgate.net/publication/396968064\_Estimating\_Pasture\_Biomass\_from\_Top-View\_Images\_A\_Dataset\_for\_Precision\_Agriculture](https://www.researchgate.net/publication/396968064_Estimating_Pasture_Biomass_from_Top-View_Images_A_Dataset_for_Precision_Agriculture)  
6. Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture, accessed January 4, 2026, [https://chatpaper.com/paper/204088](https://chatpaper.com/paper/204088)  
7. DinoV2 Fine-Tuning Tutorial: How to Maximize Accuracy for Computer Vision Tasks, accessed January 4, 2026, [https://kili-technology.com/blog/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks](https://kili-technology.com/blog/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks)  
8. dinov2/MODEL\_CARD.md at main \- GitHub, accessed January 4, 2026, [https://github.com/facebookresearch/dinov2/blob/main/MODEL\_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)  
9. Adaptation of Compositional Data Analysis in Deep Learning to Predict Pasture Biomass Proportions \- ResearchGate, accessed January 4, 2026, [https://www.researchgate.net/publication/367284604\_Adaptation\_of\_Compositional\_Data\_Analysis\_in\_Deep\_Learning\_to\_Predict\_Pasture\_Biomass\_Proportions](https://www.researchgate.net/publication/367284604_Adaptation_of_Compositional_Data_Analysis_in_Deep_Learning_to_Predict_Pasture_Biomass_Proportions)  
10. DINOv2 \- Hugging Face, accessed January 4, 2026, [https://huggingface.co/docs/transformers/v4.46.0/model\_doc/dinov2](https://huggingface.co/docs/transformers/v4.46.0/model_doc/dinov2)  
11. Vision Transformers Don't Need Trained Registers \- arXiv, accessed January 4, 2026, [https://arxiv.org/html/2506.08010v5](https://arxiv.org/html/2506.08010v5)  
12. timm/vit\_base\_patch14\_reg4\_dinov2.lvd142m · How to get register token output values ?, accessed January 4, 2026, [https://huggingface.co/timm/vit\_base\_patch14\_reg4\_dinov2.lvd142m/discussions/4](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m/discussions/4)  
13. timm/vit\_large\_patch14\_reg4\_dinov2.lvd142m \- Hugging Face, accessed January 4, 2026, [https://huggingface.co/timm/vit\_large\_patch14\_reg4\_dinov2.lvd142m](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m)  
14. DINOv2-Based Approaches Overview \- Emergent Mind, accessed January 4, 2026, [https://www.emergentmind.com/topics/dinov2-based-approaches](https://www.emergentmind.com/topics/dinov2-based-approaches)  
15. Upsampling DINOv2 features for unsupervised vision tasks and weakly supervised materials segmentation \- arXiv, accessed January 4, 2026, [https://arxiv.org/html/2410.19836v2](https://arxiv.org/html/2410.19836v2)  
16. HyperPointFormer: Multimodal Fusion in 3D Space with Dual-Branch Cross-Attention Transformers \- arXiv, accessed January 4, 2026, [https://arxiv.org/html/2505.23206v1](https://arxiv.org/html/2505.23206v1)  
17. FiLM: Visual Reasoning with a General Conditioning Layer \- arXiv, accessed January 4, 2026, [http://arxiv.org/pdf/1709.07871](http://arxiv.org/pdf/1709.07871)  
18. StreamDiT: Real-Time Text-to-Video Model, accessed January 4, 2026, [https://www.emergentmind.com/topics/streamdit](https://www.emergentmind.com/topics/streamdit)  
19. Post-Training Quantization for Diffusion Transformer via Hierarchical Timestep Grouping, accessed January 4, 2026, [https://arxiv.org/html/2503.06930](https://arxiv.org/html/2503.06930)  
20. Full article: S2TDM: Spatial-spectral transformer-based diffusion model for hyperspectral image denoising \- Taylor & Francis Online, accessed January 4, 2026, [https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2591277](https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2591277)  
21. Encoding Categorical Variables Explained \- Medium, accessed January 4, 2026, [https://medium.com/@pacosun/decoding-the-code-categorical-variables-without-the-chaos-66ec11c97f2d](https://medium.com/@pacosun/decoding-the-code-categorical-variables-without-the-chaos-66ec11c97f2d)  
22. Exploring Hierarchical Blending in Target Encoding \- Towards Data Science, accessed January 4, 2026, [https://towardsdatascience.com/exploring-hierarchical-blending-in-target-encoding-fea4c59b305b/](https://towardsdatascience.com/exploring-hierarchical-blending-in-target-encoding-fea4c59b305b/)  
23. What's the Best approach to handle Regression data ? | Kaggle, accessed January 4, 2026, [https://www.kaggle.com/discussions/questions-and-answers/461865](https://www.kaggle.com/discussions/questions-and-answers/461865)  
24. Training Procedure \- arXiv, accessed January 4, 2026, [https://arxiv.org/html/2509.26223v1](https://arxiv.org/html/2509.26223v1)  
25. Anchor-MoE: A Mean-Anchored Mixture of Experts For Probabilistic Regression \- arXiv, accessed January 4, 2026, [https://arxiv.org/pdf/2508.16802?](https://arxiv.org/pdf/2508.16802)  
26. R-Mixup: Riemannian Mixup for Biological Networks \- PMC \- NIH, accessed January 4, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10853987/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10853987/)  
27. Harnessing Hard Mixed Samples with Decoupled Regularizer \- Cheng Tan, accessed January 4, 2026, [https://chengtan9907.github.io/assets/publications/nips23\_decoupled\_mixup.pdf](https://chengtan9907.github.io/assets/publications/nips23_decoupled_mixup.pdf)  
28. First-Order Manifold Data Augmentation for Regression Learning \- GitHub, accessed January 4, 2026, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/kaufman24a/kaufman24a.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/kaufman24a/kaufman24a.pdf)  
29. DeepSaDe: Learning Neural Networks That Guarantee Domain Constraint Satisfaction, accessed January 4, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/29109/30097](https://ojs.aaai.org/index.php/AAAI/article/view/29109/30097)  
30. Constrained Loss Function \- autograd \- PyTorch Forums, accessed January 4, 2026, [https://discuss.pytorch.org/t/constrained-loss-function/123812](https://discuss.pytorch.org/t/constrained-loss-function/123812)  
31. MSRA-Net: A Multi-Task Learning Model for Soil Texture Prediction with Dynamic Weighting and Prior Knowledge Soft Constraints \- MDPI, accessed January 4, 2026, [https://www.mdpi.com/1424-8220/25/21/6519](https://www.mdpi.com/1424-8220/25/21/6519)