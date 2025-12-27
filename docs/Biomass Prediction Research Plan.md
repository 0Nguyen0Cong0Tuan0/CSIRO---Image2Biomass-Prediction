# **Methodological Synthesis for Precision Biomass Estimation: A Deep Learning Framework for the CSIRO Image2Biomass Challenge**

## **1\. Introduction and Problem Contextualization**

The quantification of pasture biomass stands as a pivotal challenge in modern precision agriculture, serving as the fundamental metric for determining stocking rates, optimizing grazing rotations, and ensuring long-term ecological sustainability. The **CSIRO Image2Biomass Prediction** competition represents a significant benchmark in this domain, moving beyond traditional, labor-intensive methodologies towards scalable, computer vision-based solutions. This report provides an exhaustive analysis of the problem space, grounded in a rigorous examination of the dataset's statistical properties, the biological constraints of the target variables, and the specific requirements of the competition's evaluation framework.

The objective is not merely to train a regression model but to construct a **biophysically consistent, statistically robust inference system** capable of handling zero-inflated continuous targets, domain shifts, and strictly hierarchical data structures. The current standard—"clip and weigh"—is destructive and unscalable, while rising plate meters often fail to account for species composition or "hidden" dead matter.1 By leveraging the 1,162 annotated top-view images provided in the dataset 3, we aim to develop a model that predicts five distinct biomass components: **Dry Green vegetation ($y_\text{green}$)**, **Dry Dead material ($y_\text{dead}$)**, **Dry Clover biomass ($y_\text{clover}$)**, **Green Dry Matter ($y_\text{GDM}$)**, and **Total Dry Biomass ($y_\text{total}$)**.

Crucially, this analysis identifies that the central challenge lies in the **zero-inflated nature** of specific components (clover and dead matter) and the **hierarchical dependencies** where component sums must equal aggregates ($y_\text{total} = y_\text{green} + y_\text{clover} + y_\text{dead}$). Standard deep learning regression approaches (e.g., ResNet with MSE loss) are theoretically insufficient for this manifold. This report proposes a sophisticated framework utilizing **Tweedie Loss functions**, **Student-Teacher distillation** to leverage hidden training metadata, and **Hard-Constraint Output Layers** to ensure biological consistency.

## **2\. Comprehensive Data Audit and Statistical Pathology**

To engineer a "perfect" solution, one must first dissect the underlying data distribution and the specific anomalies present in the training set. The dataset spans 19 locations across four Australian states (Tasmania, Victoria, New South Wales, Western Australia), introducing significant spatiotemporal variance.3

### **2.1 The Hierarchical Target Structure**

The five target variables are not independent regressands; they form a strict additive hierarchy. Understanding this is non-negotiable for model design.

| Target Variable | Biological Definition | Statistical Distribution Characteristics | Relationship Constraint |
| :---- | :---- | :---- | :---- |
| **Dry\_Green\_g** | Photosynthetically active grass/forbs (excluding legumes) | Continuous, Gamma-like. Strictly positive in viable pastures. | Primary component of $y_\text{GDM}$ |
| **Dry\_Clover\_g** | Legume biomass (nitrogen-fixing) | **Zero-Inflated Continuous**. Highly right-skewed. | Component of $y_\text{GDM}$ |
| **Dry\_Dead\_g** | Senescent, necrotic, or detached material | **Zero-Inflated Continuous**. High variance. **State-dependent anomaly**. | Component of $y_\text{total}$ |
| **GDM\_g** | Green Dry Matter | Continuous. Sum of Green and Clover. | $y_\text{GDM} = y_\text{green} + y_\text{clover}$ |
| **Dry\_Total\_g** | Total Biomass | Continuous. Sum of GDM and Dead. | $y_\text{total} = y_\text{GDM} + y_\text{dead}$ |

Analysis of Dependencies: The explicit definitions of $y_\text{GDM}$ and $y_\text{total}$ introduce hard constraints.4 A naive multi-head regression model that predicts these five targets independently will almost certainly violate the consistency rule:

$$\hat{y}_\text{total} \neq \hat{y}_\text{green} + \hat{y}_\text{clover} + \hat{y}_\text{dead}$$

Such violations not only degrade trust in the system (predicting more total biomass than the sum of its parts is physically impossible) but also arguably penalize the model in the evaluation metric, which heavily weights the Total and GDM aggregates.

### **2.2 The "Western Australia" Dead Matter Anomaly**

A granular inspection of the training data reveals a critical statistical anomaly regarding the Dry\_Dead\_g target. In the snippet provided for Western Australia (WA) samples (e.g., IDs ID1025234388, ID1051144034), the target value for Dry\_Dead\_g is consistently **0.0**.4

* **Observation:** Every visible sample from the WA state has exactly zero dead biomass, regardless of the NDVI or Height values. In contrast, samples from Tasmania (Tas) and Victoria (Vic) show significant quantities of dead matter (e.g., 31.99g for ID1011485656).  
* **Implication:** This suggests a systematic bias in data collection or phenology. Either the WA sampling campaigns were conducted exclusively during early growth stages (before senescence), or the protocol for WA specifically excluded the collection of dead litter.5  
* **Modeling Risk:** A model trained on the full dataset without state-awareness will learn to predict non-zero dead matter based on visual cues (e.g., brown soil or dry stalks) that might be present in WA images. Applying a "global" visual model to WA test samples would likely result in systematic over-prediction of Dry\_Dead\_g, incurring significant error.  
* **Strategy:** This necessitates a **conditional post-processing rule** or a **State-Aware Feature Injection**. If the inference pipeline can detect or infer that an image originates from WA, the Dry\_Dead\_g prediction should be clamped to zero.

### **2.3 The "Privileged Information" Paradox**

The training set includes high-value metadata:

1. **Pre\_GSHH\_NDVI:** A spectral index highly correlated with green biomass ($R \\approx 0.8$ for green, but saturates \> 3t/ha).6  
2. **Height\_Ave\_cm:** A structural metric correlated with volume and total biomass.

Constraint: This metadata is absent from the test set and submission requirements.2 The model must predict biomass solely from the RGB image path.  
This creates a classic Learning Using Privileged Information (LUPI) scenario. While we cannot use NDVI as an input during testing, ignoring it during training would be wasteful. The correlation between NDVI and visual "greenness" is causal. Therefore, the training strategy must involve mechanisms (like Knowledge Distillation or Auxiliary Tasks) to force the visual encoder to learn features that correlate with these hidden metadata variables.

### **2.4 Statistical Properties of "Dead" and "Clover"**

The Dry\_Dead\_g and Dry\_Clover\_g targets exhibit **Zero-Inflation**.8

* **Clover:** Many pastures are monocultures (pure grass). In these cases, clover mass is exactly zero.  
* **Dead:** Due to the WA anomaly and seasonal factors, dead mass is often zero.  
* **Coefficient of Variation (CV):** Dry\_Dead\_g has the second-highest CV (approx 1.03), and Dry\_Clover\_g has the highest (approx 1.82).5 This implies the standard deviation is higher than the mean, indicating extreme volatility.  
* **Prediction Challenge:** Standard Mean Squared Error (MSE) loss is ill-suited here. Minimizing MSE on a zero-inflated target leads to a model that predicts the mean (a small positive number) for all zero samples, failing to capture the sparsity. We require loss functions that can handle probability masses at zero, such as **Tweedie Loss** or **Zero-Inflated Gamma (ZIG)** distributions.

## **3\. Evaluation Metric Deconstruction: The Globally Weighted $R^2$**

The competition utilizes a specific variant of the coefficient of determination: **Globally Weighted $R^2$**.10

$$R^2_{weighted} = 1 - \frac{\sum_{i} \sum_{j} w_j (y_{ij} - \hat{y}_\text{ij})^2}{\sum_{i} \sum_{j} w_j (y_\text{ij} - \bar{y}_{w})^2}$$

where $w = [0.1, 0.1, 0.1, 0.2, 0.5]$ for.  
**Critical Nuances:**

1. **Dominance of Total Biomass:** Dry\_Total\_g accounts for **50%** of the weight. Furthermore, since $y_\text{total}$ is the sum of other components, its variance (and thus its contribution to the denominator $\sum (y - \bar{y})^2$) is naturally much larger than the components. This effectively means the competition is primarily about predicting **Total Biomass**. The component predictions are secondary, serving mostly as regularizers.  
2. **Global Mean Denominator:** Unlike the standard sklearn r2\_score which calculates $1 - \frac{SS_{res}}{SS_{tot}}$ per column and then averages, this metric concatenates all targets into one long vector. The denominator is the variance of the *entire weighted dataset* relative to the *global weighted mean*.11  
   * *Implication:* A model that predicts different constant values for each target can theoretically achieve a score $> 0$ (unlike standard $R^2$ where a constant prediction yields 0).  
   * Optimization: We should optimize for Weighted MSE. Since the denominator is a constant (property of the ground truth), maximizing $R^2$ is mathematically equivalent to minimizing the weighted sum of squared errors:

     $$\mathcal{L}_{opt} = 0.5(y_{tot} - \hat{y}_{tot})^2 + 0.2(y_{gdm} - \hat{y}_{gdm})^2 + 0.1 \sum_{c \in \{gr, cl, dd\}} (y_c - \hat{y}_c)^2$$

## **4\. Theoretical Framework: Zero-Inflated Regression**

To handle the semi-continuous nature of Dry\_Clover and Dry\_Dead (exact zeros mixed with continuous positive values), we must move beyond Gaussian assumptions.

### **4.1 The Tweedie Distribution ($1 < p < 2$)**

The Tweedie distribution is an exponential dispersion model that acts as a compound Poisson-Gamma process.12

* **Concept:** It models biomass as a sum of $N$ "clumps", where $N \sim Poisson(\lambda)$.  
  * If $N=0$, Mass = 0 (Exact Zero).  
  * If $N > 0$, Mass $\sim Gamma(N\alpha, \beta)$.  
* **Suitability:** This is the "perfect" theoretical fit for pasture components. Clover biomass is essentially the sum of discrete clover plants (clumps) in the quadrat. If no plants are present, mass is zero.  
* Loss Function: The Tweedie deviance loss is defined as:

  $$\mathcal{L}_{Tweedie}(y, \hat{y}, p) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\hat{y}^{1-p}}{1-p} + \frac{\hat{y}^{2-p}}{2-p} \right)$$
  For $1 < p < 2$, this loss is differentiable and convex.  
* **Recommendation:** Use $p \approx 1.5$ for Dry\_Clover and Dry\_Dead to effectively model the zero-inflation. Use $p \approx 1.99$ (approaching Gamma) or Log-MSE for Dry\_Green and Dry\_Total, which are strictly positive.

### **4.2 Zero-Inflated Gamma (ZIG) / Hurdle Models**

An alternative is a **Two-Stage Hurdle Model**.9

1. **Binary Classifier (The Hurdle):** Predicts probability of presence $\pi = P(y > 0)$. Loss: Binary Cross Entropy.  
2. **Regressor:** Predicts quantity $\mu = E[y | y > 0]$. Loss: Gamma NLL or Log-Normal.  
3. **Combination:** $\hat{y} = \pi \cdot \mu$.  
* **Pros:** Explicitly disentangles the "is it there?" question from "how much?" This is useful if the mechanism for presence (e.g., sowing) differs from growth (e.g., rainfall).  
* **Cons:** Harder to train end-to-end. The regressor only receives gradients from non-zero samples, reducing the effective training set size for the regression head.  
* **Decision:** Given the dataset size (1,162), the **Tweedie Loss** is superior as it uses all samples to update weights, providing more stable convergence than the two-stage Hurdle model.12

## **5\. Proposed Architecture: The Multi-Modal Distillation Network**

We propose a bespoke architecture, **BioFusionNet**, designed to exploit visual texture, enforce hierarchical constraints, and leverage hidden metadata.

### **5.1 Visual Backbone: Texture-Biased Encoders**

Pasture estimation relies heavily on **texture** (density of leaves, grass blades vs. clover leaves) rather than object shape.

* **Primary Encoder: DINOv2 (ViT-Giant):** Self-supervised Vision Transformers like DINOv2 are state-of-the-art for texture and dense prediction tasks.16 They possess an inherent understanding of granular features without needing label supervision.  
  * *Implementation:* Use a **Frozen DINOv2 Backbone** with a trainable **LoRA (Low-Rank Adaptation)** or lightweight adapter. This prevents overfitting on the small dataset while leveraging the massive pre-training.  
* **Secondary Encoder: EfficientNetV2-L:** Convolutional Networks (CNNs) capture local spatial hierarchies well. EfficientNetV2 offers a robust baseline.17  
* **Fusion:** Concatenate the global pool tokens ($$) from DINOv2 and the Global Average Pooling from EfficientNetV2.

### **5.2 The Hierarchical Constraint Head**

Instead of predicting 5 targets directly, we must encode the physics of the problem into the network graph.18

* **Structure:**  
  * The network predicts 3 components: $\\hat{y}\_{green}$, $\\hat{y}\_{clover}$, $\\hat{y}\_{dead}$.  
  * It *also* predicts 1 aggregate: $\\hat{y}\_{total\\\_raw}$.  
* Reconciliation Layer (Forward Pass):  
  We use a Softmax-based Projection or Geometric Reconciliation to enforce the sum.  
  Let $\hat{S} = \hat{y}_{green} + \hat{y}_{clover} + \hat{y}_{dead}$.  
  The final component predictions are adjusted:

  $$\hat{y}_{c}^{final} = \hat{y}_{c} \times \frac{\hat{y}_{total\_raw}}{\hat{S}}$$

  This ensures that $\sum \hat{y}_{c}^{final} = \hat{y}_{total\_raw}$.  
* **Why?** The visual signal for "Total Biomass" (volume, height, coverage) is often stronger and less noisy than the signal for specific species breakdown. By predicting the Total directly and using the components as *proportions*, we leverage the most reliable signal.

### **5.3 Knowledge Distillation from Privileged Metadata**

Since we have NDVI and Height in training, we train a **Teacher Network**.20

* **Teacher Input:** Image embeddings \+ MLP(NDVI, Height, Date, State).  
* **Student Input:** Image embeddings only.  
* Loss: $\mathcal{L}_{total} = \mathcal{L}_{Tweedie}(\text{Student}) + \lambda \mathcal{L}_{KL}(\text{Student}, \text{Teacher})$.  
  This forces the Student's latent space to organize itself similarly to the Teacher's, effectively embedding "NDVI-awareness" into the image features.

## **6\. Implementation Strategy and Best Practices**

### **6.1 Data Augmentation for Vegetation**

Standard ImageNet augmentations are insufficient. We require domain-specific augmentations.21

* **Geometric:** RandomRotate90, Flip, Transpose (Top-down views are rotation invariant).  
* **Environmental:** RandomShadow, RandomSunFlare (to simulate field lighting variations).  
* **Texture:** GaussNoise, ImageCompression (to simulate sensor noise).  
* **Color:** Be cautious with HueShift. Green vs. Brown is the primary discriminator for Dead vs. Green. Shifting Green to Brown via augmentation will confuse the model. Focus on RandomBrightnessContrast and RandomGamma instead.  
* **Tiling (Mosaic):** Pasture heterogeneity is high. Training on $2 \\times 2$ mosaics of images (Mixup/CutMix) forces the model to learn local features rather than global context, improving generalization.24

### **6.2 Validation Strategy: Preventing Temporal Leakage**

Random K-Fold is dangerous because pasture samples from the same paddock on adjacent days are highly correlated.

* **Algorithm:** **Stratified Group K-Fold**.25  
  * **Group:** Sampling\_Location or Date. We must ensure no data from "Location A" appears in both train and validation.  
  * **Stratification:** Stratify by Dry\_Total\_g bins. This ensures every fold has a balanced representation of sparse and dense pastures.

### **6.3 Loss Function Configuration**

We recommend a compound loss function:

$$\mathcal{L} = 0.5 \mathcal{L}_{MSE}(\log(y_{tot}), \log(\hat{y}_{tot})) + 0.1 \sum \mathcal{L}_{Tweedie}(y_{comp}, \hat{y}_{comp}, p=1.5)$$

* **Log-Space MSE for Total:** Since biomass spans orders of magnitude, Log-MSE stabilizes gradients and prevents high-biomass samples from dominating.2  
* **Tweedie for Components:** Handles the zeros in Clover/Dead.

## **7\. Actionable Roadmap for the Competition**

1. **Preprocessing:**  
   * Implement **Stratified Group K-Fold** splitting on Location/Date.  
   * Generate **Tiled Images** (slice 1024x1024 into 4x 512x512) to increase effective sample size.  
2. **Model Training (Phase 1):**  
   * Train **DINOv2 \+ Linear Head** using the Compound Loss (LogMSE \+ Tweedie).  
   * Use **CutMix** augmentation to improve robustness.  
3. **Model Training (Phase 2 \- Distillation):**  
   * Train a **Teacher Model** that uses Ground Truth NDVI and Height.  
   * Fine-tune the Student DINOv2 model to match the Teacher's logits.  
4. **Post-Processing:**  
   * **Hierarchical Reconciliation:** Normalize component predictions so they sum to the predicted Total.  
   * **WA Correction:** If State classifiers (or metadata in training) show State=WA, force Dry\_Dead\_g \= 0\.  
   * **Zero-Clamping:** If Dry\_Clover\_g \< threshold, set to 0\.  
5. **Ensembling:**  
   * Average predictions from DINOv2, EfficientNetV2, and Swin Transformer models.  
   * Use a **Hill Climbing** algorithm to optimize the ensemble weights based on the validation Weighted $R^2$.

## **8\. Conclusion**

The CSIRO Image2Biomass challenge is a test of a model's ability to integrate visual perception with physical constraints. The "perfect" method is not a single black-box model but a **structured system** that:

1. Acknowledges the **Tweedie** nature of biomass data (sparse, skewed).  
2. Enforces **Hierarchical Consistency** (Sum of parts \= Whole).  
3. Leverages **Privileged Information** (NDVI) via distillation.  
4. Corrects for **Domain Shifts** (WA Dead anomaly).

By implementing this framework, we move beyond simple regression to a physically aware AI system capable of robust, agronomy-aligned predictions.

#### **Works cited**

1. CSIRO \- Image2Biomass Prediction | Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/overview/description](https://www.kaggle.com/competitions/csiro-biomass/overview/description)  
2. Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture, accessed December 26, 2025, [https://www.researchgate.net/publication/396968064\_Estimating\_Pasture\_Biomass\_from\_Top-View\_Images\_A\_Dataset\_for\_Precision\_Agriculture](https://www.researchgate.net/publication/396968064_Estimating_Pasture_Biomass_from_Top-View_Images_A_Dataset_for_Precision_Agriculture)  
3. Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture, accessed December 26, 2025, [https://arxiv.org/html/2510.22916v1](https://arxiv.org/html/2510.22916v1)  
4. sample\_train.csv  
5. Analysis: Height\_Ave\_cm and Dead Biomass Prediction \- Testing the Host's Hypothesis \- CSIRO \- Image2Biomass Prediction | Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/discussion/650736](https://www.kaggle.com/competitions/csiro-biomass/discussion/650736)  
6. Data Augmentation and Interpolation Improves Machine Learning-Based Pasture Biomass Estimation from Sentinel-2 Imagery \- MDPI, accessed December 26, 2025, [https://www.mdpi.com/2072-4292/17/23/3787](https://www.mdpi.com/2072-4292/17/23/3787)  
7. Data Augmentation and Interpolation Improves Machine Learning-Based Pasture Biomass Estimation from Sentinel-2 Imagery \- ResearchGate, accessed December 26, 2025, [https://www.researchgate.net/publication/397854444\_Data\_Augmentation\_and\_Interpolation\_Improves\_Machine\_Learning-Based\_Pasture\_Biomass\_Estimation\_from\_Sentinel-2\_Imagery](https://www.researchgate.net/publication/397854444_Data_Augmentation_and_Interpolation_Improves_Machine_Learning-Based_Pasture_Biomass_Estimation_from_Sentinel-2_Imagery)  
8. Zero-Inflated Regression | Towards Data Science, accessed December 26, 2025, [https://towardsdatascience.com/zero-inflated-regression-c7dfc656d8af/](https://towardsdatascience.com/zero-inflated-regression-c7dfc656d8af/)  
9. A two-stage super learner for healthcare expenditures \- PMC \- NIH, accessed December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9683480/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9683480/)  
10. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass)  
11. CSIRO \- Image2Biomass Prediction \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/discussion/614237](https://www.kaggle.com/competitions/csiro-biomass/discussion/614237)  
12. Optimizing Video Recommendation Systems: A Deep Dive into Tweedie Regression for Predicting Watch Time (Tubi Case Study) \- Shaped.ai, accessed December 26, 2025, [https://www.shaped.ai/blog/optimizing-video-recommendation-systems-a-deep-dive-into-tweedie-regression-for-predicting-watch-time-tubi-case-study](https://www.shaped.ai/blog/optimizing-video-recommendation-systems-a-deep-dive-into-tweedie-regression-for-predicting-watch-time-tubi-case-study)  
13. Tweedie Loss Function. An example: Insurance pricing | by Sathesan Thavabalasingam, accessed December 26, 2025, [https://sathesant.medium.com/tweedie-loss-function-395d96883f0b](https://sathesant.medium.com/tweedie-loss-function-395d96883f0b)  
14. The Gamma Hurdle Distribution. Published here… | by Jeff Allard | Medium, accessed December 26, 2025, [https://medium.com/@jeffrey.m.allard/the-gamma-hurdle-distribution-ccd6c8cd3f8e](https://medium.com/@jeffrey.m.allard/the-gamma-hurdle-distribution-ccd6c8cd3f8e)  
15. r \- hurdle model prediction \- count vs response \- Stack Overflow, accessed December 26, 2025, [https://stackoverflow.com/questions/48794622/hurdle-model-prediction-count-vs-response](https://stackoverflow.com/questions/48794622/hurdle-model-prediction-count-vs-response)  
16. DINOv2 Small \+ 3LGBM | CSIRO \- Kaggle, accessed December 26, 2025, [https://www.kaggle.com/code/kushubhai/dinov2-small-3lgbm-csiro](https://www.kaggle.com/code/kushubhai/dinov2-small-3lgbm-csiro)  
17. An Integrated Deep Learning Model with EfficientNet and ResNet for Accurate Multi-Class Skin Disease Classification \- MDPI, accessed December 26, 2025, [https://www.mdpi.com/2075-4418/15/5/551](https://www.mdpi.com/2075-4418/15/5/551)  
18. Enforcing Hard Linear Constraints in Deep Learning Models with Decision Rules \- arXiv, accessed December 26, 2025, [https://arxiv.org/html/2505.13858v1](https://arxiv.org/html/2505.13858v1)  
19. Output-Constrained Regression Trees \- arXiv, accessed December 26, 2025, [https://arxiv.org/html/2405.15314v3](https://arxiv.org/html/2405.15314v3)  
20. CSIRO \- Image2Biomass Prediction | Kaggle, accessed December 26, 2025, [https://www.kaggle.com/competitions/csiro-biomass/discussion/651525](https://www.kaggle.com/competitions/csiro-biomass/discussion/651525)  
21. albumentations-team/albumentations: Fast and flexible ... \- GitHub, accessed December 26, 2025, [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)  
22. (PDF) Data-efficient and accurate rapeseed leaf area estimation by self-supervised vision transformer for germplasms early evaluation \- ResearchGate, accessed December 26, 2025, [https://www.researchgate.net/publication/398378941\_Data-efficient\_and\_accurate\_rapeseed\_leaf\_area\_estimation\_by\_self-supervised\_vision\_transformer\_for\_germplasms\_early\_evaluation](https://www.researchgate.net/publication/398378941_Data-efficient_and_accurate_rapeseed_leaf_area_estimation_by_self-supervised_vision_transformer_for_germplasms_early_evaluation)  
23. Semantic Segmentation \- Albumentations, accessed December 26, 2025, [https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/](https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/)  
24. (PDF) Using Generative Module and Pruning Inference for the Fast and Accurate Detection of Apple Flower in Natural Environments \- ResearchGate, accessed December 26, 2025, [https://www.researchgate.net/publication/356680032\_Using\_Generative\_Module\_and\_Pruning\_Inference\_for\_the\_Fast\_and\_Accurate\_Detection\_of\_Apple\_Flower\_in\_Natural\_Environments](https://www.researchgate.net/publication/356680032_Using_Generative_Module_and_Pruning_Inference_for_the_Fast_and_Accurate_Detection_of_Apple_Flower_in_Natural_Environments)  
25. Machine-Learning/Stratified K-Fold Cross-Validation in Python.md at main \- GitHub, accessed December 26, 2025, [https://github.com/xbeat/Machine-Learning/blob/main/Stratified%20K-Fold%20Cross-Validation%20in%20Python.md](https://github.com/xbeat/Machine-Learning/blob/main/Stratified%20K-Fold%20Cross-Validation%20in%20Python.md)  
26. Assessing Transferability of Remote Sensing Pasture Estimates Using Multiple Machine Learning Algorithms and Evaluation Structures \- MDPI, accessed December 26, 2025, [https://www.mdpi.com/2072-4292/15/11/2940](https://www.mdpi.com/2072-4292/15/11/2940)