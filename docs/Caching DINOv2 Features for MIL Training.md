# **Comprehensive Analysis of Offline Feature Extraction, Caching Architectures, and Multiple Instance Learning for Biomass Estimation**

## **1\. Introduction: The Paradigm Shift in Computational Agronomy**

The intersection of high-resolution remote sensing and deep learning has ushered in a new era of computational agronomy, fundamentally altering how biomass and vegetation health are quantified. Traditionally, biomass estimation relied on manual sampling—a labor-intensive, destructive, and spatially sparse process—or simple spectral indices like NDVI (Normalized Difference Vegetation Index), which often saturate at high biomass levels and fail to capture complex morphological traits. The contemporary challenge is no longer data scarcity but rather data complexity and scale. Modern agricultural imaging, captured via drones or high-resolution satellite platforms, generates "gigapixel" data streams where a single field survey can yield terabytes of imagery. This visual data encodes rich, fine-grained information regarding plant species composition, phenological stages, and stress indicators, yet extracting this signal requires modeling architectures of immense capacity.

The emergence of Foundation Models, particularly those trained via Self-Supervised Learning (SSL) such as DINOv2 (Discriminative Knowledge Distillation with NO labels v2), represents a critical inflection point. Unlike supervised Convolutional Neural Networks (CNNs) that require massive labeled datasets to learn robust features, DINOv2 leverages vast repositories of unlabeled data to learn semantic visual descriptors that are invariant to lighting, pose, and background clutter. For agronomic tasks, where ground truth data (e.g., dry biomass weight in grams) is expensive and rare, DINOv2 acts as a powerful, frozen feature extractor, converting raw pixels into semantically dense vector representations.3

However, the deployment of such large-scale Vision Transformers (ViTs) presents significant engineering hurdles. A naive end-to-end training pipeline—where a DINOv2 backbone is coupled with a regression head and updated via backpropagation—is often computationally intractable for high-resolution agricultural datasets due to memory constraints and input/output (I/O) bottlenecks.5 To address this, a decoupled architecture is required: an **Offline Feature Extraction and Caching** pipeline. This approach treats the feature extraction as a one-time "compilation" step, freezing the visual capabilities of the backbone and storing the resulting latent representations in an optimized format (e.g., HDF5).

This report provides an exhaustive technical analysis of this methodology. It details the architectural nuances of DINOv2 in the context of Multiple Instance Learning (MIL), evaluates the comparative advantages of storage formats like HDF5 versus WebDataset for tensor caching, and explores advanced training strategies such as Frozen Feature Augmentation (FroFA) and Zero-Inflated LogNormal (ZILN) loss functions. Furthermore, it synthesizes these components into a robust reference implementation, addressing the specific constraints and anomalies identified in recent biomass competition datasets, such as the prevalence of structural zeros in "Dead" biomass components.5

## **2\. Theoretical Foundations of DINOv2 and Feature Distillation**

To optimize an extraction pipeline, one must first understand the generative mechanism of the features being extracted. DINOv2 is not merely a deeper network; it is the product of a specific training philosophy that favors local-to-global correspondence, making it uniquely suited for the patch-based analysis required in MIL.

### **2.1 The Student-Teacher Distillation Dynamic**

The core of DINOv2 is a discriminative self-supervised approach that combines the DINO (Self-distillation with no labels) objective with iBOT (Image BERT pre-training with Online Tokenizer) masked image modeling. In this framework, two networks—a Student and a Teacher—share the same architecture (e.g., ViT-Large). The Teacher is updated as an Exponential Moving Average (EMA) of the Student, ensuring stable training dynamics.3

The training process involves two primary loss components that are critical for understanding the resulting features:

1. **Image-Level Objective (DINO Loss):** This objective forces the Student to match the class-token () output of the Teacher for different augmented views of the same image. This encourages the model to learn semantic invariance—recognizing that a rotated, zoomed, or color-jittered patch of clover is still clover. For biomass estimation, this ensures that the features are robust to varying lighting conditions in the field.6  
2. **Patch-Level Objective (iBOT Loss):** The Student is presented with masked patches (where parts of the image are obscured) and must predict the features of the masked regions based on the visible context, matching the Teacher's view of the unmasked image. This local objective is vital for agronomy because it forces the model to understand the fine-grained texture and structure of vegetation (e.g., leaf shape, branching patterns) rather than relying solely on global context.6

The result is a feature space where both the global token and the local patch tokens carry rich semantic information. Unlike supervised models that often discard "background" information to focus on a classification target, DINOv2 preserves a holistic representation of the scene, which is essential when the goal is to quantify total biomass rather than identify a single object.

### **2.2 Register Tokens and Artifact Suppression**

A known issue with standard Vision Transformers is the emergence of high-norm "artifact" tokens in background areas (e.g., sky or shadows), which the model repurposes to store global information. In MIL, these high-norm tokens can disproportionately influence the aggregation function (e.g., Attention Pooling), leading to erroneous predictions. DINOv2 addresses this by introducing **Register Tokens**—additional learnable tokens appended to the sequence that act as "sinks" for this redundant information.7

When implementing the extraction pipeline, it is imperative to correctly parse the output sequence of the model. For a ViT-L/14 with registers, the output tensor typically has the shape $(B, N\_{patches} \+ 1 \+ N\_{registers}, D)$, where $D=1024$. The pipeline must explicitly discard the register tokens (indices 1 to $N\_{registers}$) and separate the token (index 0\) from the spatial patch tokens. Failure to do so would introduce noise into the MIL bag.8

### **2.3 The "Frozen" Backbone Hypothesis**

The premise of offline extraction is that the DINOv2 encoder, pre-trained on the massive LVD-142M dataset, acts as a universal vision sensor.4 For agricultural domains, this hypothesis largely holds. The dataset includes a diverse array of natural scenes, enabling the model to distinguish between subtle variations in vegetation texture. By "freezing" the weights, we assume that the manifold of plant features is already well-represented in the DINOv2 latent space. The downstream task becomes learning a mapping from this manifold to the scalar biomass values, rather than learning the visual features from scratch. This dramatically reduces the risk of overfitting to small agronomic datasets (e.g., the 357 images in the CSIRO dataset 5) and allows for the use of lightweight regressors.

## **3\. High-Resolution Imagery and the Tiling Strategy**

Agricultural images often exceed the input resolution limits of standard ViTs. A single orthomosaic of a trial plot might be $1000 \\times 2000$ pixels or larger.5 Feeding such an image directly into a standard ViT (typically trained on $224^2$ or $518^2$ images) requires aggressive downsampling, which destroys the textural details necessary to distinguish between species (e.g., differentiating the serrated leaves of weeds from the smooth leaves of crops).

### **3.1 The Sliding Window Approach**

To preserve resolution, the extraction pipeline must employ a sliding window (tiling) strategy. This involves extracting fixed-size crops (e.g., $518 \\times 518$) from the high-resolution source image. To ensure that features at the boundaries of patches are adequately represented, an overlap (stride) is used.

* **Patch Size ($P$):** 518 pixels (matching DINOv2 native resolution to avoid interpolation artifacts).  
* **Stride ($S$):** 400 pixels. This creates an overlap of 118 pixels between adjacent patches.5

This tiling process transforms a single source image into a "bag" of $N$ instances. For a $1000 \\times 2000$ image, this might result in a grid of approximately $3 \\times 5 \= 15$ patches. The dataset effectively moves from a dimension of $(1, 3, H, W)$ to $(N, 3, 518, 518)$.

### **3.2 Feature Aggregation Levels**

Once tiled, the pipeline must decide on the granularity of features to cache. DINOv2 offers features at multiple depths (layers) and granularities (tokens).

1. **Layer Selection:** The last layer contains the most abstract semantic features, but intermediate layers often retain better geometric precision. A common best practice for dense regression tasks is to concatenate the tokens from the last $n=4$ layers, forming a feature vector of size $4 \\times 1024 \= 4096$.9  
2. **Token Selection:**  
   * **CLS-Only:** Caching only the token per patch results in a bag of $N$ vectors. This is memory-efficient and generally sufficient if the patch size is small enough to contain a single semantic concept.  
   * **Spatial Tokens:** Caching the full grid of spatial tokens (e.g., $37 \\times 37$ tokens for a 518px patch) results in $N \\times 1369$ vectors. While this preserves maximum information, it explodes storage requirements (reaching terabytes for modest datasets) and is often computationally prohibitive for MIL aggregators.10

For biomass estimation, where the goal is a global aggregate (total weight), the **CLS-Only** approach applied to overlapping patches strikes the optimal balance between spatial resolution (via the patches) and feature compactness.5

## **4\. Engineering the Extraction Pipeline: Storage and I/O**

The defining bottleneck of offline feature extraction is not computation (which happens once) but storage and retrieval (I/O). Saving millions of individual tensor files (.pt) to disk is a catastrophic anti-pattern due to inode exhaustion and file system latency.10

### **4.1 Comparison of Storage Formats**

| Feature | Individual Files (.pt/.npy) | WebDataset (.tar) | HDF5 (.h5) | LMDB |
| :---- | :---- | :---- | :---- | :---- |
| **Inode Usage** | High (Critical Risk) | Low (1 per shard) | Low (1 single file) | Low (1 single file) |
| **Random Access** | Fast | Slow (Sequential only) | **Fast (Optimized Slicing)** | Very Fast |
| **Sequential Access** | Slow (File open/close overhead) | Very Fast | Fast | Fast |
| **Metadata Support** | None (requires sidecar files) | JSON sidecars | Native Attributes | Limited |
| **Ease of Use** | High | Medium | Medium | Low |
| **Compression** | None | Gzip/LZ4 | Gzip/LZF | None |

**WebDataset** is the standard for petabyte-scale datasets stored on cloud object stores (S3/GCS) because it optimizes for sequential throughput, which aligns with standard training epochs.11 However, for biomass estimation tasks typically involving datasets in the 10GB-1TB range where cross-validation (GroupKFold) requires complex random sampling, **HDF5** is superior. It essentially acts as a file system within a file, allowing for instantaneous random access to any tensor via array slicing logic, without the need to unpack archives.12

### **4.2 Implementing the HDF5 Cache**

The extraction script should initialize a single HDF5 file (or one per fold). The file structure should organize data hierarchically, perhaps by image\_id. However, for maximum training speed, a "flat" structure where all feature vectors are stored in a single dataset array features of shape $(Total\\\_Patches, D)$ and a corresponding indices array mapping image IDs to row ranges is preferred.

Crucially, HDF5 has limitations regarding concurrent writes (SWMR \- Single Writer Multiple Reader). The extraction process should be sequential or use a "gather" pattern where worker processes send tensors to a main writer process. During training, the file is opened in swmr=True and rdcc\_nbytes (chunk cache) is tuned to match the feature size.14

## **5\. Reference Implementation: The Extraction Pipeline**

The following Python code implements a robust Feature Extraction pipeline using DINOv2 and HDF5. It includes the tiling logic, model loading (handling registers), and efficient storage.

Python

import os  
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from PIL import Image  
import h5py  
import numpy as np  
from tqdm import tqdm  
import math

\# \---------------------------------------------------------  
\# 1\. Configuration & Constants  
\# \---------------------------------------------------------  
CONFIG \= {  
    'img\_size': 518,           \# DINOv2 Native Resolution  
    'patch\_stride': 400,       \# Stride for overlapping tiles  
    'batch\_size': 16,          \# Extraction batch size  
    'num\_workers': 4,  
    'device': 'cuda' if torch.cuda.is\_available() else 'cpu',  
    'backbone': 'dinov2\_vitl14\_reg', \# ViT-Large with Registers  
    'repo': 'facebookresearch/dinov2',  
    'output\_file': 'biomass\_features.h5'  
}

\# \---------------------------------------------------------  
\# 2\. Dataset for Tiled Inference  
\# \---------------------------------------------------------  
class BiomassTilingDataset(Dataset):  
    def \_\_init\_\_(self, image\_paths, image\_ids, config):  
        self.image\_paths \= image\_paths  
        self.image\_ids \= image\_ids  
        self.config \= config  
          
        \# DINOv2 Preprocessing (ImageNet stats)  
        self.transform \= transforms.Compose(,   
                                 std=\[0.229, 0.224, 0.225\]),  
        \])

    def \_\_len\_\_(self):  
        return len(self.image\_paths)

    def get\_patches(self, img\_tensor):  
        """  
        Tiles the input tensor (C, H, W) into patches of size (C, 518, 518\)  
        with stride 400\. Returns tensor of shape (N, C, 518, 518).  
        """  
        c, h, w \= img\_tensor.shape  
        patch\_size \= self.config\['img\_size'\]  
        stride \= self.config\['patch\_stride'\]  
          
        \# Calculate number of patches along H and W  
        n\_h \= math.ceil((h \- patch\_size) / stride) \+ 1 if h \> patch\_size else 1  
        n\_w \= math.ceil((w \- patch\_size) / stride) \+ 1 if w \> patch\_size else 1  
          
        patches \=  
        for i in range(n\_h):  
            for j in range(n\_w):  
                \# Calculate start indices, handling boundary conditions (clamping)  
                y \= min(i \* stride, h \- patch\_size)  
                x \= min(j \* stride, w \- patch\_size)  
                  
                \# Ensure we don't have negative indices if image is smaller than patch  
                y \= max(0, y)  
                x \= max(0, x)  
                  
                patch \= img\_tensor\[:, y:y+patch\_size, x:x+patch\_size\]  
                  
                \# Pad if image is smaller than patch size  
                if patch.shape\[1\] \< patch\_size or patch.shape\[2\] \< patch\_size:  
                    pad\_h \= patch\_size \- patch.shape\[1\]  
                    pad\_w \= patch\_size \- patch.shape\[2\]  
                    patch \= torch.nn.functional.pad(patch, (0, pad\_w, 0, pad\_h))  
                      
                patches.append(patch)  
                  
        return torch.stack(patches)

    def \_\_getitem\_\_(self, idx):  
        path \= self.image\_paths\[idx\]  
        img\_id \= self.image\_ids\[idx\]  
          
        \# Load and convert to RGB  
        try:  
            image \= Image.open(path).convert('RGB')  
            img\_tensor \= self.transform(image)  
            patches \= self.get\_patches(img\_tensor)  
            return patches, img\_id  
        except Exception as e:  
            print(f"Error loading {path}: {e}")  
            \# Return dummy tensor in production code or handle gracefully  
            return torch.zeros(1, 3, 518, 518), img\_id

\# \---------------------------------------------------------  
\# 3\. Model Wrapper (Handling Registers)  
\# \---------------------------------------------------------  
class DINOv2Extractor(nn.Module):  
    def \_\_init\_\_(self, model\_name):  
        super().\_\_init\_\_()  
        \# Load model from hub  
        self.backbone \= torch.hub.load(CONFIG\['repo'\], model\_name)  
        self.backbone.eval() \# Explicitly set to eval mode

    def forward(self, x):  
        \# x shape: (B\_patches, 3, 518, 518\)  
        \# DINOv2 forward\_features returns dict with keys:  
        \# 'x\_norm\_clstoken', 'x\_norm\_regtokens', 'x\_norm\_patchtokens'  
          
        output \= self.backbone.forward\_features(x)  
          
        \# We only want the CLS token for MIL aggregation  
        \# Shape: (B\_patches, 1024\)  
        cls\_token \= output\['x\_norm\_clstoken'\]  
        return cls\_token

\# \---------------------------------------------------------  
\# 4\. Main Extraction Loop  
\# \---------------------------------------------------------  
def extract\_and\_cache\_features(image\_dir, csv\_path):  
    import pandas as pd  
    df \= pd.read\_csv(csv\_path)  
    \# Ensure correct paths  
    paths \= \[os.path.join(image\_dir, f) for f in df\['image\_path'\].tolist()\]  
    ids \= df\['image\_id'\].tolist()  
      
    dataset \= BiomassTilingDataset(paths, ids, CONFIG)  
      
    \# Collate function to flatten the batch of bags into a single batch of patches  
    \# Note: We process 1 image at a time (batch\_size=1 in loader) to keep logic simple  
    \# The 'batch' inside the model will be the number of patches in that image.  
    dataloader \= DataLoader(dataset, batch\_size=1, shuffle=False, num\_workers=CONFIG\['num\_workers'\])  
      
    model \= DINOv2Extractor(CONFIG\['backbone'\]).to(CONFIG\['device'\])  
      
    \# Initialize HDF5 file  
    with h5py.File(CONFIG\['output\_file'\], 'w') as f:  
        \# Create a variable length dataset for features since N\_patches varies per image  
        \# Using a special dtype or flattening strategy.   
        \# Strategy: Store all features in one big array, map image\_id to indices.  
          
        \# We'll use a simpler Group strategy for clarity: Group per ImageID  
        \# Group: "ID12345" \-\> Dataset: "features" (N, 1024\)  
          
        print("Starting extraction...")  
        with torch.no\_grad():  
            for patches, img\_id in tqdm(dataloader):  
                \# patches shape: (1, N, 3, 518, 518\) \-\> squeeze \-\> (N, 3, 518, 518\)  
                patches \= patches.squeeze(0).to(CONFIG\['device'\])  
                  
                \# Check for empty patches (error handling)  
                if patches.dim()\!= 4: continue  
                  
                \# Extract features (inference)  
                \# If N is large, we might need to batch this too  
                features \=  
                mini\_batch \= 16 \# Process patches in mini-batches to save VRAM  
                for i in range(0, len(patches), mini\_batch):  
                    batch \= patches\[i:i+mini\_batch\]  
                    feat \= model(batch)  
                    features.append(feat.cpu())  
                  
                features \= torch.cat(features, dim=0).numpy()  
                  
                \# Save to HDF5  
                img\_id\_str \= str(img\_id)  
                grp \= f.create\_group(img\_id\_str)  
                grp.create\_dataset('features', data=features, compression="gzip")  
                  
    print(f"Extraction complete. Saved to {CONFIG\['output\_file'\]}")

\# Example Usage (Commented out)  
\# extract\_and\_cache\_features('/kaggle/input/csiro-biomass/', '/kaggle/input/csiro-biomass/train.csv')

## **6\. Training Strategy: Augmentation and Specialized Losses**

With the features cached in HDF5, the training pipeline is instantiated. This phase utilizes a lightweight Multiple Instance Learning (MIL) model. Since pixel-level augmentation is impossible on frozen features, the pipeline relies on **Feature Space Augmentation**. Additionally, the statistical properties of the biomass data necessitate specific loss functions.

### **6.1 Frozen Feature Augmentation (FroFA)**

Research suggests that applying augmentations directly to the latent vectors can simulate the effect of pixel-level augmentations, improving generalizability in low-data regimes.15

* **Noise Injection:** Adding Gaussian noise ($\\mathcal{N}(0, \\sigma)$) simulates sensor noise.  
* **Feature Scaling (Brightness/Contrast):** Multiplying the feature vector by a scalar $\\alpha \\sim U(0.9, 1.1)$ simulates global illumination changes. DINOv2 features are magnitude-sensitive, so this augmentation forces the MIL head to rely on relative feature directions rather than absolute magnitudes.16  
* **Mixup:** Creating linear combinations of feature bags from different training samples ($x\_{new} \= \\lambda x\_1 \+ (1-\\lambda)x\_2$) regularizes the regression manifold.

### **6.2 The Zero-Inflated LogNormal (ZILN) Loss**

Biomass components like "Dead" or "Clover" are semi-continuous: they have a probability mass at zero (absence) and a continuous distribution for positive values (presence). The "Dead Matter Anomaly" in Western Australia (100% zeros) highlighted in the snippets 5 underscores the need for this. Standard MSE fails here because it averages the zeros into the mean prediction.

The ZILN loss predicts three parameters for each target $y$:

1. $p$: Probability of being zero (logits).  
2. $\\mu$: Mean of the log-normal distribution.  
3. $\\sigma$: Standard deviation of the log-normal distribution.

The loss function is:

$$\\mathcal{L}\_{ZILN} \= \\text{BCE}(p, \\mathbb{I}\_{y=0}) \+ \\mathbb{I}\_{y\>0} \\cdot \\left( \\frac{( \\ln y \- \\mu )^2}{2\\sigma^2} \+ \\ln(\\sigma y \\sqrt{2\\pi}) \\right)$$

### **6.3 Implementation: The Lightning Module**

The following code implements the training module, incorporating the Gated Attention MIL head, FroFA, and the specialized loss logic.

Python

import pytorch\_lightning as pl  
import torch.nn.functional as F

class GatedAttentionMIL(nn.Module):  
    def \_\_init\_\_(self, dim=1024, hidden\_dim=256):  
        super().\_\_init\_\_()  
        self.attention\_V \= nn.Sequential(nn.Linear(dim, hidden\_dim), nn.Tanh())  
        self.attention\_U \= nn.Sequential(nn.Linear(dim, hidden\_dim), nn.Sigmoid())  
        self.attention\_weights \= nn.Linear(hidden\_dim, 1)

    def forward(self, x):  
        \# x: (B, N\_instances, D)  
        A\_V \= self.attention\_V(x)  \# N x hidden  
        A\_U \= self.attention\_U(x)  \# N x hidden  
        A \= self.attention\_weights(A\_V \* A\_U) \# N x 1 element-wise mult  
        A \= torch.softmax(A, dim=1) \# Softmax over instances  
          
        \# Weighted sum  
        M \= torch.bmm(A.transpose(1, 2), x)  \# (B, 1, N) \* (B, N, D) \-\> (B, 1, D)  
        return M.squeeze(1), A \# Return aggregate and weights

class BiomassMILModel(pl.LightningModule):  
    def \_\_init\_\_(self, lr=1e-4):  
        super().\_\_init\_\_()  
        self.save\_hyperparameters()  
          
        \# MIL Aggregator  
        self.mil \= GatedAttentionMIL(dim=1024)  
          
        \# Heads for 5 targets:  
        \# For ZILN, we need 3 outputs per target (p, mu, sigma) \-\> 15 outputs  
        \# For Tweedie (Total Biomass), we might just predict mean, but let's assume ZILN for all components  
        self.head \= nn.Sequential(  
            nn.Linear(1024, 512),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(512, 5 \* 3) \# 5 targets \* 3 params  
        )  
          
    def forward(self, x):  
        \# x is bag of features: (B, N, 1024\)  
        embedding, attn\_weights \= self.mil(x)  
        output \= self.head(embedding)  
        return output.view(-1, 5, 3) \# Reshape to (Batch, Targets, Params)

    def ziln\_loss(self, pred, target):  
        \# pred: (B, 3\) \-\> \[logit\_p, mu, sigma\]  
        \# target: (B)  
          
        logit\_p \= pred\[:, 0\]  
        mu \= pred\[:, 1\]  
        sigma \= F.softplus(pred\[:, 2\]) \+ 1e-6 \# Ensure positive  
          
        \# 1\. Classification Loss (Zero vs Non-Zero)  
        is\_zero \= (target \== 0).float()  
        \# BCEWithLogits takes logits. Target 1 if zero, 0 if positive  
        class\_loss \= F.binary\_cross\_entropy\_with\_logits(logit\_p, is\_zero)  
          
        \# 2\. Regression Loss (LogNormal) \- only for non-zero targets  
        \# Create mask for positive targets  
        is\_positive \= (target \> 0)  
          
        reg\_loss \= torch.tensor(0.0, device=self.device)  
        if is\_positive.sum() \> 0:  
            pos\_targets \= target\[is\_positive\]  
            pos\_mu \= mu\[is\_positive\]  
            pos\_sigma \= sigma\[is\_positive\]  
              
            \# LogNormal Negative Log Likelihood  
            log\_target \= torch.log(pos\_targets)  
            nll \= 0.5 \* torch.pow((log\_target \- pos\_mu) / pos\_sigma, 2) \+ \\  
                  torch.log(pos\_sigma) \+ torch.log(pos\_targets) \# \+ constant  
            reg\_loss \= nll.mean()  
              
        return class\_loss \+ reg\_loss

    def training\_step(self, batch, batch\_idx):  
        features, targets \= batch \# features: (B, N, 1024), targets: (B, 5\)  
          
        \# Apply FroFA (Frozen Feature Augmentation)  
        if self.training:  
            noise \= torch.randn\_like(features) \* 0.01  
            scale \= (torch.rand\_like(features) \* 0.2 \+ 0.9)  
            features \= features \* scale \+ noise  
              
        preds \= self(features) \# (B, 5, 3\)  
          
        loss \= 0  
        weights \= \[0.1, 0.1, 0.1, 0.2, 0.5\] \# Competition weights  
          
        for i in range(5):  
            target\_i \= targets\[:, i\]  
            pred\_i \= preds\[:, i, :\]  
            loss \+= weights\[i\] \* self.ziln\_loss(pred\_i, target\_i)  
              
        self.log('train\_loss', loss)  
        return loss

    def configure\_optimizers(self):  
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight\_decay=1e-3)

## **7\. Kaggle Implementation Workflow: The Two-Kernel Solution**

To execute this pipeline effectively within Kaggle's environment (12-hour limit, non-persistent /kaggle/working), a two-stage kernel approach is mandatory.

### **7.1 Kernel 1: Feature Extractor**

This kernel runs the extract\_and\_cache\_features code. Ideally, it processes the dataset in folds.

* **Action:** Runs DINOv2 on GPU. Saves .h5 files.  
* **Output:** Generates a file biomass\_features.h5.  
* **Persistence:** You must use the Kaggle API to save this output as a **Dataset**.  
  * In the notebook: \!kaggle datasets init \-p /kaggle/working  
  * Edit dataset-metadata.json programmatically.  
  * \!kaggle datasets create \-p /kaggle/working (or version update).  
    This saves the extracted features permanently, independent of the notebook session.17

### **7.2 Kernel 2: MIL Training**

This kernel attaches the dataset created by Kernel 1\.

* **Input:** Reads from /kaggle/input/my-feature-dataset/biomass\_features.h5.  
* **Action:** Runs the BiomassMILModel training using PyTorch Lightning.  
* **Advantage:** Since there is no image decoding or backbone inference, training epochs take seconds instead of minutes. This allows for extensive hyperparameter tuning (e.g., trying different MIL aggregators or learning rates) using tools like Optuna within the same session.

## **8\. Conclusion**

Optimizing the biomass estimation pipeline requires a fundamental restructuring of the training workflow. By shifting from online, end-to-end training to an offline, cached approach using DINOv2 and HDF5, researchers can bypass the inherent computational bottlenecks of gigapixel agronomic imagery. The "frozen" nature of the features is mitigated by Feature Space Augmentation, while the "zero-heavy" nature of the target variables is addressed via the Zero-Inflated LogNormal loss. This architecture not only makes training computationally feasible but also mathematically robust to the specific distribution anomalies found in agricultural data.

**Key Recommendation Summary:**

| Component | Strategy | Rationale |
| :---- | :---- | :---- |
| **Backbone** | DINOv2 ViT-L/14 with Registers | State-of-the-art semantic features; registers clean artifacts. |
| **Caching** | HDF5 (swmr mode) | Efficient random access for bags; solves inode limits. |
| **Aggregation** | Gated Attention MIL | Learns to ignore soil/background patches automatically. |
| **Regularization** | Frozen Feature Augmentation (FroFA) | Prevents overfitting on static feature vectors. |
| **Loss** | Zero-Inflated LogNormal (ZILN) | Explicitly models the probability of biomass presence (zeros). |

#### **Works cited**

1. DINOv2 by Meta: A Self-Supervised foundational vision model \- Learn OpenCV, accessed December 31, 2025, [https://learnopencv.com/dinov2-self-supervised-vision-transformer/](https://learnopencv.com/dinov2-self-supervised-vision-transformer/)  
2. DINOv2: Self-supervised Learning Model Explained \- Encord, accessed December 31, 2025, [https://encord.com/blog/dinov2-self-supervised-learning-explained/](https://encord.com/blog/dinov2-self-supervised-learning-explained/)  
3. image2biomass-afhn.ipynb  
4. DINOv2: Learning Robust Visual Features without Supervision \- arXiv, accessed December 31, 2025, [https://arxiv.org/html/2304.07193v2](https://arxiv.org/html/2304.07193v2)  
5. PyTorch code and models for the DINOv2 self-supervised learning method. \- GitHub, accessed December 31, 2025, [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)  
6. timm/vit\_base\_patch14\_reg4\_dinov2.lvd142m · How to get register token output values ?, accessed December 31, 2025, [https://huggingface.co/timm/vit\_base\_patch14\_reg4\_dinov2.lvd142m/discussions/4](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m/discussions/4)  
7. \[FEATURE\] DINOv2 Feature Map extraction · huggingface pytorch-image-models · Discussion \#2068 \- GitHub, accessed December 31, 2025, [https://github.com/huggingface/pytorch-image-models/discussions/2068](https://github.com/huggingface/pytorch-image-models/discussions/2068)  
8. Best way to save and load lots of tensors \- data \- PyTorch Forums, accessed December 31, 2025, [https://discuss.pytorch.org/t/best-way-to-save-and-load-lots-of-tensors/170767](https://discuss.pytorch.org/t/best-way-to-save-and-load-lots-of-tensors/170767)  
9. Efficient PyTorch I/O library for Large Datasets, Many Files, Many GPUs, accessed December 31, 2025, [https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/)  
10. Is there an analysis speed or memory usage advantage to using HDF5 for large array storage (instead of flat binary files)? \- Stack Overflow, accessed December 31, 2025, [https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr](https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr)  
11. HDF5 Datasets For PyTorch \- Medium, accessed December 31, 2025, [https://medium.com/data-science/hdf5-datasets-for-pytorch-631ff1d750f5](https://medium.com/data-science/hdf5-datasets-for-pytorch-631ff1d750f5)  
12. Save torch tensors as hdf5 \- vision \- PyTorch Forums, accessed December 31, 2025, [https://discuss.pytorch.org/t/save-torch-tensors-as-hdf5/39556](https://discuss.pytorch.org/t/save-torch-tensors-as-hdf5/39556)  
13. Frozen Feature Augmentation, accessed December 31, 2025, [https://frozen-feature-augmentation.github.io/](https://frozen-feature-augmentation.github.io/)  
14. Frozen Feature Augmentation for Few-Shot Image Classification \- CVF Open Access, accessed December 31, 2025, [https://openaccess.thecvf.com/content/CVPR2024/papers/Bar\_Frozen\_Feature\_Augmentation\_for\_Few-Shot\_Image\_Classification\_CVPR\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.pdf)  
15. How to create very large datasets from a notebook \- Kaggle, accessed December 31, 2025, [https://www.kaggle.com/code/xhlulu/how-to-create-very-large-datasets-from-a-notebook](https://www.kaggle.com/code/xhlulu/how-to-create-very-large-datasets-from-a-notebook)  
16. Upload models and datasets with kaggle API \!, accessed December 31, 2025, [https://www.kaggle.com/code/tareksherif/upload-models-and-datasets-with-kaggle-api](https://www.kaggle.com/code/tareksherif/upload-models-and-datasets-with-kaggle-api)