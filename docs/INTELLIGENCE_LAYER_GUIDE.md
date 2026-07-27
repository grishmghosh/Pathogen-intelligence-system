# Pathogen Intelligence Layer Technical Guide

## System Architecture Overview

The **Intelligence Layer** is the analytical core of the Pathogen Intelligence System. Rather than relying on static softmax classification accuracy, it stress-tests neural network predictions across controlled clinical corruptions, applies logit temperature scaling for probability calibration, and quantifies inter-model consensus.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      PATHOGEN INTELLIGENCE SYSTEM                      │
└────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Input Image  │────►│ Perturbation │────►│ 10 Corrupted │
│ (RGB/HEIC)   │     │    Engine    │     │   Variants   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 │
┌──────────────┐                        ┌──────────────┐
│ 4 Trained    │                        │ Tensor Batch │
│ Checkpoints  │────────────────────────►│  Inference   │
│ (EffNet-B0,  │                        │ (4 parallel  │
│  ResNet-50,  │                        │  model runs) │
│  Swin-T,     │                        └──────────────┘
│  ConvNeXt-T) │                               │
└──────────────┘                               │
                                                 │
┌──────────────┐                        ┌──────────────┐
│ Diagnostic   │◄────────────────────────│  Robustness  │
│ Reports      │                        │   Analyzer   │
└──────────────┘                        └──────────────┘
```

---

## 1. Accelerated Batch Inference (`inference/batch_inference.py`)

### Key Capabilities
- **Parallel Tensor Batching (`predict_batch`)**: Converts all 10 perturbation variants into a single 4D tensor `[10, 3, 224, 224]` and evaluates them in a single forward pass.
- **Logit Temperature Scaling**: Supports scalar temperature scaling ($T \in \mathbb{R}$) and class-wise vector temperature scaling ($T \in \mathbb{R}^4$).
- **Multi-Format Input Support**: Handles uint8 RGB, 4-channel RGBA, single-channel Grayscale, float arrays in `[0, 1]`, and HEIC images.

### Primary Functions

#### `load_model(model_name, checkpoint_dir=None, device=None)`
Loads PyTorch model checkpoints and corresponding temperature parameters. Supported model names: `"efficientnet_b0"`, `"resnet50"`, `"swin_t"`, `"convnext_tiny"`:

```python
from inference.batch_inference import load_model

# CNN models
model, temperature = load_model("efficientnet_b0", checkpoint_dir="checkpoints", device="cpu")
model, temperature = load_model("resnet50", checkpoint_dir="checkpoints", device="cpu")

# Vision Transformer & Hybrid CNN
model, temperature = load_model("swin_t", checkpoint_dir="checkpoints", device="cpu")
model, temperature = load_model("convnext_tiny", checkpoint_dir="checkpoints", device="cpu")
```

#### `predict_batch(model, perturbations_dict, temperature=1.0, device=None)`
Executes parallel inference across all perturbation variants:

```python
# Returns dictionary keyed by perturbation ID with calibrated probabilities and confidence
batch_results = predict_batch(model, perturbations, temperature=temperature, device="cuda")
```

---

## 2. Robustness Analyzer (`analysis/robustness_analyzer.py`)

### Composite Robustness Score (0–100)

The system computes a weighted composite score evaluating model stability under corruption:

$$S_{\text{robustness}} = 0.30 \cdot C_{\text{consistency}} + 0.30 \cdot S_{\text{stability}} + 0.20 \cdot R_{\text{resistance}} + 0.20 \cdot K_{\text{calibration}}$$

### Score Components

1. **Prediction Consistency ($C_{\text{consistency}}$)**:
   $$\text{Consistency} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = \hat{y}_{\text{orig}})}{N} \times 100$$
   Tracks the percentage of corruptions where the model maintains its original class prediction.

2. **Confidence Stability ($S_{\text{stability}}$)**:
   $$\text{Stability} = \max\left(0, 100 - 100 \times \frac{\sigma_{\text{conf}}}{\mu_{\text{conf}} + \epsilon}\right)$$
   Measures the coefficient of variation in confidence across all variants.

3. **Perturbation Resistance ($R_{\text{resistance}}$)**:
   $$\text{Resistance} = \max\left(0, 100 - 20 \times N_{\text{flips}}\right)$$
   Penalizes prediction flips caused by corruptions.

4. **Calibration Quality ($K_{\text{calibration}}$)**:
   $$\text{Calibration} = \max\left(0, 100 - 500 \cdot \text{ECE}\right) - \text{Penalty}_{\text{saturation}}$$
   Evaluates Expected Calibration Error (ECE) and applies a penalty if confidence saturation exceeds $0.97$ on incorrect predictions.

---

## 3. Logit Calibration & Temperature Scaling (`analysis/calibration.py`)

### Temperature Scaling Formulation

Raw logits $\mathbf{z} \in \mathbb{R}^K$ are scaled by temperature parameter $\mathbf{T} > 0$ before Softmax evaluation:

$$p_i(\mathbf{z}, \mathbf{T}) = \frac{\exp(z_i / T_i)}{\sum_{j=1}^K \exp(z_j / T_j)}$$

### Optimization via L-BFGS

Class-wise vector temperatures $\mathbf{T} \in \mathbb{R}^4$ are fitted on validation logits by minimizing Negative Log-Likelihood (NLL):

```python
from analysis.calibration import fit_vector_temperature_scaling

# Optimize per-class temperature scaling parameters
optimal_temperatures = fit_vector_temperature_scaling(val_logits, val_labels)
```

---

## 4. Inter-Model Disagreement & Consensus (`analysis/disagreement/`)

### Consensus Metrics
- **Cohen's $\kappa$**: Pairwise inter-model agreement adjusted for chance.
- **Fleiss' $\kappa$**: Multi-model agreement metric across model ensembles.
- **False Consensus Detection**: Flags cases where models agree on a prediction ($\hat{y}_A = \hat{y}_B$) but are wrong under perturbation noise.

### Trust Classification Categories
1. **High Trust**: High confidence, zero prediction flips, inter-model agreement.
2. **Fragile Consensus**: Models agree, but confidence variance under noise is high.
3. **Critical Disagreement**: Models output conflicting pathogen classes.

---

## 5. Visual Explainability & Attention Drift (`analysis/explainability/`)

### Grad-CAM Saliency Maps
Generates class activation heatmaps highlighting discriminative visual regions on microscopy slides.

### Spatial Attention Drift
Tracks the displacement of peak attention coordinates $(\Delta x, \Delta y)$ between original and corrupted images to detect when image quality loss causes visual focus collapse.
