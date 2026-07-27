# Pathogen Intelligence System

A robustness-aware deep learning framework for pathogen identification, logit calibration, and stress-testing under clinical microscopy corruptions.

## System Overview

Standard convolutional neural networks trained on microbiological images often achieve high in-distribution lab accuracy but can suffer from overconfidence and fragility under real-world slide artifacts. 

The **Pathogen Intelligence System** evaluates model stability under 10 controlled corruptions (e.g., Gram stain variations, optical defocus, JPEG compression), applies post-hoc Temperature Scaling to calibrate softmax probabilities, and quantifies inter-model consensus between CNN architectures.

---

## Key Features

- **Multi-Model Architecture**: EfficientNet-B0, ResNet-50, Swin-T (Vision Transformer), and ConvNeXt-Tiny backbones.
- **4 Pathogen Target Classes**: *E. coli, K. pneumoniae, P. aeruginosa, S. aureus*.
- **Clinical Corruption Engine**: 10 controlled transformations modeling optical, staining, and sensor artifacts.
- **Accelerated Parallel Tensor Batch Inference**: Stacks corruptions into a single 4D tensor (`[10, 3, 224, 224]`) for single-pass GPU inference.
- **Logit Temperature Scaling**: Scalar and class-wise vector temperature parameters ($T \in \mathbb{R}^4$) optimized via L-BFGS to minimize Expected Calibration Error (ECE).
- **Composite Robustness Metric**: 30/30/20/20 weighted scoring formula (Consistency, Stability, Resistance, Calibration).
- **Inter-Model Consensus Analysis**: Cohen's and Fleiss' $\kappa$ metrics, false consensus detection, and trust classification.
- **Visual Explainability (Grad-CAM)**: Attention drift quantification showing feature location changes under stress.
- **Plate-Aware Splitting**: Strictly partitions images by physical petri dish plate boundaries to eliminate data leakage.

---

## System Architecture

```text
Input Image ──► Perturbation Engine ──► Tensor Batch Inference ──► Calibration & Analysis ──► Diagnostic Report
```

### Complete Workflow

1. **Plate-Aware Data Partitioning**: Splitting by physical petri-dish plate boundaries (9,106 train / 2,159 val / 1,083 test images).
2. **Model Training**: EfficientNet-B0 (~5.3M params), ResNet-50 (~25.6M params), Swin-T (~28M params), and ConvNeXt-Tiny (~28.6M params) with label smoothing, AdamW optimizer, and cosine learning rate schedules.
3. **Clinical Perturbation Engine**: Generates 10 corrupted variants in-memory with seed control.
4. **Calibrated Batch Inference**: Temperature scaling applied to raw logits before softmax evaluation.
5. **Robustness Scoring**: Evaluates prediction flips, confidence variance, and calibration penalties.
6. **Consensus Reliability**: Computes inter-model agreement matrix and flags false consensus states.
7. **Visualization & Reporting**: Exports reliability diagrams, confusion matrices, heatmaps, and publication-ready tables.

---

## Project Structure

```text
PATHOGEN-INTELLIGENCE-SYSTEM/
├── checkpoints/         # Local model weight binaries (*.pth) & temperature locks (git-ignored)
├── configs/             # Perturbation parameters & system constants
├── docs/                # System documentation & architectural guides
├── experiments/         # Benchmarking scripts & statistical evaluation
├── inference/           # Accelerated tensor batch inference & calibration engine
├── loaders/             # PyTorch dataset loaders & plate-level split logic
├── models/              # Network architecture definitions (EfficientNet-B0, ResNet-50, Swin-T, ConvNeXt-Tiny)
├── perturbations/       # 10-corruption clinical transformation engine
├── pipeline/            # Experiment registry, runner, & reproducibility seeding
├── reporting/           # HTML/Markdown report generator & publication tables
├── results/             # Generated plots, CSV evaluation metrics, & report manifests
├── stabilization/       # Schema validation & artifact integrity checks
├── tools/               # System health test scripts
├── training/            # Training scripts for EfficientNet-B0, ResNet-50, Swin-T, and ConvNeXt-Tiny
└── visualization/       # Calibration diagrams, heatmaps, & robustness plots
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/grishmghosh/Pathogen-intelligence-system.git
cd Pathogen-intelligence-system

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional)* If using `uv` for fast package management:
```bash
uv sync
```

---

### 2. Model Training & Checkpoint Setup

> **Note**: Pre-trained model weights (`*.pth`) are excluded from Git version control via `.gitignore`. You can train the models and generate temperature calibration parameters locally:

```bash
# Train network backbones (saves weights to checkpoints/)
uv run python training/train_efficientnet.py
uv run python training/train_resnet.py
uv run python training/train_swin.py
uv run python training/train_convnext.py

# Calibrate temperature scaling parameters (T) across models
uv run python generate_resnet_temp.py
```

---

### 3. Run Pipeline Test (Single Image Execution)

Test batch inference, temperature scaling, and robustness scoring on any input image:

```bash
uv run python test_intelligence_pipeline.py --image "/path/to/image.png"
```

If no image path is supplied, the script creates a synthetic test image automatically:
```bash
uv run python test_intelligence_pipeline.py --demo
```

---

### 4. Run Comprehensive 10-Phase Test Suite

Verify all pipeline modules (checkpoints, perturbations, inference, metadata, robustness scoring, error handling):

```bash
uv run python test_intelligence_layer_comprehensive.py --image "/path/to/image.png"
```

---

### 5. Run Dataset Evaluation & Plot Generation

Evaluate models on an entire dataset directory and generate figures in `results/evaluation/`:

```bash
uv run python run_full_eval.py --dataset-root dataset_split --output-dir results/evaluation
```

---

### 6. Run Statistical Significance & Analytical Benchmarks

```bash
# Run multi-seed robustness benchmark and statistical significance tests
uv run python experiments/run_statistical_benchmark.py --image "/path/to/image.png"

# Test inter-model disagreement & false consensus detection
uv run python test_step4_consensus_reliability.py

# Run stabilization & schema integrity checks
uv run python tools/run_stabilization_test.py

# Generate end-to-end visualization plots
uv run python visualization/test_visualization_e2e.py
```

---

## Python API Usage

### Batch Inference

```python
from inference.batch_inference import run_batch_inference

# Execute batch inference across all perturbations
results = run_batch_inference("path/to/microscopy_slide.png")

# Extract calibrated predictions and confidence
model_output = results["efficientnet_b0"]["original"]
prediction = model_output["prediction"]              # e.g. 'e_coli'
calibrated_conf = model_output["confidence"]          # Temperature-calibrated
raw_conf = model_output["raw_confidence"]             # Pre-calibration confidence
all_probabilities = model_output["all_probabilities"] # Dict {class: prob}
```

### Robustness Report Generation

```python
from analysis.robustness_analyzer import generate_robustness_report

# Generate robustness report
report = generate_robustness_report(results)

# Extract composite robustness score (0-100)
score = report["efficientnet_b0"]["robustness_score"]["robustness_score"]
print(f"EfficientNet-B0 Robustness Score: {score:.2f}/100")
```

---

## Robustness Metrics & Scoring

The overall **Robustness Score (0–100)** combines four diagnostic dimensions:

$$S_{\text{robustness}} = 0.30 \cdot C_{\text{consistency}} + 0.30 \cdot S_{\text{stability}} + 0.20 \cdot R_{\text{resistance}} + 0.20 \cdot K_{\text{calibration}}$$

1. **Prediction Consistency (30%)**: Percentage of corrupted variants yielding the same class prediction as the original image.
2. **Confidence Stability (30%)**: Variance of calibrated confidence scores across perturbations.
3. **Perturbation Resistance (20%)**: Resistance against class prediction flips under heavy noise.
4. **Calibration Quality (20%)**: Expected Calibration Error (ECE) metric; penalizes overconfidence saturation above $0.97$.

| Score Range | Interpretation |
| :--- | :--- |
| **90 – 100** | Excellent Robustness |
| **80 – 90** | Good Robustness |
| **70 – 80** | Moderate Robustness |
| **60 – 70** | Fair Robustness |
| **< 60** | Poor Robustness |

---

## Clinical Perturbation Suite

The system evaluates model stability across 10 controlled corruptions:

| Corruption | Type | Parameter | Physical Clinical Cause |
| :--- | :--- | :--- | :--- |
| `original` | Baseline | None | Reference image |
| `bright` | Brightness | $1.2\times$ | Overexposure under microscope lamp |
| `dark` | Brightness | $0.8\times$ | Underexposure / low lamp intensity |
| `high_contrast` | Contrast | $1.15\times$ | High-contrast condenser adjustment |
| `low_contrast` | Contrast | $0.85\times$ | Low-contrast / thick slide specimen |
| `gaussian_noise` | Noise | $\sigma = 8$ | Camera sensor thermal noise |
| `gaussian_blur` | Blur | Kernel = 5 | Slight optical focal misadjustment |
| `stain_shift` | Color | Hue shift = 12 | Gram staining reagent variations |
| `defocus_blur` | Optical | Kernel = 7 | High-magnification focal distance error |
| `jpeg_compression` | Artifact | Quality = 45 | Digital image transmission compression |

---

## Model Evaluation Results

### Performance Summary

| Metric | EfficientNet-B0 | ResNet-50 | Swin-T | ConvNeXt-Tiny |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | ~5.3M | ~25.6M | ~28M | ~28.6M |
| **Architecture Type** | CNN | CNN | Vision Transformer | Hybrid CNN |
| **Input Size** | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ | $224 \times 224$ |
| **Best Val Accuracy** | 99.58% | 99.58% | — | — |
| **Temperature ($T$)** | 0.8336 | 0.8329 | 0.8520 | 0.8415 |
| **ECE (Calibration)** | 0.0635 | **0.0224** | 0.0451 | 0.0714 |
| **Mean Robustness Score** | 62.82/100 | **82.04/100** | 74.97/100 | 68.33/100 |
| **Most Damaging Perturbation** | High Contrast | Low Contrast | Dark | Gaussian Noise |

### Per-Class Test Set Metrics (EfficientNet-B0)

| Pathogen Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| *E. coli* | 0.9704 | 0.9888 | 0.9795 | 265 |
| *K. pneumoniae* | 0.9870 | 0.9784 | 0.9827 | 232 |
| *P. aeruginosa* | 0.9891 | 0.9714 | 0.9802 | 280 |
| *S. aureus* | 0.9967 | 0.9967 | 0.9967 | 306 |

### Multi-Seed Robustness Benchmark (5-Seed Statistical Significance)

| Model Architecture | Robustness Score (Mean ± SD) | 95% Confidence Interval | p-value (vs Best) | Effect Size |
| :--- | :--- | :--- | :--- | :--- |
| **`swin_t`** | **91.40 ± 2.26** | [88.59, 94.20] | — *(Baseline)* | — |
| `convnext_tiny` | **84.54 ± 16.58** | [63.95, 105.13] | p = 0.4019 (ns) | 0.58 (Medium) |
| `efficientnet_b0` | **64.44 ± 0.00** | [64.44, 64.44] | p < 0.0001 (\*\*\*) | 16.87 (Large) |
| `resnet50` | **58.89 ± 0.00** | [58.89, 58.89] | p < 0.0001 (\*\*\*) | 20.34 (Large) |

> Note: Swin-T ranks highest in the multi-seed robustness benchmark. The Vision Transformer's global attention captures morphological patterns more consistently across stochastic perturbation seeds than CNN-based models.

---

## Generated Evaluation Figures

Figures are generated by `run_full_eval.py` and stored in `results/evaluation/`:

### Confusion Matrices
![EfficientNet-B0 Confusion Matrix](results/evaluation/EfficientNet_B0_1_confusion_matrix.png)
![ResNet-50 Confusion Matrix](results/evaluation/ResNet_50_1_confusion_matrix.png)
![Swin-T Confusion Matrix](results/evaluation/Swin_T_1_confusion_matrix.png)
![ConvNeXt-Tiny Confusion Matrix](results/evaluation/ConvNeXt_Tiny_1_confusion_matrix.png)

### Per-Class Metrics
![EfficientNet-B0 Per-Class Metrics](results/evaluation/EfficientNet_B0_2_per_class_metrics.png)
![ResNet-50 Per-Class Metrics](results/evaluation/ResNet_50_2_per_class_metrics.png)
![Swin-T Per-Class Metrics](results/evaluation/Swin_T_2_per_class_metrics.png)
![ConvNeXt-Tiny Per-Class Metrics](results/evaluation/ConvNeXt_Tiny_2_per_class_metrics.png)

### ROC Curves (One-vs-Rest)
![EfficientNet-B0 ROC Curves](results/evaluation/EfficientNet_B0_3_roc_curves.png)
![ResNet-50 ROC Curves](results/evaluation/ResNet_50_3_roc_curves.png)
![Swin-T ROC Curves](results/evaluation/Swin_T_3_roc_curves.png)
![ConvNeXt-Tiny ROC Curves](results/evaluation/ConvNeXt_Tiny_3_roc_curves.png)

### PR Curves
![EfficientNet-B0 PR Curves](results/evaluation/EfficientNet_B0_4_pr_curves.png)
![ResNet-50 PR Curves](results/evaluation/ResNet_50_4_pr_curves.png)
![Swin-T PR Curves](results/evaluation/Swin_T_4_pr_curves.png)
![ConvNeXt-Tiny PR Curves](results/evaluation/ConvNeXt_Tiny_4_pr_curves.png)

### Confidence Distributions
![EfficientNet-B0 Confidence Distribution](results/evaluation/EfficientNet_B0_5_confidence_distribution.png)
![ResNet-50 Confidence Distribution](results/evaluation/ResNet_50_5_confidence_distribution.png)
![Swin-T Confidence Distribution](results/evaluation/Swin_T_5_confidence_distribution.png)
![ConvNeXt-Tiny Confidence Distribution](results/evaluation/ConvNeXt_Tiny_5_confidence_distribution.png)

### Model Comparison & Heatmaps
![Model Comparison](results/evaluation/comparison_6_model_vs_model.png)
![Misclassification Heatmap](results/evaluation/comparison_7_misclassification_heatmap.png)

---

## Dataset Description

* **Source**: Microscopic Image Repository of Bacterial Pathogens.
* **Target Classes**: *E. coli*, *K. pneumoniae*, *P. aeruginosa*, *S. aureus*.
* **Plate-Aware Split**: 140 train / 32 val / 28 test plates per class. Images from the same physical petri dish plate are kept strictly within a single split (9,106 train / 2,159 val / 1,083 test images).

---

## Documentation Links

- **[System Architecture Guide](docs/PROJECT_STRUCTURE.md)**: Repository layout and package map.
- **[Intelligence Layer Technical Guide](docs/INTELLIGENCE_LAYER_GUIDE.md)**: Calibration formulas, logit scaling, and consensus scoring details.
- **[Perturbation Engine Report](docs/PERTURBATION_TEST_REPORT.md)**: Perturbation validation and transformation parameters.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.