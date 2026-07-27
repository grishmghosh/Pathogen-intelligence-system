# Pathogen Intelligence System - Project Structure

## Directory Layout

```text
PATHOGEN-INTELLIGENCE-SYSTEM/
│
├── checkpoints/                        # Trained PyTorch state dicts (*.pth) & temperature locks
│   ├── efficientnet_b0_best.pth       # EfficientNet-B0 weights (~16 MB)
│   ├── resnet50_best.pth              # ResNet-50 weights (~91 MB)
│   ├── swin_t_best.pth                # Swin-T weights (~108 MB)
│   ├── convnext_tiny_best.pth         # ConvNeXt-Tiny weights (~110 MB)
│   ├── efficientnet_b0_temperature.pth # Temperature calibration parameter (T=0.8336)
│   ├── resnet50_temperature.pth       # Temperature calibration parameter (T=0.8329)
│   ├── swin_t_temperature.pth         # Temperature calibration parameter (T=0.8520)
│   └── convnext_tiny_temperature.pth  # Temperature calibration parameter (T=0.8415)
│
├── configs/                            # Configuration modules
│   ├── __init__.py
│   └── perturbation_config.py         # 10-corruption transformation parameters
│
├── dataset_split/                      # Processed dataset splits (Plate-aware)
│   ├── train/                         # Training images by class
│   ├── val/                           # Validation images by class
│   └── test/                          # Test images by class
│
├── loaders/                            # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py                 # PyTorch DataLoader setup
│
├── models/                             # Neural network architectures
│   ├── __init__.py
│   ├── efficientnet_setup.py          # EfficientNet-B0 with ImageNet pretrained backbone
│   ├── resnet_setup.py                # ResNet-50 with ImageNet pretrained backbone
│   ├── swin_setup.py                  # Swin-T (Swin Transformer Tiny) with ImageNet pretrained backbone
│   └── convnext_setup.py              # ConvNeXt-Tiny with ImageNet pretrained backbone
│
├── perturbations/                      # Clinical corruption engine
│   ├── __init__.py
│   ├── perturbation_engine.py         # In-memory corruption generators
│   └── test_perturbation_pipeline.py  # Perturbation pipeline unit tests
│
├── inference/                          # Batch inference engine
│   ├── __init__.py
│   └── batch_inference.py             # Accelerated 4D tensor batching & logit scaling
│
├── analysis/                           # Analytical intelligence modules
│   ├── __init__.py
│   ├── calibration.py                 # ECE, MCE, & vector Temperature Scaling optimization
│   ├── robustness_analyzer.py         # 30/30/20/20 weighted robustness scoring formula
│   ├── disagreement/                  # Inter-model consensus, Cohen/Fleiss Kappa, false consensus
│   ├── uncertainty/                   # Shannon entropy & confidence dispersion
│   └── explainability/                # Grad-CAM saliency maps & attention drift analysis
│
├── visualization/                      # Plotting and figure generation
│   ├── __init__.py
│   ├── calibration_plots.py           # Reliability diagrams & confidence histograms
│   ├── heatmaps.py                    # Prediction flip heatmaps & severity charts
│   ├── robustness_plots.py            # Accuracy vs severity degradation curves
│   └── test_visualization_e2e.py      # Visualization integration tests
│
├── stabilization/                      # Framework health & quality assurance
│   ├── __init__.py
│   ├── artifact_integrity.py          # Hash verification for checkpoints & configs
│   ├── dataset_readiness.py           # Plate leakage & image format checks
│   └── schema_validation.py           # JSON schema validation for reports
│
├── experiments/                        # Experiment runners & benchmark suites
│   ├── experiment_runner.py           # Evaluation pipeline executor
│   └── experiment_registry.py         # Metadata logging & experiment tracking
│
├── reporting/                          # Automated report generation
│   ├── report_generator.py            # Markdown, HTML, & publication table exporter
│   └── narrative_summary.py           # Natural language diagnostic summaries
│
├── tools/                              # System health verification tools
│   └── run_stabilization_test.py      # Stabilization audit script
│
├── training/                           # Model training pipelines
│   ├── __init__.py
│   ├── train_efficientnet.py          # Training loop for EfficientNet-B0
│   ├── train_resnet.py                # Training loop for ResNet-50
│   ├── train_swin.py                  # Training loop for Swin-T
│   └── train_convnext.py              # Training loop for ConvNeXt-Tiny
│
├── docs/                               # System documentation & architectural guides
│   ├── INTEGRATION_LAYER_GUIDE.md     # Intelligence layer technical guide
│   ├── PERTURBATION_TEST_REPORT.md    # Perturbation engine validation report
│   └── PROJECT_STRUCTURE.md           # Repository architecture (this file)
│
├── README.md                           # Main repository documentation
├── requirements.txt                    # Python package dependencies
├── split_dataset.py                    # Plate-aware dataset splitting script
├── run_full_eval.py                    # End-to-end dataset evaluation entry point
├── test_intelligence_pipeline.py       # Primary single-image demo runner
└── test_intelligence_layer_comprehensive.py # 10-phase system verification suite
```

---

## Component Responsibilities

### 1. Data & Preprocessing (`loaders/`, `split_dataset.py`)
- **`split_dataset.py`**: Partitions raw images into train/val/test splits strictly at the physical petri dish plate level to prevent data leakage across split boundaries.
- **`loaders/data_loader.py`**: Builds PyTorch `DataLoader` instances with standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 2. Model Architectures & Training (`models/`, `training/`)
- **`models/efficientnet_setup.py`**: Constructs EfficientNet-B0 with a custom 4-class linear head.
- **`models/resnet_setup.py`**: Constructs ResNet-50 with a custom 4-class `nn.Sequential(Dropout, Linear)` head.
- **`models/swin_setup.py`**: Constructs Swin-T (Swin Transformer Tiny, ~28M params) with a custom 4-class linear head. Uses hierarchical shifted-window self-attention to capture global morphological patterns.
- **`models/convnext_setup.py`**: Constructs ConvNeXt-Tiny (~28.6M params) with a custom 4-class linear head in the `model.classifier[2]` slot.
- **`training/`**: Training scripts for all four models featuring label smoothing, AdamW optimizer, cosine annealing learning rate schedules, and early stopping.

### 3. Perturbation Engine (`perturbations/`, `configs/`)
- **`configs/perturbation_config.py`**: Defines transformation bounds for 10 clinical corruptions.
- **`perturbations/perturbation_engine.py`**: Applies transformations in-memory with NumPy/OpenCV float32 arithmetic and random seed control.

### 4. Inference & Logit Calibration (`inference/`)
- **`inference/batch_inference.py`**: Stacks corruptions into a single 4D tensor `[N, 3, 224, 224]` for accelerated GPU forward passes. Applies scalar or class-wise vector Temperature Scaling ($T \in \mathbb{R}^4$) to raw logits.

### 5. Analytical Intelligence (`analysis/`)
- **`analysis/calibration.py`**: Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE). Fits vector temperature parameters using L-BFGS.
- **`analysis/robustness_analyzer.py`**: Computes the 30/30/20/20 composite robustness score (Consistency, Stability, Resistance, Calibration).
- **`analysis/disagreement/`**: Computes Cohen's $\kappa$, Fleiss' $\kappa$, false consensus detection, and model trust classifications.
- **`analysis/uncertainty/`**: Computes Shannon entropy and confidence dispersion under noise.
- **`analysis/explainability/`**: Generates Grad-CAM saliency maps and calculates spatial attention drift across perturbations.

### 6. Visualization & Reporting (`visualization/`, `reporting/`)
- **`visualization/`**: Generates reliability diagrams, confusion matrices, prediction flip heatmaps, and per-class accuracy degradation curves.
- **`reporting/`**: Assembles diagnostic reports into structured JSON, Markdown, and publication-ready tables.

---

## Import Architecture

All package imports use standard root-relative imports:

```python
from configs.perturbation_config import PERTURBATION_CONFIG
from inference.batch_inference import run_batch_inference, load_model
from analysis.robustness_analyzer import generate_robustness_report
from analysis.calibration import fit_vector_temperature_scaling
from analysis.disagreement.consensus_reliability import compute_consensus_reliability
```

---

## Data Execution Pipeline

```text
Input Image ──► perturbation_engine.py (10 corruptions)
                      │
                      ▼
             batch_inference.py (predict_batch: [10, 3, 224, 224])
                      │
                      ▼
             robustness_analyzer.py (30/30/20/20 scoring)
                      │
                      ▼
             report_generator.py (JSON / Markdown Output)
```
