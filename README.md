# Pathogen Intelligence System

A robustness-aware deep learning system for pathogen classification with comprehensive stability analysis.

## Overview

The Pathogen Intelligence System goes beyond traditional classification accuracy to provide **robustness-aware intelligence** about pathogen identification. It evaluates not just what the model predicts, but how stable and reliable those predictions are under realistic perturbations.

### Key Features

- 🧬 **Multi-Model Classification**: EfficientNet-B0 and ResNet-50 architectures
- 🔬 **4 Pathogen Classes**: E. coli, K. pneumoniae, P. aeruginosa, S. aureus
- 🎯 **Perturbation Framework**: 6 controlled perturbations (brightness, contrast, noise, blur)
- 🤖 **Batch Inference**: Automated prediction collection across perturbations
- 📊 **Robustness Analysis**: Comprehensive stability and confidence evaluation
- 🏆 **Model Comparison**: Side-by-side robustness comparison
- 🌡️ **Temperature Scaling Calibration**: Log-parameterised; honest confidence scores, not raw softmax
- 🔍 **Disagreement & Consensus Analysis**: Inter-model agreement, false consensus detection, trust classification
- 📈 **Visualization Suite**: Calibration plots, reliability diagrams, robustness heatmaps, experiment summaries
- 🛡️ **Stabilization Framework**: Schema validation, artifact integrity, dataset readiness checks

## System Architecture

```
Input Image → Perturbation Engine → Batch Inference → Robustness Analyzer → Intelligence Report
```

### Complete Pipeline

1. **Dataset Preparation** - Plate-level splitting — no image from the same plate crosses split boundaries
2. **Model Training** - EfficientNet-B0 and ResNet-50 with label smoothing, AdamW, cosine LR, early stopping
3. **Perturbation Generation** - 6 controlled variants (in-memory, reproducible)
4. **Batch Inference** - Calibrated CNN predictions (temperature scaling applied) on all variants
5. **Robustness Analysis** - Stability, confidence, and sensitivity evaluation (30/30/20/20 formula)
6. **Disagreement Analysis** - Inter-model agreement, confidence disagreement, perturbation-induced disagreement
7. **Consensus Reliability** - False consensus detection, trust classification, consistency metrics
8. **Stabilization Checks** - Schema validation, artifact integrity, dataset readiness
9. **Visualization** - Calibration plots, heatmaps, robustness charts, experiment summaries

## Project Structure

```
PATHOGEN-INTELLIGENCE-SYSTEM/
├── checkpoints/         # Trained model weights + temperature files
├── configs/             # Configuration files
├── loaders/             # Data loading utilities
├── models/              # CNN architectures
├── training/            # Training scripts (EfficientNet-B0, ResNet-50)
├── perturbations/       # Perturbation framework
├── inference/           # Calibrated batch inference module
├── analysis/
│   ├── robustness_analyzer.py   # 30/30/20/20 robustness score
│   ├── calibration.py           # ECE, MCE, reliability diagram, temperature scaling
│   ├── disagreement/            # Agreement metrics, consensus reliability, trust analysis
│   ├── uncertainty/             # Entropy analysis, confidence dispersion
│   └── explainability/          # GradCAM, attention analysis
├── visualization/       # Calibration plots, heatmaps, experiment summaries
├── stabilization/       # Artifact integrity, schema validation, dataset readiness
├── experiments/         # Benchmark runner, perturbation benchmarking
├── pipeline/            # Experiment config, registry, runner
├── reporting/           # Narrative summary, publication tables, report generator
├── results/             # Generated JSON, CSV, and plot outputs
├── docs/                # Documentation
├── data/                # Raw dataset
└── dataset_split/       # Train/val/test splits (plate-level)
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Split dataset into train/val/test at the plate level (no leakage)
python split_dataset.py
```

### 3. Train Models

```bash
# Train EfficientNet-B0
python training/train_efficientnet.py

# Train ResNet-50
python training/train_resnet.py
```

### 4. Run Intelligence Pipeline

```bash
# Complete pipeline: Perturbations → Inference → Analysis
python test_intelligence_pipeline.py
```

### 5. Run Analysis Suite

```bash
python test_disagreement_analysis.py
python test_step2_confidence_disagreement.py
python test_step3_perturbation_disagreement.py
python test_step4_consensus_reliability.py
python tools/run_stabilization_test.py
python visualization/test_visualization_e2e.py
```

## Usage Examples

### Batch Inference

```python
from inference.batch_inference import run_batch_inference

# Run inference on image with all perturbations
results = run_batch_inference("path/to/pathogen_image.jpg")

# Access predictions
efficientnet_pred = results["efficientnet_b0"]["original"]["prediction"]
confidence     = results["efficientnet_b0"]["original"]["confidence"]      # calibrated
raw_confidence = results["efficientnet_b0"]["original"]["raw_confidence"]  # pre-scaling
all_probs      = results["efficientnet_b0"]["original"]["all_probabilities"]  # dict {class: prob}
```

### Robustness Analysis

```python
from analysis.robustness_analyzer import generate_robustness_report

# Generate comprehensive robustness report
report = generate_robustness_report(results)

# Access robustness score
score = report["efficientnet_b0"]["robustness_score"]["robustness_score"]
print(f"Robustness: {score:.2f}/100")
```

## Robustness Metrics

### 1. Prediction Consistency
Measures how often predictions remain stable across perturbations.
- **100%**: Perfect stability
- **80-99%**: Good stability
- **<80%**: Unstable predictions

### 2. Confidence Stability
Measures variance in confidence scores across perturbations.
- **Low variance**: Reliable confidence
- **High variance**: Unstable confidence

### 3. Perturbation Sensitivity
Identifies which perturbations most affect the model.
- **Most damaging**: Highest impact perturbation
- **Prediction flips**: Perturbations causing class changes

### 4. Overall Robustness Score (0-100)
Composite metric combining all aspects:
- 30% Prediction consistency
- 30% Confidence stability
- 20% Perturbation resistance
- 20% Calibration quality (penalises confidence saturation above 0.97)

**Score Interpretation:**
- 90-100: Excellent robustness
- 80-90: Good robustness
- 70-80: Moderate robustness
- 60-70: Fair robustness
- <60: Poor robustness

## Documentation

- **[Intelligence Layer Guide](docs/INTELLIGENCE_LAYER_GUIDE.md)** - Comprehensive usage guide
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed architecture
- **[Perturbation Testing](docs/PERTURBATION_TEST_REPORT.md)** - Perturbation validation
- **[Implementation Status](INTELLIGENCE_LAYER_STATUS.md)** - Current status

## Key Components

### Perturbation Engine
- Generates 6 controlled perturbations
- In-memory processing (no disk writes)
- Reproducible with seed control
- Complete metadata tracking

### Batch Inference
- Loads trained CNN models with temperature scaling applied
- Returns both calibrated confidence and raw pre-scaling confidence
- Processes all perturbation variants
- Preserves metadata throughout

### Robustness Analyzer
- Analyzes prediction consistency
- Measures confidence drift
- Identifies vulnerabilities
- Compares model robustness
- Rebalanced 30/30/20/20 formula with calibration penalty

### Calibration Module
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Reliability diagram data
- Overconfidence ratio
- Log-parameterised temperature scaling (always positive T)

### Disagreement & Consensus Analysis
- Inter-model agreement matrix
- Confidence-aware disagreement scoring
- Perturbation-induced disagreement detection
- False consensus detection, trust classification, consensus reliability scoring

### Visualization Suite
- Calibration plots (reliability diagram, confidence histogram, calibration curve)
- Robustness heatmaps (prediction flip, severity vs instability, model comparison)
- Experiment summary (CSV + JSON)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- seaborn
- pillow-heif (for HEIC image conversion)

See `requirements.txt` for complete list.

## Model Performance

| Metric | EfficientNet-B0 | ResNet-50 |
|---|---|---|
| Best Val Loss | 0.3598 (epoch 9) | 0.3608 (epoch 10) |
| Best Val Accuracy | 99.58% | 99.58% |
| Early Stopping | Epoch 17 | Epoch 18 |
| Temperature (T) | 0.8336 | 0.8329 |
| Test Accuracy | 98.43% | **99.63%** |
| Robustness Score | 96.83/100 | **97.74/100** |

**Most Robust Model: ResNet-50**

### EfficientNet-B0
- Parameters: ~5.3M
- Input size: 224×224
- Training: Mixed precision (FP16)

### ResNet-50
- Parameters: ~25.6M
- Input size: 224×224
- Training: Mixed precision (FP16)

## Dataset

**Source:** A Microbiological Image Repository of Escherichia

**Classes:**
- E. coli (e_coli)
- K. pneumoniae (k_pneumoniae)
- P. aeruginosa (p_aeruginosa)
- S. aureus (s_aureus)

**Split:** Plate-level splitting — 140 train / 32 val / 28 test plates per class. No image from the same physical plate crosses split boundaries. Total: 9,106 train / 2,159 val / 1,083 test images.

## Perturbations

1. **Brightness Increase** (factor: 1.2)
2. **Brightness Decrease** (factor: 0.8)
3. **Contrast Increase** (factor: 1.15)
4. **Contrast Decrease** (factor: 0.85)
5. **Gaussian Noise** (sigma: 8)
6. **Gaussian Blur** (kernel: 5)

All perturbations use float32 arithmetic to prevent overflow and are reproducible with seed control.

## Testing

```bash
# Test perturbation generation
python perturbations/test_perturbation_pipeline.py

# Test complete intelligence pipeline
python test_intelligence_pipeline.py

# Run full analysis suite
python test_disagreement_analysis.py
python test_step2_confidence_disagreement.py
python test_step3_perturbation_disagreement.py
python test_step4_consensus_reliability.py
python tools/run_stabilization_test.py
python visualization/test_visualization_e2e.py
```

## Future Enhancements

- [ ] Ensemble methods
- [ ] Per-class robustness analysis
- [x] Uncertainty quantification
- [ ] Adversarial robustness testing
- [ ] Real-time monitoring
- [ ] Automated PDF reports
- [x] Explainability (Grad-CAM)

## Contributing

This is a research project for pathogen classification with robustness analysis. Contributions welcome for:
- Additional perturbation types
- New robustness metrics
- Visualization improvements
- Documentation enhancements

## License

[Specify your license here]

## Citation

If you use this system in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]

---

**Status:** Production Ready  
**Version:** 1.0  
**Last Updated:** May 27, 2026