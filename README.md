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

## System Architecture

```
Input Image → Perturbation Engine → Batch Inference → Robustness Analyzer → Intelligence Report
```

### Complete Pipeline

1. **Dataset Preparation** - Image-level splitting with guaranteed minimum samples
2. **Model Training** - EfficientNet-B0 and ResNet-50 with mixed precision
3. **Perturbation Generation** - 6 controlled variants (in-memory, reproducible)
4. **Batch Inference** - CNN predictions on all variants
5. **Robustness Analysis** - Stability, confidence, and sensitivity evaluation

## Project Structure

```
PATHOGEN-INTELLIGENCE-SYSTEM/
├── checkpoints/         # Trained model weights
├── configs/            # Configuration files
├── loaders/            # Data loading utilities
├── models/             # CNN architectures
├── training/           # Training scripts
├── perturbations/      # Perturbation framework
├── inference/          # Batch inference module
├── analysis/           # Robustness analyzer
├── docs/               # Documentation
├── data/               # Raw dataset
└── dataset_split/      # Train/val/test splits
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
# Split dataset into train/val/test
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

## Usage Examples

### Batch Inference

```python
from inference.batch_inference import run_batch_inference

# Run inference on image with all perturbations
results = run_batch_inference("path/to/pathogen_image.jpg")

# Access predictions
efficientnet_pred = results["efficientnet_b0"]["original"]["prediction"]
confidence = results["efficientnet_b0"]["original"]["confidence"]
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
- 40% Prediction consistency
- 40% Confidence stability
- 20% Perturbation resistance

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
- Loads trained CNN models
- Processes all perturbation variants
- Collects structured predictions
- Preserves metadata throughout

### Robustness Analyzer
- Analyzes prediction consistency
- Measures confidence drift
- Identifies vulnerabilities
- Compares model robustness

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+

See `requirements.txt` for complete list.

## Model Performance

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

**Split:** Image-level splitting with guaranteed minimum samples per class

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
```

## Future Enhancements

- [ ] Ensemble methods
- [ ] Per-class robustness analysis
- [ ] Uncertainty quantification
- [ ] Adversarial robustness testing
- [ ] Real-time monitoring
- [ ] Automated PDF reports
- [ ] Explainability (Grad-CAM)

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
**Last Updated:** May 16, 2026