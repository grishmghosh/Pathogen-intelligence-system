# Pathogen Intelligence System - Project Structure

## Directory Organization

```
PATHOGEN-INTELLIGENCE-SYSTEM/
│
├── checkpoints/                    # Trained model weights and logs
│   ├── efficientnet_b0_best.pth   # Best EfficientNet-B0 model
│   ├── resnet50_best.pth          # Best ResNet-50 model
│   ├── efficientnet_log.csv       # EfficientNet training logs
│   └── resnet_log.csv             # ResNet training logs
│
├── configs/                        # Configuration files
│   ├── __init__.py
│   └── perturbation_config.py     # Perturbation parameters
│
├── data/                           # Raw dataset (original images)
│   └── A Microbiological Image Repository of Escherichia/
│
├── dataset_split/                  # Processed dataset splits
│   ├── train/                     # Training images by class
│   ├── val/                       # Validation images by class
│   └── test/                      # Test images by class
│
├── loaders/                        # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py             # PyTorch DataLoader setup
│
├── models/                         # Model architectures
│   ├── __init__.py
│   ├── efficientnet_setup.py      # EfficientNet-B0 setup
│   └── resnet_setup.py            # ResNet-50 setup
│
├── training/                       # Training scripts
│   ├── __init__.py
│   ├── train_efficientnet.py      # EfficientNet training
│   └── train_resnet.py            # ResNet training
│
├── perturbations/                  # Perturbation framework
│   ├── __init__.py
│   ├── perturbation_engine.py     # Perturbation generation
│   └── test_perturbation_pipeline.py  # Pipeline tests
│
├── inference/                      # Inference and prediction
│   └── __init__.py
│   # Future: batch_inference.py
│
├── analysis/                       # Robustness analysis
│   └── __init__.py
│   # Future: robustness_analysis.py
│
├── explainability/                 # Model interpretability
│   └── __init__.py
│   # Future: grad_cam.py, attention_maps.py
│
├── docs/                           # Documentation
│   ├── PERTURBATION_TEST_REPORT.md
│   ├── PERTURBATION_PIPELINE_STATUS.txt
│   ├── perturbation_test_results.png
│   └── PROJECT_STRUCTURE.md       # This file
│
├── outputs/                        # Generated outputs
│   # Future: inference results, analysis reports
│
├── utils/                          # Utility functions
│   └── __init__.py
│   # Future: visualization.py, metrics.py
│
├── venv/                           # Virtual environment
│
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
├── split_dataset.py                # Dataset splitting script
└── .gitignore                      # Git ignore rules
```

## Module Descriptions

### Core Modules

#### `configs/`
Configuration files for experiments and perturbations.
- `perturbation_config.py`: Perturbation parameters (brightness, contrast, noise, blur)

#### `loaders/`
Data loading and preprocessing utilities.
- `data_loader.py`: PyTorch DataLoader with ImageNet normalization

#### `models/`
Neural network architectures.
- `efficientnet_setup.py`: EfficientNet-B0 with pretrained weights
- `resnet_setup.py`: ResNet-50 with pretrained weights

#### `training/`
Model training scripts.
- `train_efficientnet.py`: EfficientNet training with mixed precision
- `train_resnet.py`: ResNet training with mixed precision

#### `perturbations/`
Perturbation generation framework for robustness testing.
- `perturbation_engine.py`: Generate controlled image perturbations
- `test_perturbation_pipeline.py`: Comprehensive pipeline tests

### Future Modules

#### `inference/`
Batch inference and prediction pipelines.
- Future: `batch_inference.py` - Run inference on perturbed images

#### `analysis/`
Robustness analysis and metrics.
- Future: `robustness_analysis.py` - Analyze model performance under perturbations

#### `explainability/`
Model interpretability and visualization.
- Future: `grad_cam.py` - Gradient-weighted Class Activation Mapping
- Future: `attention_maps.py` - Attention visualization

#### `utils/`
Shared utility functions.
- Future: `visualization.py` - Plotting and visualization helpers
- Future: `metrics.py` - Custom evaluation metrics

#### `outputs/`
Generated outputs from inference and analysis.
- Future: Inference results, robustness reports, visualizations

## Import Structure

### From Root Directory

```python
# Config
from configs.perturbation_config import PERTURBATION_CONFIG

# Data loading
from loaders.data_loader import get_data_loaders

# Models
from models.efficientnet_setup import build_efficientnet_b0
from models.resnet_setup import build_resnet50

# Perturbations
from perturbations.perturbation_engine import generate_perturbations
```

### From Subdirectories

Scripts in subdirectories use relative imports:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.perturbation_config import PERTURBATION_CONFIG
from loaders.data_loader import get_data_loaders
```

## Running Scripts

### Training

```bash
# From root directory
python training/train_efficientnet.py
python training/train_resnet.py
```

### Dataset Splitting

```bash
# From root directory
python split_dataset.py
```

### Testing Perturbations

```bash
# From root directory
python perturbations/test_perturbation_pipeline.py
```

## Data Flow

1. **Raw Data** → `data/` (original images)
2. **Split Data** → `split_dataset.py` → `dataset_split/` (train/val/test)
3. **Training** → `training/*.py` → `checkpoints/` (model weights)
4. **Perturbations** → `perturbations/perturbation_engine.py` → In-memory variants
5. **Inference** → `inference/*.py` → `outputs/` (predictions)
6. **Analysis** → `analysis/*.py` → `outputs/` (reports)

## Best Practices

1. **Always run from root directory** to ensure imports work correctly
2. **Use virtual environment** (`venv/`) for dependency isolation
3. **Keep configs separate** from code for easy experimentation
4. **Document changes** in `docs/` folder
5. **Save outputs** to `outputs/` folder, not in code directories

## Version Control

The `.gitignore` excludes:
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `checkpoints/*.pth` - Large model files
- `data/` - Raw dataset
- `dataset_split/` - Processed dataset
- `outputs/` - Generated outputs

## Next Steps

1. Implement `inference/batch_inference.py`
2. Implement `analysis/robustness_analysis.py`
3. Add visualization utilities in `utils/`
4. Implement explainability methods in `explainability/`
