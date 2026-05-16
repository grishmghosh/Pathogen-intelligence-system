

# Intelligence Layer Guide

## Overview

The Intelligence Layer is the core analytical component of the Pathogen Intelligence System. It transforms raw CNN predictions into robustness-aware intelligence through systematic perturbation analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PATHOGEN INTELLIGENCE SYSTEM                │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dataset    │────▶│ CNN Training │────▶│   Trained    │
│              │     │              │     │   Models     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input Image │────▶│ Perturbation │────▶│  Perturbed   │
│              │     │    Engine    │     │   Variants   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │    Batch     │
                                          │  Inference   │◀─── Trained Models
                                          └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  Robustness  │
                                          │   Analyzer   │
                                          └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │ Intelligence │
                                          │    Report    │
                                          └──────────────┘
```

---

## Components

### 1. Batch Inference (`inference/batch_inference.py`)

**Purpose:** Bridge between perturbation generation and robustness analysis.

**Key Functions:**

#### `load_model(model_name, checkpoint_path, device)`
Loads trained CNN models from checkpoints.

```python
model = load_model("efficientnet_b0", "checkpoints/efficientnet_b0_best.pth", device)
```

#### `preprocess_image(image_np, target_size=(224, 224))`
Preprocesses numpy images for PyTorch inference.
- Resizes to 224×224
- Converts to tensor
- Applies ImageNet normalization
- Adds batch dimension

#### `predict_single(model, image_tensor, device)`
Runs inference on a single image.

Returns:
```python
{
    "prediction": "e_coli",
    "confidence": 0.97,
    "probabilities": [0.97, 0.01, 0.01, 0.01],
    "predicted_idx": 0
}
```

#### `run_batch_inference(image_path, models_config=None, device=None)`
**Main entry point.** Runs complete inference pipeline.

Returns structured results:
```python
{
    "efficientnet_b0": {
        "original": {
            "prediction": "e_coli",
            "confidence": 0.97,
            "probabilities": [...],
            "predicted_idx": 0,
            "metadata": {
                "type": "none",
                "parameter": None,
                "id": "original"
            }
        },
        "gaussian_noise": {
            "prediction": "e_coli",
            "confidence": 0.89,
            ...
        },
        ...
    },
    "resnet50": {
        ...
    }
}
```

---

### 2. Robustness Analyzer (`analysis/robustness_analyzer.py`)

**Purpose:** Intelligence layer for robustness evaluation.

**Key Functions:**

#### `analyze_prediction_consistency(model_results)`
Measures prediction stability across perturbations.

Returns:
```python
{
    "original_prediction": "e_coli",
    "consistent_predictions": 6,
    "total_perturbations": 7,
    "consistency_rate": 85.71,
    "prediction_distribution": {"e_coli": 6, "k_pneumoniae": 1},
    "inconsistent_perturbations": ["gaussian_noise"]
}
```

**Interpretation:**
- 100%: Perfect consistency (all perturbations → same prediction)
- 80-99%: Good consistency (minor instability)
- 60-79%: Moderate consistency (some vulnerability)
- <60%: Poor consistency (unstable predictions)

#### `analyze_confidence_drift(model_results)`
Measures confidence stability across perturbations.

Returns:
```python
{
    "original_confidence": 0.97,
    "mean_confidence": 0.89,
    "std_confidence": 0.05,
    "min_confidence": 0.82,
    "max_confidence": 0.97,
    "confidence_drop": 0.15,
    "confidence_variance": 0.0025,
    "per_perturbation_drift": {
        "bright": -0.03,
        "dark": -0.08,
        ...
    }
}
```

**Interpretation:**
- Low std (<0.05): Stable confidence
- Medium std (0.05-0.10): Moderate stability
- High std (>0.10): Unstable confidence

#### `analyze_perturbation_sensitivity(model_results)`
Identifies which perturbations most affect the model.

Returns:
```python
{
    "most_damaging_perturbation": {
        "name": "gaussian_noise",
        "impact_score": 0.65,
        "confidence_drop": 0.15
    },
    "least_damaging_perturbation": {
        "name": "bright",
        "impact_score": 0.03,
        "confidence_drop": 0.03
    },
    "perturbation_impact_ranking": [...],
    "prediction_flip_perturbations": ["gaussian_noise"]
}
```

**Impact Score Formula:**
```
impact_score = confidence_drop + (0.5 if prediction_changed else 0)
```

#### `compute_robustness_score(consistency, confidence, sensitivity)`
Computes overall robustness score (0-100).

**Formula:**
```
robustness_score = 0.40 × consistency_rate
                 + 0.40 × stability_score
                 + 0.20 × resistance_score
```

**Score Interpretation:**
- 90-100: Excellent robustness
- 80-90: Good robustness
- 70-80: Moderate robustness
- 60-70: Fair robustness
- <60: Poor robustness

#### `compare_models(inference_results)`
Compares robustness between models (EfficientNet vs ResNet).

Returns:
```python
{
    "model_rankings": [
        ("efficientnet_b0", {"robustness_score": 87.5, ...}),
        ("resnet50", {"robustness_score": 82.3, ...})
    ],
    "comparative_analysis": {...},
    "winner": "efficientnet_b0"
}
```

#### `generate_robustness_report(inference_results)`
**Main entry point.** Generates complete robustness analysis.

---

## Usage Examples

### Example 1: Basic Pipeline

```python
from inference.batch_inference import run_batch_inference
from analysis.robustness_analyzer import generate_robustness_report

# Run inference
results = run_batch_inference("path/to/pathogen_image.jpg")

# Analyze robustness
report = generate_robustness_report(results)

# Access results
efficientnet_score = report["efficientnet_b0"]["robustness_score"]["robustness_score"]
print(f"EfficientNet Robustness: {efficientnet_score:.2f}/100")
```

### Example 2: Custom Model Configuration

```python
# Specify custom checkpoints
models_config = {
    "efficientnet_b0": "custom_checkpoints/efficientnet_v2.pth",
    "resnet50": "custom_checkpoints/resnet_v2.pth"
}

results = run_batch_inference(
    image_path="image.jpg",
    models_config=models_config,
    device=torch.device("cuda")
)
```

### Example 3: Detailed Analysis

```python
from analysis.robustness_analyzer import (
    analyze_prediction_consistency,
    analyze_confidence_drift,
    analyze_perturbation_sensitivity
)

# Get model results
model_results = results["efficientnet_b0"]

# Individual analyses
consistency = analyze_prediction_consistency(model_results)
confidence = analyze_confidence_drift(model_results)
sensitivity = analyze_perturbation_sensitivity(model_results)

print(f"Consistency Rate: {consistency['consistency_rate']:.2f}%")
print(f"Confidence Drop: {confidence['confidence_drop']:.4f}")
print(f"Most Damaging: {sensitivity['most_damaging_perturbation']['name']}")
```

### Example 4: Complete Pipeline Test

```bash
# Run complete pipeline test
python test_intelligence_pipeline.py
```

---

## Output Formats

### Inference Results Structure

```python
{
    "model_name": {
        "perturbation_name": {
            "prediction": str,           # Predicted class
            "confidence": float,         # Confidence score (0-1)
            "probabilities": list,       # Softmax probabilities
            "predicted_idx": int,        # Class index
            "metadata": {
                "type": str,             # Perturbation type
                "parameter": float,      # Parameter value
                "id": str                # Unique ID
            }
        }
    }
}
```

### Robustness Report Structure

```python
{
    "model_name": {
        "consistency_analysis": {...},
        "confidence_analysis": {...},
        "sensitivity_analysis": {...},
        "robustness_score": {
            "robustness_score": float,
            "consistency_score": float,
            "stability_score": float,
            "resistance_score": float,
            "interpretation": str
        }
    },
    "model_comparison": {
        "model_rankings": [...],
        "comparative_analysis": {...},
        "winner": str
    }
}
```

---

## Robustness Metrics Explained

### 1. Prediction Consistency
**What it measures:** How often the model maintains the same prediction across perturbations.

**Why it matters:** A robust model should not change its prediction due to minor input variations.

**Formula:** `(consistent_predictions / total_perturbations) × 100`

### 2. Confidence Stability
**What it measures:** How much confidence scores vary across perturbations.

**Why it matters:** Stable confidence indicates reliable uncertainty estimation.

**Metrics:**
- Standard deviation of confidence
- Maximum confidence drop
- Confidence variance

### 3. Perturbation Sensitivity
**What it measures:** Which perturbations most affect the model.

**Why it matters:** Identifies model vulnerabilities and weak points.

**Metrics:**
- Impact score per perturbation
- Prediction flip count
- Confidence drop per perturbation

### 4. Overall Robustness Score
**What it measures:** Composite metric combining all aspects.

**Why it matters:** Single interpretable score for model comparison.

**Components:**
- 40% Prediction consistency
- 40% Confidence stability
- 20% Perturbation resistance

---

## Best Practices

### 1. Model Evaluation
- Always test on multiple images
- Compare both EfficientNet and ResNet
- Analyze per-class robustness
- Test on edge cases

### 2. Perturbation Selection
- Use realistic perturbations (current config is well-tuned)
- Avoid extreme perturbations that destroy image content
- Test domain-specific perturbations (e.g., microscopy artifacts)

### 3. Interpretation
- Don't rely solely on robustness score
- Examine individual metrics
- Identify specific vulnerabilities
- Consider application requirements

### 4. Reporting
- Include both quantitative metrics and qualitative analysis
- Visualize perturbation effects
- Document model comparison rationale
- Provide actionable recommendations

---

## Integration with Other Modules

### Export to CSV

```python
import pandas as pd

# Convert report to DataFrame
data = []
for model_name, model_report in report.items():
    if model_name != "model_comparison":
        robustness = model_report["robustness_score"]
        data.append({
            "model": model_name,
            "robustness_score": robustness["robustness_score"],
            "consistency": robustness["consistency_score"],
            "stability": robustness["stability_score"]
        })

df = pd.DataFrame(data)
df.to_csv("outputs/robustness_report.csv", index=False)
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot confidence drift
confidence_analysis = report["efficientnet_b0"]["confidence_analysis"]
drift = confidence_analysis["per_perturbation_drift"]

plt.figure(figsize=(10, 6))
plt.bar(drift.keys(), drift.values())
plt.xlabel("Perturbation")
plt.ylabel("Confidence Drift")
plt.title("EfficientNet Confidence Drift per Perturbation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/confidence_drift.png")
```

---

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution:** Ensure models are trained and checkpoints exist in `checkpoints/` folder.

### Issue: "CUDA out of memory"
**Solution:** Use CPU device or reduce batch size:
```python
results = run_batch_inference(image_path, device=torch.device("cpu"))
```

### Issue: "No valid predictions found"
**Solution:** Check that inference completed successfully. Review error messages in results.

### Issue: Low robustness scores
**Analysis:**
- Check if model is properly trained
- Verify perturbation parameters are reasonable
- Test on multiple images to confirm pattern
- Consider model architecture limitations

---

## Future Enhancements

### Planned Features
1. **Ensemble Analysis** - Combine multiple models for improved robustness
2. **Per-Class Robustness** - Analyze robustness for each pathogen class
3. **Uncertainty Quantification** - Bayesian approaches for confidence estimation
4. **Adversarial Testing** - Test against adversarial perturbations
5. **Real-time Monitoring** - Track robustness over time
6. **Automated Reporting** - Generate PDF reports with visualizations

### Extension Points
- Custom perturbation types
- Additional robustness metrics
- Model-specific analysis
- Domain-specific evaluations

---

## References

### Related Modules
- `perturbations/perturbation_engine.py` - Perturbation generation
- `models/efficientnet_setup.py` - EfficientNet architecture
- `models/resnet_setup.py` - ResNet architecture
- `loaders/data_loader.py` - Data preprocessing

### Documentation
- `docs/PROJECT_STRUCTURE.md` - Overall project structure
- `docs/PERTURBATION_TEST_REPORT.md` - Perturbation testing results

---

## Contact & Support

For questions or issues with the intelligence layer:
1. Check this documentation
2. Review example usage in `test_intelligence_pipeline.py`
3. Examine module docstrings
4. Test with sample images

---

**Last Updated:** May 16, 2026  
**Version:** 1.0  
**Status:** Production Ready
