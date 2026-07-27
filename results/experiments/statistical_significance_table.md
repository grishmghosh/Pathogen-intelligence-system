# Statistical Significance Benchmark Table

| Model Architecture | Robustness Score (Mean ± SD) | 95% Confidence Interval | p-value (vs Best) | Effect Size (Cohen's d) |
| :--- | :--- | :--- | :--- | :--- |
| `swin_t` | **91.40 ± 2.26** | [88.59, 94.20] | — *(Baseline)* | — |
| `convnext_tiny` | **84.54 ± 16.58** | [63.95, 105.13] | p = 0.4019 (ns) | 0.58 (Medium) |
| `efficientnet_b0` | **64.44 ± 0.00** | [64.44, 64.44] | p = 0.0000 (***) | 16.87 (Large) |
| `resnet50` | **58.89 ± 0.00** | [58.89, 58.89] | p = 0.0000 (***) | 20.34 (Large) |
