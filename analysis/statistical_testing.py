"""
analysis/statistical_testing.py
================================
Statistical Significance Testing & Multi-Seed Analysis Engine
Provides descriptive statistics (Mean ± SD, 95% CI), hypothesis testing
(Paired t-test, Wilcoxon Signed-Rank Test), and effect size calculation (Cohen's d).
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def compute_descriptive_stats(data: List[float] | np.ndarray) -> Dict[str, float]:
    """Compute mean, standard deviation, median, 95% confidence interval for a list of scores."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))

    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        ci_margin = stats.t.ppf(0.975, df=n - 1) * se
        ci_lower = float(mean - ci_margin)
        ci_upper = float(mean + ci_margin)
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_samples": n,
    }


def compute_cohens_d(group_a: List[float] | np.ndarray, group_b: List[float] | np.ndarray) -> Dict[str, float | str]:
    """Compute Cohen's d effect size between two groups."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {"cohens_d": 0.0, "interpretation": "Insufficient data"}

    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        d = 0.0
    else:
        d = float((mean_a - mean_b) / pooled_std)

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "Negligible"
    elif abs_d < 0.5:
        interp = "Small"
    elif abs_d < 0.8:
        interp = "Medium"
    else:
        interp = "Large"

    return {"cohens_d": round(d, 4), "interpretation": interp}


def run_statistical_tests(scores_a: List[float] | np.ndarray, scores_b: List[float] | np.ndarray) -> Dict[str, float | str]:
    """Run Paired t-test and Wilcoxon Signed-Rank Test between two score distributions."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    results = {}

    # Paired t-test
    if len(a) == len(b) and len(a) >= 2:
        t_stat, p_val_t = stats.ttest_rel(a, b)
        results["t_statistic"] = round(float(t_stat), 4)
        results["p_value_ttest"] = round(float(p_val_t), 6)
        results["is_significant_ttest"] = bool(p_val_t < 0.05)
    else:
        results["t_statistic"] = 0.0
        results["p_value_ttest"] = 1.0
        results["is_significant_ttest"] = False

    # Wilcoxon Signed-Rank Test
    if len(a) == len(b) and len(a) >= 3 and not np.all(a == b):
        try:
            w_stat, p_val_w = stats.wilcoxon(a, b)
            results["wilcoxon_stat"] = round(float(w_stat), 4)
            results["p_value_wilcoxon"] = round(float(p_val_w), 6)
            results["is_significant_wilcoxon"] = bool(p_val_w < 0.05)
        except Exception:
            results["wilcoxon_stat"] = 0.0
            results["p_value_wilcoxon"] = 1.0
            results["is_significant_wilcoxon"] = False
    else:
        results["wilcoxon_stat"] = 0.0
        results["p_value_wilcoxon"] = 1.0
        results["is_significant_wilcoxon"] = False

    # Cohen's d effect size
    effect = compute_cohens_d(a, b)
    results.update(effect)

    return results


def generate_publication_table(model_stats: Dict[str, Dict]) -> str:
    """Format a Markdown publication table with Mean ± SD, 95% CIs, and significance annotations."""
    table_lines = []
    table_lines.append("| Model Architecture | Robustness Score (Mean ± SD) | 95% Confidence Interval | p-value (vs Best) | Effect Size (Cohen's d) |")
    table_lines.append("| :--- | :--- | :--- | :--- | :--- |")

    sorted_models = sorted(model_stats.items(), key=lambda item: item[1]["descriptive"]["mean"], reverse=True)

    best_model_name = sorted_models[0][0]

    for model_name, data in sorted_models:
        desc = data["descriptive"]
        mean_sd_str = f"**{desc['mean']:.2f} ± {desc['std']:.2f}**"
        ci_str = f"[{desc['ci_lower']:.2f}, {desc['ci_upper']:.2f}]"

        if model_name == best_model_name:
            p_str = "— *(Baseline)*"
            effect_str = "—"
        else:
            comp = data.get("comparison_vs_best", {})
            p_val = comp.get("p_value_ttest", 1.0)
            sig_star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            p_str = f"p = {p_val:.4f} ({sig_star})"
            effect_str = f"{comp.get('cohens_d', 0.0):.2f} ({comp.get('interpretation', 'N/A')})"

        table_lines.append(f"| `{model_name}` | {mean_sd_str} | {ci_str} | {p_str} | {effect_str} |")

    return "\n".join(table_lines)
