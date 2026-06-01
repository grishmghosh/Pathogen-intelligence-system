"""
Full Evaluation Script — Pathogen Intelligence System
Generates confusion matrices, per-class metrics, ROC curves,
confidence distributions, PR curves, and more for both models.

Outputs saved to: results/evaluation/
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = r'C:\Pathogen-intelligence-system\checkpoints'
DATASET_ROOT   = r'C:\Pathogen-intelligence-system\dataset_split'
OUTPUT_DIR     = r'C:\Pathogen-intelligence-system\results\evaluation'
CLASS_NAMES    = ['e_coli', 'k_pneumoniae', 'p_aeruginosa', 's_aureus']
COLORS         = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data loader ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_ROOT, 'test'), transform=transform
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
)

print(f"Test set: {len(test_dataset)} images | Device: {DEVICE}\n")


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_efficientnet(ckpt_dir, device):
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_f, 4))
    ckpt  = torch.load(os.path.join(ckpt_dir, 'efficientnet_b0_best.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    temp  = torch.load(os.path.join(ckpt_dir, 'efficientnet_b0_temperature.pth'), map_location=device)['temperature']
    return model.to(device).eval(), temp

def load_resnet(ckpt_dir, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 4)
    )
    ckpt  = torch.load(os.path.join(ckpt_dir, 'resnet50_best.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    temp  = torch.load(os.path.join(ckpt_dir, 'resnet50_temperature.pth'), map_location=device)['temperature']
    return model.to(device).eval(), temp


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, temperature, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)) / temperature
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = probs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()   # shape: (N, 4)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 1 — Confusion Matrix (counts + normalised side by side)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(cm, class_names, model_name, output_dir):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_name} — Confusion Matrix', fontsize=14, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['{:d}', '{:.1%}'],
        ['Raw Counts', 'Normalised (row %)']
    ):
        im = ax.imshow(data, interpolation='nearest',
                       cmap=plt.cm.Blues, vmin=0, vmax=data.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(class_names)),
               yticks=np.arange(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               title=title, ylabel='True Label', xlabel='Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                txt = fmt.format(val)
                ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                        color='white' if val > thresh else 'black')

    fig.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_1_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 1] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 2 — Per-Class Precision / Recall / F1 (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_metrics(report_dict, class_names, model_name, output_dir):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        values = [report_dict[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_ylim(0.7, 1.08)
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Per-Class Precision / Recall / F1', fontweight='bold')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_2_per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 2] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 3 — ROC Curves (one-vs-rest, per class)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(labels, probs, class_names, model_name, output_dir):
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    fig, axes  = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f'{model_name} — ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')

    for idx, (ax, cls_name, color) in enumerate(zip(axes.flat, class_names, COLORS)):
        fpr, tpr, _ = roc_curve(labels_bin[:, idx], probs[:, idx])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(cls_name, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_3_roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 3] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 4 — Precision-Recall Curves (per class)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pr_curves(labels, probs, class_names, model_name, output_dir):
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    fig, axes  = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f'{model_name} — Precision-Recall Curves', fontsize=13, fontweight='bold')

    for idx, (ax, cls_name, color) in enumerate(zip(axes.flat, class_names, COLORS)):
        prec, rec, _ = precision_recall_curve(labels_bin[:, idx], probs[:, idx])
        ap           = average_precision_score(labels_bin[:, idx], probs[:, idx])
        ax.plot(rec, prec, color=color, lw=2, label=f'AP = {ap:.4f}')
        ax.fill_between(rec, prec, alpha=0.08, color=color)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(cls_name, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_4_pr_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 4] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 5 — Confidence Distribution (per class, violin + box)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confidence_distribution(labels, probs, class_names, model_name, output_dir):
    confs = probs.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Confidence Distribution', fontsize=13, fontweight='bold')

    # Left: per-class violin
    ax = axes[0]
    data_by_class = [confs[labels == i] for i in range(len(class_names))]
    parts = ax.violinplot(data_by_class, positions=range(len(class_names)),
                          showmedians=True, showextrema=True)
    for pc, color in zip(parts['bodies'], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.97, color='red', linestyle='--', alpha=0.6, label='Saturation threshold (0.97)')
    ax.legend(fontsize=8)
    ax.set_title('Per-Class Confidence (Violin)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Right: overall histogram by class
    ax = axes[1]
    for i, (cls_name, color) in enumerate(zip(class_names, COLORS)):
        cls_confs = confs[labels == i]
        ax.hist(cls_confs, bins=20, alpha=0.55, color=color, label=cls_name, density=True)
    ax.axvline(0.97, color='red', linestyle='--', alpha=0.7, label='Saturation (0.97)')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Histogram by Class')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_5_confidence_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 5] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 6 — Model Comparison: EfficientNet vs ResNet side by side
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(reports, accuracies, class_names, output_dir):
    model_names = list(reports.keys())
    metrics     = ['precision', 'recall', 'f1-score']
    fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('EfficientNet-B0 vs ResNet-50 — Per-Class Metrics', fontsize=13, fontweight='bold')

    palette = ['#2196F3', '#E91E63']
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(class_names))
        width = 0.35
        for i, (mname, color) in enumerate(zip(model_names, palette)):
            values = [reports[mname][cls][metric] for cls in class_names]
            bars = ax.bar(x + i * width, values, width, label=mname, color=color, alpha=0.82)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0.75, 1.05)
        ax.set_title(metric.capitalize(), fontweight='bold')
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Add accuracy annotation
    acc_text = '  |  '.join([f'{m}: {a*100:.2f}%' for m, a in accuracies.items()])
    fig.text(0.5, -0.02, f'Test Accuracy — {acc_text}', ha='center', fontsize=10, style='italic')

    fig.tight_layout()
    path = os.path.join(output_dir, 'comparison_6_model_vs_model.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 6] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH 7 — Misclassification Heatmap (which class gets confused with which)
# ══════════════════════════════════════════════════════════════════════════════
def plot_misclassification_heatmap(cms, class_names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Misclassification Heatmap (off-diagonal errors only)', fontsize=13, fontweight='bold')

    for ax, (mname, cm) in zip(axes, cms.items()):
        # Zero out diagonal (correct predictions) to highlight errors only
        cm_err = cm.copy().astype(float)
        np.fill_diagonal(cm_err, 0)
        im = ax.imshow(cm_err, cmap='Reds', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(class_names)),
               yticks=np.arange(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               title=mname, xlabel='Predicted (wrong)', ylabel='True')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j:
                    ax.text(j, i, str(int(cm_err[i, j])), ha='center', va='center',
                            fontsize=11, color='black' if cm_err[i, j] < cm_err.max() / 2 else 'white')

    fig.tight_layout()
    path = os.path.join(output_dir, 'comparison_7_misclassification_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Graph 7] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluate one model — runs all 5 per-model graphs
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model_name, preds, labels, probs, class_names, output_dir):
    confs = probs.max(axis=1)
    acc   = accuracy_score(labels, preds)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Mean Confidence: {confs.mean():.4f}  Std: {confs.std():.4f}")

    report_str  = classification_report(labels, preds, target_names=class_names, digits=4)
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(f"\n  Classification Report:\n{report_str}")

    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n  {cm}\n")

    print(f"  Misclassifications:")
    for ti, tc in enumerate(class_names):
        for pi, pc in enumerate(class_names):
            if ti != pi and cm[ti, pi] > 0:
                print(f"    {tc} → {pc}: {cm[ti, pi]} images")

    safe_name = model_name.replace('-', '_').replace(' ', '_')
    plot_confusion_matrices(cm, class_names, safe_name, output_dir)
    plot_per_class_metrics(report_dict, class_names, safe_name, output_dir)
    plot_roc_curves(labels, probs, class_names, safe_name, output_dir)
    plot_pr_curves(labels, probs, class_names, safe_name, output_dir)
    plot_confidence_distribution(labels, probs, class_names, safe_name, output_dir)

    return report_dict, cm, acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PATHOGEN INTELLIGENCE SYSTEM — FULL EVALUATION")
    print("=" * 60)

    reports, accuracies, cms = {}, {}, {}

    # EfficientNet-B0
    print("\n[1/2] Loading EfficientNet-B0...")
    eff_model, eff_temp = load_efficientnet(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {eff_temp:.4f}")
    eff_preds, eff_labels, eff_probs = run_inference(eff_model, test_loader, eff_temp, DEVICE)
    reports['EfficientNet-B0'], cms['EfficientNet-B0'], accuracies['EfficientNet-B0'] = \
        evaluate_model('EfficientNet-B0', eff_preds, eff_labels, eff_probs, CLASS_NAMES, OUTPUT_DIR)
    del eff_model  # free memory

    # ResNet-50
    print("\n[2/2] Loading ResNet-50...")
    res_model, res_temp = load_resnet(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {res_temp:.4f}")
    res_preds, res_labels, res_probs = run_inference(res_model, test_loader, res_temp, DEVICE)
    reports['ResNet-50'], cms['ResNet-50'], accuracies['ResNet-50'] = \
        evaluate_model('ResNet-50', res_preds, res_labels, res_probs, CLASS_NAMES, OUTPUT_DIR)
    del res_model

    # Cross-model graphs (6 + 7)
    print("\n[Generating cross-model comparison graphs...]")
    plot_model_comparison(reports, accuracies, CLASS_NAMES, OUTPUT_DIR)
    plot_misclassification_heatmap(cms, CLASS_NAMES, OUTPUT_DIR)

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for mname, acc in accuracies.items():
        print(f"  {mname:20s}: {acc*100:.2f}% test accuracy")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {f}")
    print("\nDone.")


if __name__ == '__main__':
    main()
