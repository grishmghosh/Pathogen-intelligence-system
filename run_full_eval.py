"""
Full Evaluation Script — Pathogen Intelligence System
Generates confusion matrices, per-class metrics, ROC curves,
confidence distributions, PR curves, and more for both models.

Outputs saved to: results/evaluation/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_DATASET_ROOT = Path(r"F:\Pathogen-intelligence-system-old\dataset_split")
if not (DEFAULT_DATASET_ROOT / "test").exists():
    DEFAULT_DATASET_ROOT = Path(r"C:\Pathogen-intelligence-system\dataset_split")
if not (DEFAULT_DATASET_ROOT / "test").exists():
    DEFAULT_DATASET_ROOT = ROOT_DIR / "dataset_split"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "evaluation"
CLASS_NAMES = ["e_coli", "k_pneumoniae", "p_aeruginosa", "s_aureus"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Root directory containing train/val/test splits")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Directory containing trained checkpoints")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated evaluation plots")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    return parser.parse_args()


args = parse_args()
DEVICE = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
CHECKPOINT_DIR = Path(args.checkpoint_dir).expanduser().resolve()
DATASET_ROOT = Path(args.dataset_root).expanduser().resolve()
OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def create_demo_dataset(dataset_root: Path) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / "test" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            image[:, :, 0] = 60 + (index * 12)
            image[:, :, 1] = 80 + (index * 10)
            image[:, :, 2] = 120 + (index * 8)
            image[40:180, 40:180] = 255
            image[60:160, 60:160] = 0
            image[80:140, 80:140] = 200
            PILImage.fromarray(image).save(class_dir / f"{class_name}_{index}.png")
    return dataset_root


if not DATASET_ROOT.exists() or not (DATASET_ROOT / "test").exists():
    print(f"[WARN] Dataset not found at {DATASET_ROOT}. Creating a synthetic demo dataset for evaluation.")
    DATASET_ROOT = create_demo_dataset(ROOT_DIR / "results" / "demo_dataset")

print(f"Test set: {len(list((DATASET_ROOT / 'test').glob('**/*')))} images | Device: {DEVICE}\n")


def _is_lfs_pointer(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def load_efficientnet(ckpt_dir, device):
    from torchvision import models as tvm

    model = tvm.efficientnet_b0(weights="DEFAULT")
    in_f = int(model.classifier[1].in_features)
    model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_f, 4))

    ckpt_path = ckpt_dir / "efficientnet_b0_best.pth"
    temp_path = ckpt_dir / "efficientnet_b0_temperature.pth"
    if ckpt_path.exists() and not _is_lfs_pointer(ckpt_path):
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  [INFO] Loaded EfficientNet checkpoint from {ckpt_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load EfficientNet checkpoint: {exc}")
    else:
        print(f"  [WARN] EfficientNet checkpoint unavailable; using pretrained ImageNet initialization")

    temp = 1.0
    if temp_path.exists():
        try:
            temp = torch.load(str(temp_path), map_location=device, weights_only=False)["temperature"]
        except Exception as exc:
            print(f"  [WARN] Could not load EfficientNet temperature: {exc}")
    return model.to(device).eval(), temp


def load_resnet(ckpt_dir, device):
    from torchvision import models as tvm

    model = tvm.resnet50(weights="DEFAULT")
    in_feat = int(model.fc.in_features)
    setattr(model, "fc", nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_feat, 4)))

    ckpt_path = ckpt_dir / "resnet50_best.pth"
    temp_path = ckpt_dir / "resnet50_temperature.pth"
    if ckpt_path.exists() and not _is_lfs_pointer(ckpt_path):
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  [INFO] Loaded ResNet checkpoint from {ckpt_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load ResNet checkpoint: {exc}")
    else:
        print(f"  [WARN] ResNet checkpoint unavailable; using pretrained ImageNet initialization")

    temp = 1.0
    if temp_path.exists():
        try:
            temp = torch.load(str(temp_path), map_location=device, weights_only=False)["temperature"]
        except Exception as exc:
            print(f"  [WARN] Could not load ResNet temperature: {exc}")
    return model.to(device).eval(), temp


def load_swin(ckpt_dir, device):
    from models.swin_setup import build_swin_t

    model = build_swin_t(num_classes=4)

    ckpt_path = ckpt_dir / "swin_t_best.pth"
    temp_path = ckpt_dir / "swin_t_temperature.pth"
    if ckpt_path.exists() and not _is_lfs_pointer(ckpt_path):
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  [INFO] Loaded Swin-T checkpoint from {ckpt_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load Swin-T checkpoint: {exc}")
    else:
        print(f"  [WARN] Swin-T checkpoint unavailable; using pretrained ImageNet initialization")

    temp = 1.0
    if temp_path.exists():
        try:
            temp = torch.load(str(temp_path), map_location=device, weights_only=False)["temperature"]
        except Exception as exc:
            print(f"  [WARN] Could not load Swin-T temperature: {exc}")
    return model.to(device).eval(), temp


def load_convnext(ckpt_dir, device):
    from torchvision import models as tvm

    model = tvm.convnext_tiny(weights="DEFAULT")
    in_feat = int(model.classifier[2].in_features)
    model.classifier[2] = nn.Linear(in_feat, 4)

    ckpt_path = ckpt_dir / "convnext_tiny_best.pth"
    temp_path = ckpt_dir / "convnext_tiny_temperature.pth"
    if ckpt_path.exists() and not _is_lfs_pointer(ckpt_path):
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  [INFO] Loaded ConvNeXt-Tiny checkpoint from {ckpt_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load ConvNeXt-Tiny checkpoint: {exc}")
    else:
        print(f"  [WARN] ConvNeXt-Tiny checkpoint unavailable; using pretrained ImageNet initialization")

    temp = 1.0
    if temp_path.exists():
        try:
            temp = torch.load(str(temp_path), map_location=device, weights_only=False)["temperature"]
        except Exception as exc:
            print(f"  [WARN] Could not load ConvNeXt-Tiny temperature: {exc}")
    return model.to(device).eval(), temp


def run_inference(model, loader, temperature, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)) / temperature
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()
    )


def plot_confusion_matrices(cm, class_names, model_name, output_dir):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["{:d}", "{:.1%}"],
        ["Raw Counts", "Normalised (row %)"]
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=data.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, title=title, ylabel="True Label", xlabel="Predicted Label")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                txt = fmt.format(val)
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, color="white" if val > thresh else "black")

    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_1_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 1] Saved: {path}")


def plot_per_class_metrics(report_dict, class_names, model_name, output_dir):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    all_values = [report_dict[cls][metric] for cls in class_names for metric in metrics]

    # Dynamic y-axis scaling so low values or zeros are never clipped below axis
    min_v = min(all_values) if all_values else 0.0
    max_v = max(all_values) if all_values else 1.0
    ymin = 0.0 if min_v < 0.65 else max(0.0, min_v - 0.1)
    ymax = min(1.01, max_v + 0.06)  # cap at 101% so axis never exceeds 100%
    ax.set_ylim(ymin, ymax)

    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        values = [report_dict[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)
        offset = 0.015 * (ymax - ymin)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Per-Class Precision / Recall / F1", fontweight="bold")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_2_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 2] Saved: {path}")


def plot_roc_curves(labels, probs, class_names, model_name, output_dir):
    labels_bin = np.asarray(label_binarize(labels, classes=list(range(len(class_names)))))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{model_name} — ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")

    for idx, (ax, cls_name, color) in enumerate(zip(axes.flat, class_names, COLORS)):
        fpr, tpr, _ = roc_curve(labels_bin[:, idx], probs[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(cls_name, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_3_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 3] Saved: {path}")


def plot_pr_curves(labels, probs, class_names, model_name, output_dir):
    labels_bin = np.asarray(label_binarize(labels, classes=list(range(len(class_names)))))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{model_name} — Precision-Recall Curves", fontsize=13, fontweight="bold")

    for idx, (ax, cls_name, color) in enumerate(zip(axes.flat, class_names, COLORS)):
        prec, rec, _ = precision_recall_curve(labels_bin[:, idx], probs[:, idx])
        ap = average_precision_score(labels_bin[:, idx], probs[:, idx])
        ax.plot(rec, prec, color=color, lw=2, label=f"AP = {ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.08, color=color)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(cls_name, fontweight="bold")
        ax.legend(loc="lower left")
        ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_4_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 4] Saved: {path}")


def plot_confidence_distribution(labels, probs, class_names, model_name, output_dir):
    confs = probs.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Confidence Distribution", fontsize=13, fontweight="bold")

    ax = axes[0]
    data_by_class = [confs[labels == i] for i in range(len(class_names))]
    parts = ax.violinplot(data_by_class, positions=range(len(class_names)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.97, color="red", linestyle="--", alpha=0.6, label="Saturation threshold (0.97)")
    ax.legend(fontsize=8)
    ax.set_title("Per-Class Confidence (Violin)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    for i, (cls_name, color) in enumerate(zip(class_names, COLORS)):
        cls_confs = confs[labels == i]
        ax.hist(cls_confs, bins=20, alpha=0.55, color=color, label=cls_name, density=True)
    ax.axvline(0.97, color="red", linestyle="--", alpha=0.7, label="Saturation (0.97)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Histogram by Class")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_5_confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 5] Saved: {path}")


def plot_model_comparison(reports, accuracies, class_names, output_dir):
    model_names = list(reports.keys())
    metrics = ["precision", "recall", "f1-score"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{len(model_names)}-Model Ensemble Comparison — Per-Class Metrics", fontsize=13, fontweight="bold")

    palette = ["#2196F3", "#E91E63", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"][:len(model_names)]
    n_models = len(model_names)
    total_group_width = 0.8
    bar_width = total_group_width / n_models

    all_vals = [
        reports[mname][cls][metric]
        for mname in model_names
        for cls in class_names
        for metric in metrics
    ]
    min_v = min(all_vals) if all_vals else 0.0
    max_v = max(all_vals) if all_vals else 1.0
    ymin = 0.0 if min_v < 0.65 else max(0.0, min_v - 0.1)
    ymax = min(1.15, max_v + 0.1)

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(class_names))
        for i, (mname, color) in enumerate(zip(model_names, palette)):
            values = [reports[mname][cls][metric] for cls in class_names]
            pos = x + (i * bar_width) - (total_group_width / 2) + (bar_width / 2)
            bars = ax.bar(pos, values, bar_width, label=mname, color=color, alpha=0.82)
            offset = 0.012 * (ymax - ymin)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, f"{val:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    acc_text = "  |  ".join([f"{m}: {a * 100:.2f}%" for m, a in accuracies.items()])
    fig.text(0.5, -0.04, f"Test Accuracy — {acc_text}", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    path = os.path.join(output_dir, "comparison_6_model_vs_model.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 6] Saved: {path}")


def plot_misclassification_heatmap(cms, class_names, output_dir):
    n_models = len(cms)
    if n_models <= 2:
        nrows, ncols = 1, n_models
        figsize = (14, 5)
    else:
        nrows = (n_models + 1) // 2
        ncols = 2
        figsize = (12, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle("Misclassification Heatmap (off-diagonal errors only)", fontsize=13, fontweight="bold")

    axes_flat = np.atleast_1d(axes).flat
    for idx, (mname, cm) in enumerate(cms.items()):
        ax = axes_flat[idx]
        cm_err = cm.copy().astype(float)
        np.fill_diagonal(cm_err, 0)
        im = ax.imshow(cm_err, cmap="Reds", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, title=mname, xlabel="Predicted (wrong)", ylabel="True")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        max_err = cm_err.max()
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j:
                    ax.text(j, i, str(int(cm_err[i, j])), ha="center", va="center", fontsize=11, color="black" if cm_err[i, j] < (max_err / 2 if max_err > 0 else 1) else "white")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "comparison_7_misclassification_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Graph 7] Saved: {path}")


def evaluate_model(model_name, preds, labels, probs, class_names, output_dir):
    confs = probs.max(axis=1)
    acc = accuracy_score(labels, preds)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Mean Confidence: {confs.mean():.4f}  Std: {confs.std():.4f}")

    report_str = classification_report(labels, preds, target_names=class_names, digits=4)
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(f"\n  Classification Report:\n{report_str}")

    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n  {cm}\n")

    print("  Misclassifications:")
    for ti, tc in enumerate(class_names):
        for pi, pc in enumerate(class_names):
            if ti != pi and cm[ti, pi] > 0:
                print(f"    {tc} -> {pc}: {cm[ti, pi]} images")

    safe_name = model_name.replace("-", "_").replace(" ", "_")
    plot_confusion_matrices(cm, class_names, safe_name, output_dir)
    plot_per_class_metrics(report_dict, class_names, safe_name, output_dir)
    plot_roc_curves(labels, probs, class_names, safe_name, output_dir)
    plot_pr_curves(labels, probs, class_names, safe_name, output_dir)
    plot_confidence_distribution(labels, probs, class_names, safe_name, output_dir)

    return report_dict, cm, acc


def create_demo_dataset(dataset_root: Path) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / "test" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            image[:, :, 0] = 60 + (index * 12)
            image[:, :, 1] = 80 + (index * 10)
            image[:, :, 2] = 120 + (index * 8)
            image[40:180, 40:180] = 255
            image[60:160, 60:160] = 0
            image[80:140, 80:140] = 200
            PILImage.fromarray(image).save(class_dir / f"{class_name}_{index}.png")
    return dataset_root


try:
    import PIL.Image as Image
except Exception:
    Image = None


def main():
    print("=" * 60)
    print("  PATHOGEN INTELLIGENCE SYSTEM — FULL EVALUATION")
    print("=" * 60)

    test_dataset = datasets.ImageFolder(str(DATASET_ROOT / "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    reports, accuracies, cms = {}, {}, {}

    print("\n[1/4] Loading EfficientNet-B0...")
    eff_model, eff_temp = load_efficientnet(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {eff_temp:.4f}")
    eff_preds, eff_labels, eff_probs = run_inference(eff_model, test_loader, eff_temp, DEVICE)
    reports["EfficientNet-B0"], cms["EfficientNet-B0"], accuracies["EfficientNet-B0"] = evaluate_model("EfficientNet-B0", eff_preds, eff_labels, eff_probs, CLASS_NAMES, OUTPUT_DIR)
    del eff_model

    print("\n[2/4] Loading ResNet-50...")
    res_model, res_temp = load_resnet(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {res_temp:.4f}")
    res_preds, res_labels, res_probs = run_inference(res_model, test_loader, res_temp, DEVICE)
    reports["ResNet-50"], cms["ResNet-50"], accuracies["ResNet-50"] = evaluate_model("ResNet-50", res_preds, res_labels, res_probs, CLASS_NAMES, OUTPUT_DIR)
    del res_model

    print("\n[3/4] Loading Swin-T...")
    swin_model, swin_temp = load_swin(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {swin_temp:.4f}")
    swin_preds, swin_labels, swin_probs = run_inference(swin_model, test_loader, swin_temp, DEVICE)
    reports["Swin-T"], cms["Swin-T"], accuracies["Swin-T"] = evaluate_model("Swin-T", swin_preds, swin_labels, swin_probs, CLASS_NAMES, OUTPUT_DIR)
    del swin_model

    print("\n[4/4] Loading ConvNeXt-Tiny...")
    cnx_model, cnx_temp = load_convnext(CHECKPOINT_DIR, DEVICE)
    print(f"  Temperature: {cnx_temp:.4f}")
    cnx_preds, cnx_labels, cnx_probs = run_inference(cnx_model, test_loader, cnx_temp, DEVICE)
    reports["ConvNeXt-Tiny"], cms["ConvNeXt-Tiny"], accuracies["ConvNeXt-Tiny"] = evaluate_model("ConvNeXt-Tiny", cnx_preds, cnx_labels, cnx_probs, CLASS_NAMES, OUTPUT_DIR)
    del cnx_model

    print("\n[Generating cross-model comparison graphs...]")
    plot_model_comparison(reports, accuracies, CLASS_NAMES, OUTPUT_DIR)
    plot_misclassification_heatmap(cms, CLASS_NAMES, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for mname, acc in accuracies.items():
        print(f"  {mname:20s}: {acc * 100:.2f}% test accuracy")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("  Files generated:")
    for file_name in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {file_name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
