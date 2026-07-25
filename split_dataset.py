"""
split_dataset.py  –  Plate-level train/val/test splitter
=========================================================
FIX: Previously split at IMAGE level, which allowed images from the
     SAME plate to appear in both train and test sets — causing severe
     data leakage and inflated accuracy.

FIX 2: Auto-discovers Plate folders recursively so nested dataset
        structures (controlled/uncontrolled subfolders) are handled
        automatically regardless of nesting depth.
"""

import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple
from collections import Counter

# =========================
# Default configuration
# =========================
# Use environment variables when available, otherwise default to repository-relative paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_ROOT = os.getenv(
    "PATHOGEN_RAW_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "data", "A Microbiological Image Repository of Escherichia"),
)
DEFAULT_OUTPUT_ROOT = os.getenv(
    "PATHOGEN_SPLIT_OUTPUT_ROOT",
    os.path.join(PROJECT_ROOT, "dataset_split"),
)
DEFAULT_INPUT_ROOT  = r"C:\Pathogen-intelligence-system\data\A Microbiological Image Repository of Escherichia"
DEFAULT_OUTPUT_ROOT = r"C:\Pathogen-intelligence-system\dataset_split"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".HEIC", ".heic"}

# Maps top-level source folder name → output class name
CLASS_NAME_MAP = {
    "e.coli on BH":                 "e_coli",
    "Klebsiella on BH":             "k_pneumoniae",
    "Pseudomonas aeruginosa on BH": "p_aeruginosa",
    "Staph On BH":                  "s_aureus",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in {".jpg", ".jpeg", ".png", ".heic"}


def find_plate_dirs(root: str) -> List[str]:
    """
    Recursively find all folders whose name starts with 'Plate'
    anywhere under root. This handles any nesting depth.
    """
    plate_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower().startswith("plate"):
                plate_dirs.append(os.path.join(dirpath, d))
    plate_dirs.sort()
    return plate_dirs


def count_images_in_plate(plate_dir: str) -> int:
    count = 0
    for root, _, files in os.walk(plate_dir):
        count += sum(1 for f in files if is_image_file(f))
    return count


def split_plates(plates: List[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    """Split at the PLATE level — no plate ever crosses split boundaries."""
    rng = random.Random(seed)

    if len(plates) == 0:
        return [], [], []
    if len(plates) == 1:
        print("  [WARN] Only 1 plate — placed entirely in train. Val/test will be empty.")
        return plates, [], []
    if len(plates) == 2:
        return [plates[0]], [plates[1]], []

    rng.shuffle(plates)
    plates_with_counts = [(p, count_images_in_plate(p)) for p in plates]
    plates_with_counts.sort(key=lambda x: x[1], reverse=True)

    n       = len(plates_with_counts)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val   = max(1, round(n * VAL_RATIO))
    n_test  = max(1, n - n_train - n_val)

    while n_train + n_val + n_test > n:
        if n_val > 1:   n_val   -= 1
        elif n_train > 1: n_train -= 1
        else:             n_test  -= 1

    all_plates = [p for p, _ in plates_with_counts]
    train_plates, val_plates, test_plates = [], [], []

    # Distribute round-robin so large and small plates are spread evenly
    buckets = (
        [train_plates] * n_train +
        [val_plates]   * n_val   +
        [test_plates]  * n_test
    )
    rng.shuffle(buckets)
    for plate, bucket in zip(all_plates, buckets):
        bucket.append(plate)

    return train_plates, val_plates, test_plates


def copy_plate(plate_dir: str, class_dest_dir: str) -> int:
    """Copy all images from plate_dir into class_dest_dir.
    HEIC files are converted to JPG automatically."""
    from PIL import Image
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        print("  [WARN] pillow-heif not installed. HEIC files will be skipped.")

    plate_name = os.path.basename(plate_dir)
    copied = 0
    for root, _, files in os.walk(plate_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
                continue
            src = os.path.join(root, filename)
            rel = os.path.relpath(src, plate_dir)

            # Convert HEIC to JPG
            if ext.lower() == ".heic":
                rel_jpg = os.path.splitext(rel)[0] + ".jpg"
                dst = os.path.join(class_dest_dir, plate_name, rel_jpg)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    img = Image.open(src)
                    img.convert("RGB").save(dst, "JPEG", quality=95)
                    copied += 1
                except Exception as e:
                    print(f"  [WARN] Could not convert {src}: {e}")
            else:
                dst = os.path.join(class_dest_dir, plate_name, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    return copied

def prepare_output_structure(output_root: str, class_names: List[str]) -> None:
    if os.path.exists(output_root):
        print(f"[WARN] Deleting existing output: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        for cls in class_names:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

def split_dataset(input_root: str, output_root: str, seed: int) -> None:
    random.seed(seed)
    class_names = list(set(CLASS_NAME_MAP.values()))
    prepare_output_structure(output_root, class_names)

    grand_total = {"train": 0, "val": 0, "test": 0}

    for src_cls, out_cls in CLASS_NAME_MAP.items():
        class_src_dir = os.path.join(input_root, src_cls)

        if not os.path.isdir(class_src_dir):
            print(f"[WARN] Source folder not found: {class_src_dir}")
            continue

        # Auto-discover all Plate* folders anywhere under this class dir
        plates = find_plate_dirs(class_src_dir)

        if not plates:
            print(f"[WARN] No Plate folders found under: {class_src_dir}")
            continue

        train_plates, val_plates, test_plates = split_plates(plates, seed)

        print(f"\nClass: {out_cls}  ({len(plates)} plates found under '{src_cls}')")
        print(f"  Train: {len(train_plates)} plates")
        print(f"  Val:   {len(val_plates)} plates")
        print(f"  Test:  {len(test_plates)} plates")

        for split_name, split_plates_list in [
            ("train", train_plates),
            ("val",   val_plates),
            ("test",  test_plates),
        ]:
            dest = os.path.join(output_root, split_name, out_cls)
            n_images = 0
            for plate in split_plates_list:
                n_images += copy_plate(plate, dest)
            grand_total[split_name] += n_images
            print(f"    {split_name}: {n_images} images copied")

    total = sum(grand_total.values())
    print("\n=== Final totals ===")
    for split_name, cnt in grand_total.items():
        pct = 100 * cnt / total if total else 0
        print(f"  {split_name}: {cnt} images ({pct:.1f}%)")
    print(f"  TOTAL: {total} images")
    print(f"\nOutput: {output_root}")
    print("\n[NOTE] Split is at the PLATE level — no leakage between splits.")

    # Warn if val class distribution is badly skewed
    for split_name in ("val", "test"):
        counts = {}
        split_dir = os.path.join(output_root, split_name)
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            n = sum(
                1 for root, _, files in os.walk(cls_dir)
                for f in files if is_image_file(f)
            )
            counts[cls] = n
        if sum(counts.values()) > 0:
            mx = max(counts.values())
            total_split = sum(counts.values())
            if mx / total_split > 0.8:
                print(f"[WARN] {split_name} set is >80% one class: {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split pathogen image dataset at PLATE level (no leakage)."
    )
    parser.add_argument("--input-root",  default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(args.input_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()