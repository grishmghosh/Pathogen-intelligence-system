"""
split_dataset.py  –  Plate-level train/val/test splitter
=========================================================
FIX: Previously split at IMAGE level, which allowed images from the
     SAME plate to appear in both train and test sets — causing severe
     data leakage and inflated accuracy.

This version splits at the PLATE level:
  • All images from a plate go exclusively into ONE split.
  • No plate ever straddles train/val/test.
  • s_aureus class added (was silently missing from CLASS_NAME_MAP).
"""

import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple

# =========================
# Default configuration
# =========================
DEFAULT_INPUT_ROOT = r"C:\Pathogen-intelligence-system\data\A Microbiological Image Repository of Escherichia"
DEFAULT_OUTPUT_ROOT = r"C:\Pathogen-intelligence-system\dataset_split"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# FIX: s_aureus was listed in README but missing from this map.
# Added it back; update the key to match your actual folder name.
CLASS_NAME_MAP = {
    "e.coli on BH":                "e_coli",
    "Klebsiella on BH":            "k_pneumoniae",
    "Pseudomonas aeruginosa on BH": "p_aeruginosa",
    "Staph On BH":                 "s_aureus",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def list_plate_directories(class_dir: str) -> List[str]:
    if not os.path.isdir(class_dir):
        return []
    plates = [e.path for e in os.scandir(class_dir) if e.is_dir()]
    plates.sort()
    return plates


def count_images_in_plate(plate_dir: str) -> int:
    count = 0
    for root, _, files in os.walk(plate_dir):
        count += sum(1 for f in files if is_image_file(f))
    return count


def split_plates(plates: List[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    FIX: Split at the PLATE level so no plate crosses split boundaries.

    Strategy
    --------
    Sort plates by descending image count so large plates are distributed
    greedily (avoids one split getting all big plates), then assign
    the remainder randomly to hit the target ratios.
    """
    rng = random.Random(seed)

    if len(plates) == 0:
        return [], [], []
    if len(plates) == 1:
        print("  [WARN] Only 1 plate — placed entirely in train. Val/test will be empty.")
        return plates, [], []
    if len(plates) == 2:
        return [plates[0]], [plates[1]], []

    # Shuffle first, then sort by image count for greedy distribution
    rng.shuffle(plates)
    plates_with_counts = [(p, count_images_in_plate(p)) for p in plates]
    plates_with_counts.sort(key=lambda x: x[1], reverse=True)

    n = len(plates_with_counts)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val   = max(1, round(n * VAL_RATIO))
    # Guarantee at least 1 plate in test if we have ≥3 plates
    n_test  = max(1, n - n_train - n_val)

    # Re-adjust if rounding pushed us over
    while n_train + n_val + n_test > n:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        else:
            n_test -= 1

    all_plates = [p for p, _ in plates_with_counts]
    # Interleave so each split gets a mix of large and small plates
    train_plates, val_plates, test_plates = [], [], []
    buckets = [train_plates] * n_train + [val_plates] * n_val + [test_plates] * n_test
    # Distribute round-robin so large plates are spread
    targets = (
        [train_plates] * n_train
        + [val_plates]  * n_val
        + [test_plates] * n_test
    )
    rng.shuffle(targets)
    for plate, target in zip(all_plates, targets):
        target.append(plate)

    return train_plates, val_plates, test_plates


def copy_plate(plate_dir: str, class_dest_dir: str) -> int:
    """Copy all images from plate_dir into class_dest_dir, preserving structure."""
    plate_name = os.path.basename(plate_dir)
    copied = 0
    for root, _, files in os.walk(plate_dir):
        for filename in files:
            if not is_image_file(filename):
                continue
            src = os.path.join(root, filename)
            rel = os.path.relpath(src, plate_dir)
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
    """
    Plate-level split: every plate goes entirely into train, val, or test.
    This prevents the leakage that caused inflated accuracy scores.
    """
    random.seed(seed)
    class_names = list(CLASS_NAME_MAP.values())
    prepare_output_structure(output_root, class_names)

    grand_total = {"train": 0, "val": 0, "test": 0}

    for src_cls, out_cls in CLASS_NAME_MAP.items():
        class_src_dir = os.path.join(input_root, src_cls)
        plates = list_plate_directories(class_src_dir)

        if not plates:
            print(f"[WARN] No plates for '{src_cls}' at: {class_src_dir}")
            continue

        train_plates, val_plates, test_plates = split_plates(plates, seed)

        print(f"\nClass: {out_cls}  ({len(plates)} plates total)")
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
