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
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Input class folder -> normalized output class folder
CLASS_NAME_MAP = {
    "e.coli on BH": "e_coli",
    "Klebsiella on BH": "k_pneumoniae",
    "Pseudomonas aeruginosa": "p_aeruginosa",
    "Staph On BH": "s_aureus",
}


def is_image_file(filename: str) -> bool:
    """Return True if file extension is an allowed image extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def list_plate_directories(class_dir: str) -> List[str]:
    """List immediate subdirectories (plates) under a class directory."""
    if not os.path.isdir(class_dir):
        return []
    plates = [entry.path for entry in os.scandir(class_dir) if entry.is_dir()]
    plates.sort()
    return plates


def allocate_split_counts(num_plates: int) -> Tuple[int, int, int]:
    """
    Allocate plate counts for train/val/test with ratios 70/15/15.

    Edge-case policy:
    - 0 plates -> (0, 0, 0)
    - 1 plate  -> (1, 0, 0)
    - 2 plates -> (1, 1, 0)
    - >=3 plates -> at least 1 plate in each split
    """
    if num_plates <= 0:
        return 0, 0, 0
    if num_plates == 1:
        return 1, 0, 0
    if num_plates == 2:
        return 1, 1, 0

    # Initial ratio-based counts
    train_count = int(num_plates * TRAIN_RATIO)
    val_count = int(num_plates * VAL_RATIO)
    test_count = num_plates - train_count - val_count

    # Ensure minimum one plate in each split for n >= 3
    train_count = max(train_count, 1)
    val_count = max(val_count, 1)
    test_count = max(test_count, 1)

    counts = [train_count, val_count, test_count]
    targets = [num_plates * TRAIN_RATIO, num_plates * VAL_RATIO, num_plates * TEST_RATIO]

    # If total is too high, remove from split with largest surplus while keeping min 1
    while sum(counts) > num_plates:
        surpluses = [(counts[i] - targets[i], i) for i in range(3) if counts[i] > 1]
        if not surpluses:
            break
        _, idx = max(surpluses, key=lambda x: x[0])
        counts[idx] -= 1

    # If total is too low, add to split with largest deficit
    while sum(counts) < num_plates:
        deficits = [(targets[i] - counts[i], i) for i in range(3)]
        _, idx = max(deficits, key=lambda x: x[0])
        counts[idx] += 1

    return counts[0], counts[1], counts[2]


def copy_plate_images(plate_src_dir: str, class_dest_dir: str) -> int:
    """
    Copy allowed image files from a single plate directory into class destination,
    preserving plate folder name and any nested relative folder structure.

    Returns:
        Number of copied image files.
    """
    copied_images = 0
    plate_name = os.path.basename(plate_src_dir)
    plate_dest_root = os.path.join(class_dest_dir, plate_name)

    for root, _, files in os.walk(plate_src_dir):
        rel_path = os.path.relpath(root, plate_src_dir)
        dest_dir = plate_dest_root if rel_path == "." else os.path.join(plate_dest_root, rel_path)

        for filename in files:
            if not is_image_file(filename):
                continue

            os.makedirs(dest_dir, exist_ok=True)
            src_file = os.path.join(root, filename)
            dst_file = os.path.join(dest_dir, filename)
            shutil.copy2(src_file, dst_file)
            copied_images += 1

    return copied_images


def prepare_output_structure(output_root: str, normalized_classes: List[str]) -> None:
    """
    Prepare split folder tree.

    If output_root exists, print warning and delete it before recreating.
    """
    if os.path.exists(output_root):
        print(f"[WARNING] Output directory already exists and will be deleted: {output_root}")
        shutil.rmtree(output_root)

    for split_name in ("train", "val", "test"):
        for class_name in normalized_classes:
            os.makedirs(os.path.join(output_root, split_name, class_name), exist_ok=True)


def split_dataset(input_root: str, output_root: str, seed: int) -> None:
    """Split dataset at PLATE level and copy images to train/val/test structure."""
    random.seed(seed)

    normalized_classes = list(CLASS_NAME_MAP.values())
    prepare_output_structure(output_root, normalized_classes)

    # Tracking logs
    per_class_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    total_split_plates = {"train": 0, "val": 0, "test": 0}
    total_split_images = {"train": 0, "val": 0, "test": 0}

    for src_class_name, out_class_name in CLASS_NAME_MAP.items():
        class_src_dir = os.path.join(input_root, src_class_name)
        plate_dirs = list_plate_directories(class_src_dir)

        if not plate_dirs:
            print(f"[WARNING] No plates found for class '{src_class_name}' at: {class_src_dir}")
            per_class_stats[out_class_name] = {
                "train": {"plates": 0, "images": 0},
                "val": {"plates": 0, "images": 0},
                "test": {"plates": 0, "images": 0},
            }
            continue

        random.shuffle(plate_dirs)

        n_train, n_val, n_test = allocate_split_counts(len(plate_dirs))
        train_plates = plate_dirs[:n_train]
        val_plates = plate_dirs[n_train:n_train + n_val]
        test_plates = plate_dirs[n_train + n_val:n_train + n_val + n_test]

        class_stats = {
            "train": {"plates": len(train_plates), "images": 0},
            "val": {"plates": len(val_plates), "images": 0},
            "test": {"plates": len(test_plates), "images": 0},
        }

        for split_name, selected_plates in (
            ("train", train_plates),
            ("val", val_plates),
            ("test", test_plates),
        ):
            class_dest_dir = os.path.join(output_root, split_name, out_class_name)

            for plate_dir in selected_plates:
                copied_count = copy_plate_images(plate_dir, class_dest_dir)
                class_stats[split_name]["images"] += copied_count
                total_split_images[split_name] += copied_count

            total_split_plates[split_name] += class_stats[split_name]["plates"]

        per_class_stats[out_class_name] = class_stats

    # =========================
    # Logs
    # =========================
    print("\n=== Per-class split summary (plate-level, no leakage) ===")
    for class_name in sorted(per_class_stats.keys()):
        stats = per_class_stats[class_name]
        print(f"\nClass: {class_name}")
        print(f"  Train -> plates: {stats['train']['plates']}, images: {stats['train']['images']}")
        print(f"  Val   -> plates: {stats['val']['plates']}, images: {stats['val']['images']}")
        print(f"  Test  -> plates: {stats['test']['plates']}, images: {stats['test']['images']}")

    total_plates = sum(total_split_plates.values())
    total_images = sum(total_split_images.values())

    print("\n=== Split totals ===")
    print(f"Train -> plates: {total_split_plates['train']}, images: {total_split_images['train']}")
    print(f"Val   -> plates: {total_split_plates['val']}, images: {total_split_images['val']}")
    print(f"Test  -> plates: {total_split_plates['test']}, images: {total_split_images['test']}")

    print("\n=== Dataset total ===")
    print(f"Total plates: {total_plates}")
    print(f"Total images copied: {total_images}")
    print(f"\nOutput directory: {output_root}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split image dataset at PLATE level into train/val/test without leakage."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Input dataset root path.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output split dataset root path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for plate shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(args.input_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()
