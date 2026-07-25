import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple

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


def collect_all_images_from_plates(plate_dirs: List[str]) -> List[Tuple[str, str]]:
    """
    Collect all image file paths from plate directories.
    
    Returns:
        List of tuples (plate_dir, image_relative_path)
    """
    all_images = []
    for plate_dir in plate_dirs:
        for root, _, files in os.walk(plate_dir):
            for filename in files:
                if is_image_file(filename):
                    rel_path = os.path.relpath(os.path.join(root, filename), plate_dir)
                    all_images.append((plate_dir, rel_path))
    return all_images


def allocate_image_splits(images: List[Tuple[str, str]]) -> Tuple[List, List, List]:
    """
    Split images into train/val/test ensuring minimum 1 image in train and val.
    
    Args:
        images: List of (plate_dir, relative_path) tuples
        
    Returns:
        (train_images, val_images, test_images)
    """
    num_images = len(images)
    
    if num_images == 0:
        return [], [], []
    
    if num_images == 1:
        # Only 1 image: put in train, leave val/test empty (will warn user)
        return [images[0]], [], []
    
    if num_images == 2:
        # 2 images: 1 train, 1 val, 0 test
        return [images[0]], [images[1]], []
    
    # For 3+ images, ensure at least 1 in train and 1 in val
    # Step 1: Compute train count (at least 1, respect ratio)
    train_count = max(1, int(num_images * TRAIN_RATIO))
    
    # Step 2: Compute val count (at least 1, respect ratio)
    val_count = max(1, int(num_images * VAL_RATIO))
    
    # Step 3: Check if train + val exceeds total
    remaining = num_images - train_count - val_count

    if remaining < 0:
        val_count = max(1, num_images - train_count - 1)
        remaining = num_images - train_count - val_count

    test_count = remaining
    
    # Step 4: Assign remaining to test (guaranteed non-negative)
    test_count = num_images - train_count - val_count
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    return train_images, val_images, test_images


def copy_single_image(plate_src_dir: str, rel_path: str, class_dest_dir: str) -> None:
    """
    Copy a single image file preserving its relative structure.
    
    Args:
        plate_src_dir: Source plate directory
        rel_path: Relative path of image within plate directory
        class_dest_dir: Destination class directory
    """
    plate_name = os.path.basename(plate_src_dir)
    src_file = os.path.join(plate_src_dir, rel_path)
    
    # Preserve plate folder structure
    dest_file = os.path.join(class_dest_dir, plate_name, rel_path)
    dest_dir = os.path.dirname(dest_file)
    
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_file, dest_file)





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
    """Split dataset at IMAGE level and copy images to train/val/test structure."""
    random.seed(seed)

    normalized_classes = list(CLASS_NAME_MAP.values())
    prepare_output_structure(output_root, normalized_classes)

    # Tracking logs
    per_class_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
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

        # Collect all images from all plates for this class
        all_images = collect_all_images_from_plates(plate_dirs)
        
        if not all_images:
            print(f"[WARNING] No images found for class '{src_class_name}'")
            per_class_stats[out_class_name] = {
                "train": {"plates": 0, "images": 0},
                "val": {"plates": 0, "images": 0},
                "test": {"plates": 0, "images": 0},
            }
            continue
        
        # Shuffle images for random split
        random.shuffle(all_images)
        
        # Split images ensuring minimum requirements
        train_images, val_images, test_images = allocate_image_splits(all_images)
        
        # Warn if validation is empty (only happens with 1 image total)
        if len(all_images) == 1:
            print(f"[WARNING] Class '{src_class_name}' has only 1 image. Placing in train only.")
        
        class_stats = {
            "train": {"plates": 0, "images": len(train_images)},
            "val": {"plates": 0, "images": len(val_images)},
            "test": {"plates": 0, "images": len(test_images)},
        }

        # Copy images to respective splits
        for split_name, split_images in (
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ):
            class_dest_dir = os.path.join(output_root, split_name, out_class_name)
            
            for plate_dir, rel_path in split_images:
                copy_single_image(plate_dir, rel_path, class_dest_dir)
            
            total_split_images[split_name] += len(split_images)

        per_class_stats[out_class_name] = class_stats

    # =========================
    # Logs
    # =========================
    print("\n=== Per-class split summary (image-level splitting) ===")
    for class_name in sorted(per_class_stats.keys()):
        stats = per_class_stats[class_name]
        print(f"\nClass: {class_name}")
        print(f"  Train -> images: {stats['train']['images']}")
        print(f"  Val   -> images: {stats['val']['images']}")
        print(f"  Test  -> images: {stats['test']['images']}")

    total_images = sum(total_split_images.values())

    print("\n=== Split totals ===")
    print(f"Train -> images: {total_split_images['train']}")
    print(f"Val   -> images: {total_split_images['val']}")
    print(f"Test  -> images: {total_split_images['test']}")

    print("\n=== Dataset total ===")
    print(f"Total images copied: {total_images}")
    print(f"\nOutput directory: {output_root}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split image dataset at IMAGE level into train/val/test ensuring minimum images per split."
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
