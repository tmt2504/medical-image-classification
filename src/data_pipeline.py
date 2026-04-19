"""
Data Pipeline for ISIC Skin Cancer Dataset.

Handles dataset loading, preprocessing, augmentation, and
class imbalance management for dermoscopic image classification.

Dataset source: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ISIC 9-class mapping (sorted alphabetically to match folder order)
CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

# Malignant / high-risk classes for clinical flagging
MALIGNANT_CLASSES = {"melanoma", "basal cell carcinoma", "squamous cell carcinoma"}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224


class ISICSkinDataset(Dataset):
    """ISIC Skin Cancer dataset for dermoscopic image classification."""

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split: str = "train") -> transforms.Compose:
    """Get data transforms for a given split with augmentation for training."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(p=0.2),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def download_dataset(data_dir: str = "data") -> str:
    """
    Download the ISIC Skin Cancer dataset from Kaggle using kagglehub.

    Returns:
        Path to the downloaded dataset root directory.
    """
    import kagglehub

    logger.info("Downloading ISIC Skin Cancer dataset from Kaggle...")
    path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    logger.info(f"Dataset downloaded to: {path}")
    return path


def _resolve_data_path(data_dir: str) -> Path:
    """Resolve the dataset root, handling nested Kaggle structure."""
    data_path = Path(data_dir)
    nested = data_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    if nested.exists():
        data_path = nested
    return data_path


def _find_split_dirs(data_path: Path) -> Tuple:
    """Locate Train and Test directories."""
    train_dir = None
    test_dir = None
    for child in data_path.iterdir():
        if child.is_dir() and child.name.lower() == "train":
            train_dir = child
        elif child.is_dir() and child.name.lower() == "test":
            test_dir = child

    if train_dir is None and test_dir is None:
        raise FileNotFoundError(
            f"No Train/ or Test/ directory found in {data_path}. "
            f"Contents: {[c.name for c in data_path.iterdir()]}"
        )
    return train_dir, test_dir


def _load_folder(folder: Path) -> Tuple[list, list]:
    """Load images and labels from a single split folder (Train/ or Test/)."""
    class_to_idx = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
    image_paths = []
    labels = []
    for class_folder in sorted(folder.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name.lower().strip()
        if class_name not in class_to_idx:
            logger.warning(f"Unknown class folder '{class_folder.name}', skipping")
            continue
        label = class_to_idx[class_name]
        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append(str(img_file))
                labels.append(label)
    return image_paths, labels


def load_isic_dataset(data_dir: str) -> Tuple[list, list]:
    """
    Load all images from the ISIC dataset (Train + Test combined).

    Returns:
        image_paths, labels
    """
    data_path = _resolve_data_path(data_dir)
    train_dir, test_dir = _find_split_dirs(data_path)

    image_paths = []
    labels = []
    for split_dir in [train_dir, test_dir]:
        if split_dir is None:
            continue
        paths, lbls = _load_folder(split_dir)
        image_paths.extend(paths)
        labels.extend(lbls)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {data_path}")

    logger.info(f"Loaded {len(image_paths)} images with {len(set(labels))} classes")
    return image_paths, labels


def load_train_test_split(data_dir: str) -> Dict[str, Tuple[list, list]]:
    """
    Load Train/ and Test/ folders separately.

    Returns dict with 'train_all' and 'test' keys.
    """
    data_path = _resolve_data_path(data_dir)
    train_dir, test_dir = _find_split_dirs(data_path)

    result = {}
    if train_dir:
        result["train_all"] = _load_folder(train_dir)
        logger.info(f"Train folder: {len(result['train_all'][0])} images")
    if test_dir:
        result["test"] = _load_folder(test_dir)
        logger.info(f"Test folder: {len(result['test'][0])} images")
    return result


def split_dataset(
    image_paths: list,
    labels: list,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[list, list]]:
    """Split a set of images into train/val with stratification."""
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=random_state,
    )

    logger.info(f"Split sizes - Train: {len(train_paths)}, " f"Val: {len(val_paths)}")

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
    }


def compute_class_weights(labels: list) -> torch.Tensor:
    """Compute inverse frequency class weights for handling imbalance."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts.astype(float) + 1e-6)
    weights = torch.FloatTensor(weights)
    logger.info(f"Class weights: {weights.tolist()}")
    return weights


def get_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """Create weighted random sampler for handling class imbalance."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts.astype(float) + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def _balance_classes(
    image_paths: list,
    labels: list,
    samples_per_class: int = 1000,
) -> Tuple[list, list]:
    """
    Balance classes by oversampling minority classes using Augmentor.

    Generates synthetic images via rotation, flipping, and zoom for
    underrepresented classes to reach samples_per_class per class.

    Args:
        image_paths: List of image file paths
        labels: Corresponding labels
        samples_per_class: Target number of images per class

    Returns:
        Balanced (image_paths, labels)
    """
    import Augmentor
    import shutil

    # Group paths by class
    class_to_paths: Dict[int, list] = {}
    for path, label in zip(image_paths, labels):
        class_to_paths.setdefault(label, []).append(path)

    balanced_paths = []
    balanced_labels = []

    # Use a persistent directory so augmented images are reused across runs
    augmented_dir = Path("data") / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    # Quick check: if ALL classes already have enough augmented images, skip entirely
    all_ready = True
    for label_idx in range(NUM_CLASSES):
        paths = class_to_paths.get(label_idx, [])
        if len(paths) >= samples_per_class:
            continue
        needed = samples_per_class - len(paths)
        out_dir = augmented_dir / f"output_{label_idx}"
        existing_count = len(list(out_dir.glob("*.*"))) if out_dir.exists() else 0
        if existing_count < needed:
            all_ready = False
            break
    if all_ready:
        logger.info("All augmented data already exists, skipping augmentation")
        for label_idx in range(NUM_CLASSES):
            paths = class_to_paths.get(label_idx, [])
            balanced_paths.extend(paths)
            balanced_labels.extend([label_idx] * len(paths))
            needed = samples_per_class - len(paths)
            if needed > 0:
                out_dir = augmented_dir / f"output_{label_idx}"
                for gen_path in sorted(out_dir.glob("*.*"))[:needed]:
                    balanced_paths.append(str(gen_path))
                    balanced_labels.append(label_idx)
        final_counts = np.bincount(balanced_labels, minlength=NUM_CLASSES)
        logger.info(f"Balanced dataset: {len(balanced_paths)} total images")
        for i, name in enumerate(CLASS_NAMES):
            logger.info(f"  {name}: {final_counts[i]}")
        return balanced_paths, balanced_labels

    for label_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[label_idx]
        paths = class_to_paths.get(label_idx, [])
        count = len(paths)

        # Always include all original images
        balanced_paths.extend(paths)
        balanced_labels.extend([label_idx] * count)

        if count >= samples_per_class:
            logger.info(
                f"Class '{class_name}': {count} images (no augmentation needed)"
            )
            continue

        # Need to generate (samples_per_class - count) extra images
        needed = samples_per_class - count

        class_output_dir = augmented_dir / f"output_{label_idx}"

        # Check if augmented images already exist
        existing = (
            sorted(class_output_dir.glob("*.*")) if class_output_dir.exists() else []
        )
        if len(existing) >= needed:
            logger.info(
                f"Class '{class_name}': {count} images, "
                f"reusing {needed} existing augmented images"
            )
            for gen_path in existing[:needed]:
                balanced_paths.append(str(gen_path))
                balanced_labels.append(label_idx)
            continue

        logger.info(
            f"Class '{class_name}': {count} images, "
            f"generating {needed} augmented images"
        )

        # Create directories for Augmentor input/output
        class_input_dir = augmented_dir / f"input_{label_idx}"
        class_input_dir.mkdir(parents=True, exist_ok=True)
        class_output_dir.mkdir(parents=True, exist_ok=True)

        for i, src_path in enumerate(paths):
            ext = Path(src_path).suffix
            dst = class_input_dir / f"img_{i}{ext}"
            if not dst.exists():
                shutil.copy2(src_path, dst)

        # Augment with Augmentor
        # NOTE: Augmentor treats output_directory as relative to source_directory,
        # so we must use an absolute path to get images in the right place.
        p = Augmentor.Pipeline(
            source_directory=str(class_input_dir),
            output_directory=str(class_output_dir.resolve()),
        )
        p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=2)
        p.sample(needed)

        # Collect generated images
        generated = sorted(class_output_dir.glob("*.*"))
        for gen_path in generated[:needed]:
            balanced_paths.append(str(gen_path))
            balanced_labels.append(label_idx)

    # Log final distribution
    final_counts = np.bincount(balanced_labels, minlength=NUM_CLASSES)
    logger.info(f"Balanced dataset: {len(balanced_paths)} total images")
    for i, name in enumerate(CLASS_NAMES):
        logger.info(f"  {name}: {final_counts[i]}")

    return balanced_paths, balanced_labels


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    balance_classes: bool = True,
    samples_per_class: int = 1000,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train/val/test data loaders with preprocessing and augmentation.

    Args:
        data_dir: Path to dataset root
        batch_size: Batch size
        num_workers: DataLoader workers
        use_weighted_sampling: Use weighted random sampler
        balance_classes: Oversample minority classes with Augmentor
        samples_per_class: Target images per class when balancing

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    data = load_train_test_split(data_dir)
    train_all_paths, train_all_labels = data["train_all"]
    test_paths, test_labels = data["test"]

    # Split val BEFORE augmentation to avoid data leakage
    splits = split_dataset(train_all_paths, train_all_labels)
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]

    # Balance only training set
    if balance_classes:
        train_paths, train_labels = _balance_classes(
            train_paths, train_labels, samples_per_class
        )

    # Compute class weights from training set
    class_weights = compute_class_weights(train_labels)

    # Create datasets with appropriate transforms
    train_dataset = ISICSkinDataset(train_paths, train_labels, get_transforms("train"))
    val_dataset = ISICSkinDataset(val_paths, val_labels, get_transforms("val"))
    test_dataset = ISICSkinDataset(test_paths, test_labels, get_transforms("test"))

    # Weighted sampling for training
    train_sampler = (
        get_weighted_sampler(train_labels) if use_weighted_sampling else None
    )

    # Only use pin_memory on CUDA (not supported on MPS/CPU)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        f"DataLoaders created - "
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader, class_weights


def validate_data_quality(data_dir: str) -> Dict[str, any]:
    """Run data quality checks on the dataset."""
    results = {"passed": True, "checks": {}}

    try:
        image_paths, labels = load_isic_dataset(data_dir)
    except (FileNotFoundError, ValueError) as e:
        results["passed"] = False
        results["checks"]["load"] = {"passed": False, "error": str(e)}
        return results

    # Check: minimum dataset size
    min_size = 100
    size_ok = len(image_paths) >= min_size
    results["checks"]["min_size"] = {
        "passed": size_ok,
        "actual": len(image_paths),
        "expected": f">= {min_size}",
    }
    if not size_ok:
        results["passed"] = False

    # Check: all classes represented
    unique_labels = set(labels)
    all_classes = len(unique_labels) == NUM_CLASSES
    results["checks"]["all_classes"] = {
        "passed": all_classes,
        "actual": len(unique_labels),
        "expected": NUM_CLASSES,
    }
    if not all_classes:
        results["passed"] = False

    # Check: no corrupt images (sample check)
    sample_size = min(50, len(image_paths))
    sample_indices = np.random.choice(len(image_paths), sample_size, replace=False)
    corrupt_count = 0
    for idx in sample_indices:
        try:
            img = Image.open(image_paths[idx])
            img.verify()
        except Exception:
            corrupt_count += 1

    results["checks"]["image_integrity"] = {
        "passed": corrupt_count == 0,
        "corrupt_in_sample": corrupt_count,
        "sample_size": sample_size,
    }
    if corrupt_count > 0:
        results["passed"] = False

    # Check: class distribution (no class < 1% of total)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    min_ratio = class_counts.min() / len(labels)
    results["checks"]["class_balance"] = {
        "passed": True,  # Informational, imbalance is expected
        "distribution": {CLASS_NAMES[i]: int(c) for i, c in enumerate(class_counts)},
        "min_class_ratio": float(min_ratio),
        "note": "Class imbalance handled via weighted sampling/loss",
    }

    return results
