"""
BeeGuard HiveSense - Utilities & Helper Functions
- tf.data pipeline creation with autotuning and prefetching
- Class weighting computation for imbalanced training
- Comprehensive training/validation loss & accuracy plotting
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from .config import (
        IMAGE_SIZE,
        BATCH_SIZE,
        CLASS_NAMES,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        TRAINING_HISTORY_PLOT,
    )
except ImportError:
    from config import (
        IMAGE_SIZE,
        BATCH_SIZE,
        CLASS_NAMES,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        TRAINING_HISTORY_PLOT,
    )


def build_data_pipelines(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset]]:
    """
    Constructs high-performance tf.data pipelines for Train, Validation, and Test
    splits using categorical labels matching the 7-class taxonomy.
    """
    def create_split_dataset(directory: Path, shuffle: bool):
        if not directory.exists():
            return None
        # Check if there are any images in subdirectories
        has_images = any(directory.rglob("*.png")) or any(directory.rglob("*.jpg"))
        if not has_images:
            return None

        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            seed=42,
        )
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_ds = create_split_dataset(train_dir, shuffle=True)
    val_ds = create_split_dataset(val_dir, shuffle=False)
    test_ds = create_split_dataset(test_dir, shuffle=False)

    return train_ds, val_ds, test_ds


def compute_class_weights(counts: Dict[str, int]) -> Dict[int, float]:
    """
    Computes balanced class weights following the scikit-learn heuristic:
    weight = N / (n_classes * count_per_class)
    """
    active_counts = {c: counts.get(c, 0) for c in CLASS_NAMES if counts.get(c, 0) > 0}
    if not active_counts:
        return {idx: 1.0 for idx in range(len(CLASS_NAMES))}

    total_samples = sum(active_counts.values())
    n_active = len(active_counts)

    weights = {}
    for idx, c in enumerate(CLASS_NAMES):
        cnt = counts.get(c, 0)
        if cnt > 0:
            weights[idx] = float(total_samples / (n_active * cnt))
        else:
            # Classes with 0 samples get neutral weight
            weights[idx] = 1.0

    return weights


def plot_combined_history(
    history1: dict,
    history2: Optional[dict] = None,
    save_path: Path = TRAINING_HISTORY_PLOT,
):
    """
    Plots training and validation accuracy and loss graphs across Phase 1
    (Transfer Learning) and Phase 2 (Fine-Tuning), highlighting the transition epoch.
    """
    # Merge histories if Phase 2 was run
    acc = list(history1.get("accuracy", []))
    val_acc = list(history1.get("val_accuracy", []))
    loss = list(history1.get("loss", []))
    val_loss = list(history1.get("val_loss", []))
    
    transition_epoch = len(acc)

    if history2:
        acc.extend(history2.get("accuracy", []))
        val_acc.extend(history2.get("val_accuracy", []))
        loss.extend(history2.get("loss", []))
        val_loss.extend(history2.get("val_loss", []))

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    if history2 and transition_epoch < len(acc):
        ax1.axvline(x=transition_epoch, color="red", linestyle=":", label="Fine-Tuning Transition")
    ax1.set_title("BeeGuard HiveSense - Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#d62728", linewidth=2)
    if history2 and transition_epoch < len(loss):
        ax2.axvline(x=transition_epoch, color="red", linestyle=":", label="Fine-Tuning Transition")
    ax2.set_title("BeeGuard HiveSense - Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [OK] Training curves saved to: {save_path}")
