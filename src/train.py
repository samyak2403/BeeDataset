"""
BeeGuard HiveSense - End-to-End CNN Training Pipeline
Orchestrates:
1. Two-phase training:
   - Phase 1: Feature extraction (Frozen MobileNetV2 backbone, Adam @ 1e-3)
   - Phase 2: Fine-tuning (Unfrozen top layers, Adam @ 1e-5)
2. Class weight balancing
3. Robust callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
4. Complete loss & accuracy curve visualization
5. Saving best model artifact: bee_disease_model.keras
"""

import os
import sys
import json
import time
from pathlib import Path
import tensorflow as tf

try:
    from .config import (
        RANDOM_SEED,
        PHASE1_EPOCHS,
        PHASE1_LR,
        PHASE2_EPOCHS,
        PHASE2_LR,
        FINE_TUNE_AT_LAYER,
        EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE,
        REDUCE_LR_FACTOR,
        MIN_LR,
        KERAS_MODEL_PATH,
        MODELS_DIR,
        CLASS_NAMES,
    )
    from .model import create_bee_disease_model, prepare_for_finetuning
    from .utils import build_data_pipelines, compute_class_weights, plot_combined_history
    from .dataset_validator import DatasetValidator
except ImportError:
    from config import (
        RANDOM_SEED,
        PHASE1_EPOCHS,
        PHASE1_LR,
        PHASE2_EPOCHS,
        PHASE2_LR,
        FINE_TUNE_AT_LAYER,
        EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE,
        REDUCE_LR_FACTOR,
        MIN_LR,
        KERAS_MODEL_PATH,
        MODELS_DIR,
        CLASS_NAMES,
    )
    from model import create_bee_disease_model, prepare_for_finetuning
    from utils import build_data_pipelines, compute_class_weights, plot_combined_history
    from dataset_validator import DatasetValidator


def run_training_pipeline(
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
    fine_tune_at: int = FINE_TUNE_AT_LAYER,
    save_model_path: Path = KERAS_MODEL_PATH,
):
    """Executes the complete two-phase training lifecycle."""
    print("=" * 70)
    print("BeeGuard HiveSense - MobileNetV2 Training Pipeline Initiated")
    print("=" * 70)

    # 1. Deterministic Seeding
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 2. Run Pre-flight Dataset Validation
    validator = DatasetValidator()
    report = validator.run_all_checks(generate_plots=True)
    counts = report.get("image_counts", {})
    train_counts = counts.get("train", {})
    total_train = sum(train_counts.values())

    if total_train == 0:
        raise RuntimeError(
            "Training dataset contains zero samples. "
            "Please run `python -m src.dataset_prep` to ingest real data before training."
        )

    # 3. Build Data Pipelines
    print("\nBuilding streaming tf.data pipelines...")
    train_ds, val_ds, test_ds = build_data_pipelines()
    if train_ds is None or val_ds is None:
        raise RuntimeError("Train or Validation dataset could not be constructed.")

    # 4. Class Weights for Handling Severe Imbalance
    class_weights = compute_class_weights(train_counts)
    print("Computed Class Weights:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  [{idx}] {name:<22}: {class_weights[idx]:.4f}")

    # 5. Callbacks
    save_model_path = Path(save_model_path)
    save_model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(save_model_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1,
    )

    # 6. Build Initial Model (Phase 1: Feature Extraction)
    print("\n[Phase 1] Initializing MobileNetV2 with ImageNet Weights (Backbone Frozen)...")
    model = create_bee_disease_model(learning_rate=PHASE1_LR)
    model.summary()

    start_time = time.time()
    print(f"\n[Phase 1] Commencing Head Training for up to {phase1_epochs} epochs...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        class_weight=class_weights,
        verbose=1,
    )

    # 7. Phase 2: Fine-Tuning
    history2 = None
    if phase2_epochs > 0:
        print(f"\n[Phase 2] Unfreezing MobileNetV2 from Layer {fine_tune_at} for Fine-Tuning...")
        model = prepare_for_finetuning(model, fine_tune_at=fine_tune_at, learning_rate=PHASE2_LR)

        fine_tune_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        fine_tune_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        print(f"\n[Phase 2] Commencing Fine-Tuning for up to {phase2_epochs} epochs...")
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase2_epochs,
            callbacks=[fine_tune_checkpoint, fine_tune_early_stop, reduce_lr_cb],
            class_weight=class_weights,
            verbose=1,
        )

    elapsed = time.time() - start_time
    print(f"\n[OK] Training Complete in {elapsed / 60:.2f} minutes.")
    print(f"    Best model saved to: {save_model_path}")

    # 8. Plot Combined Training Curves
    plot_combined_history(history1.history, history2.history if history2 else None)

    return model, history1, history2


if __name__ == "__main__":
    run_training_pipeline()
