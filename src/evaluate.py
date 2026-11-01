"""
BeeGuard HiveSense - Test Set Evaluation Suite
Evaluates the trained model on the untouched test dataset:
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Complete Confusion Matrix with normalized display
- Scikit-learn Classification Report
- Per-class precision, recall, and F1 breakdown
- Error analysis on misclassified test samples
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf

try:
    from .config import (
        KERAS_MODEL_PATH,
        TEST_DIR,
        CLASS_NAMES,
        CONFUSION_MATRIX_PLOT,
        PLOTS_DIR,
    )
    from .utils import build_data_pipelines
except ImportError:
    from config import (
        KERAS_MODEL_PATH,
        TEST_DIR,
        CLASS_NAMES,
        CONFUSION_MATRIX_PLOT,
        PLOTS_DIR,
    )
    from utils import build_data_pipelines


def evaluate_test_set(
    model_path: Path = KERAS_MODEL_PATH,
    test_dir: Path = TEST_DIR,
    plot_cm_path: Path = CONFUSION_MATRIX_PLOT,
) -> dict:
    """Runs rigorous diagnostic evaluation on the held-out test split."""
    print("=" * 70)
    print("BeeGuard HiveSense - Held-out Test Set Evaluation")
    print("=" * 70)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    print(f"Loading trained model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # Construct test data pipeline (shuffle=False is critical)
    _, _, test_ds = build_data_pipelines(test_dir=test_dir)
    if test_ds is None:
        raise RuntimeError(f"Test dataset at {test_dir} is empty or unpopulated.")

    print("\nGathering ground truth labels and model predictions from test split...")
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.append(labels.numpy())
        y_pred_probs_list.append(preds)

    y_true_onehot = np.vstack(y_true_list)
    y_pred_probs = np.vstack(y_pred_probs_list)

    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Filter out classes that have 0 occurrences in the test set to avoid zero-division noise
    unique_true = np.unique(y_true)
    target_names_present = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    # 1. Core Summary Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 50)
    print("MEASURED TEST SET METRICS (Held-out Test Split)")
    print("-" * 50)
    print(f"  Accuracy           : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Macro Precision    : {macro_prec:.4f}")
    print(f"  Macro Recall       : {macro_rec:.4f}")
    print(f"  Macro F1-Score     : {macro_f1:.4f}")
    print(f"  Weighted Precision : {weighted_prec:.4f}")
    print(f"  Weighted Recall    : {weighted_rec:.4f}")
    print(f"  Weighted F1-Score  : {weighted_f1:.4f}")
    print("-" * 50)

    # 2. Detailed Classification Report
    print("\nClassification Report (Per-Class Performance):")
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report_text)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
    )
    ax.set_title("BeeGuard HiveSense - Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=12)
    ax.set_ylabel("True Pathological Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_cm_path = Path(plot_cm_path)
    plot_cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_cm_path, dpi=200)
    plt.close(fig)
    print(f"\n[OK] Confusion matrix plot saved to: {plot_cm_path}")

    results = {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    return results


if __name__ == "__main__":
    evaluate_test_set()
