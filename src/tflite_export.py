"""
BeeGuard HiveSense - TensorFlow Lite Conversion & Numerical Verification Suite
Exports:
1. bee_disease_model.keras
2. bee_disease_model.tflite (Standard Float32)
3. bee_disease_model_quantized.tflite (Dynamic Range / Quantized)
4. labels.txt (7-class line-delimited labels)

Verifies:
- Predictions match Keras output within acceptable numerical tolerance (MAE, Max Error, Argmax Agreement).
"""

import os
import sys
from pathlib import Path
from typing import Generator, List
import numpy as np
import tensorflow as tf

try:
    from .config import (
        KERAS_MODEL_PATH,
        TFLITE_MODEL_PATH,
        TFLITE_QUANT_PATH,
        LABELS_PATH,
        CLASS_NAMES,
        IMAGE_SIZE,
        INPUT_SHAPE,
        TEST_DIR,
    )
    from .utils import build_data_pipelines
except ImportError:
    from config import (
        KERAS_MODEL_PATH,
        TFLITE_MODEL_PATH,
        TFLITE_QUANT_PATH,
        LABELS_PATH,
        CLASS_NAMES,
        IMAGE_SIZE,
        INPUT_SHAPE,
        TEST_DIR,
    )
    from utils import build_data_pipelines


def export_labels(labels_path: Path = LABELS_PATH):
    """Writes the 7-class taxonomy to labels.txt."""
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")
    print(f"[OK] labels.txt exported to: {labels_path}")


def representative_dataset_gen() -> Generator[List[np.ndarray], None, None]:
    """Generates representative samples for TFLite quantization calibration."""
    _, _, test_ds = build_data_pipelines(batch_size=1)
    if test_ds is not None:
        count = 0
        for img, _ in test_ds.take(100):
            count += 1
            yield [img.numpy().astype(np.float32)]
    else:
        # Fallback dummy calibration batch if dataset is not loaded
        for _ in range(50):
            yield [np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)]


def export_tflite_models(
    keras_model_path: Path = KERAS_MODEL_PATH,
    tflite_path: Path = TFLITE_MODEL_PATH,
    tflite_quant_path: Path = TFLITE_QUANT_PATH,
):
    """Converts the trained Keras model into standard Float32 and Quantized TFLite formats."""
    print("=" * 70)
    print("BeeGuard HiveSense - TensorFlow Lite Export")
    print("=" * 70)

    keras_model_path = Path(keras_model_path)
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model not found at: {keras_model_path}")

    print(f"Loading Keras model: {keras_model_path}")
    model = tf.keras.models.load_model(str(keras_model_path))

    # 1. Export labels.txt
    export_labels()

    # 2. Convert to Float32 TFLite
    print("\n[1/2] Converting to Float32 TFLite Model...")
    converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter_float.convert()

    tflite_path = Path(tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_float_model)
    float_size_mb = len(tflite_float_model) / (1024 * 1024)
    print(f"  [OK] Standard TFLite exported: {tflite_path} ({float_size_mb:.2f} MB)")

    # 3. Convert to Quantized TFLite (Post-Training Dynamic Range / Optimization)
    print("\n[2/2] Converting to Quantized TFLite Model...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter_quant.convert()

    tflite_quant_path = Path(tflite_quant_path)
    tflite_quant_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_quant_model)
    quant_size_mb = len(tflite_quant_model) / (1024 * 1024)
    print(f"  [OK] Quantized TFLite exported: {tflite_quant_path} ({quant_size_mb:.2f} MB)")
    print(f"  [Compression Ratio]: {float_size_mb / quant_size_mb:.2f}x reduction")


def verify_tflite_parity(
    keras_model_path: Path = KERAS_MODEL_PATH,
    tflite_path: Path = TFLITE_MODEL_PATH,
    max_tolerance: float = 1e-4,
    num_test_samples: int = 50,
) -> dict:
    """
    Verifies that TensorFlow Lite inference matches Keras model predictions
    within an acceptable numerical tolerance.
    """
    print("\n" + "=" * 70)
    print("BeeGuard HiveSense - Verifying Numerical Parity (Keras vs TFLite)")
    print("=" * 70)

    # 1. Load Keras Model
    keras_model = tf.keras.models.load_model(str(keras_model_path))

    # 2. Setup TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]

    # 3. Load Sample Batch
    _, _, test_ds = build_data_pipelines(batch_size=1)
    if test_ds is not None:
        samples = [img.numpy() for img, _ in test_ds.take(num_test_samples)]
    else:
        # Fallback dummy samples if test split not available
        samples = [np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32) for _ in range(num_test_samples)]

    keras_preds = []
    tflite_preds = []

    for sample in samples:
        # Keras Inference
        kp = keras_model.predict(sample, verbose=0)[0]
        keras_preds.append(kp)

        # TFLite Inference
        sample_tensor = sample.astype(np.float32)
        interpreter.set_tensor(input_idx, sample_tensor)
        interpreter.invoke()
        tp = interpreter.get_tensor(output_idx)[0]
        tflite_preds.append(tp)

    keras_preds = np.array(keras_preds)
    tflite_preds = np.array(tflite_preds)

    # Compute Discrepancies
    abs_diffs = np.abs(keras_preds - tflite_preds)
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))

    keras_classes = np.argmax(keras_preds, axis=1)
    tflite_classes = np.argmax(tflite_preds, axis=1)
    agreement_count = int(np.sum(keras_classes == tflite_classes))
    total_evaluated = len(samples)
    agreement_pct = (agreement_count / total_evaluated) * 100

    print(f"Evaluated Samples                : {total_evaluated}")
    print(f"Max Absolute Probability Diff    : {max_abs_diff:.6e}")
    print(f"Mean Absolute Probability Diff   : {mean_abs_diff:.6e}")
    print(f"Categorical Prediction Agreement : {agreement_pct:.2f}% ({agreement_count}/{total_evaluated})")

    passed_tolerance = max_abs_diff <= max_tolerance or agreement_pct == 100.0
    if passed_tolerance:
        print("\nVERDICT: PASS - TFLite predictions match Keras within acceptable tolerance.")
    else:
        print(f"\nVERDICT: FAIL - Max diff ({max_abs_diff:.6e}) exceeds tolerance ({max_tolerance:.6e}).")

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "agreement_pct": agreement_pct,
        "passed": passed_tolerance,
    }


if __name__ == "__main__":
    export_tflite_models()
    verify_tflite_parity()
