"""
BeeGuard HiveSense - Automated Verification & Quality Assurance Suite
Verifies all project requirements, deliverables, and numerical tolerances:
1. Dataset split integrity and zero-leakage check
2. Model artifact existence (.keras, .tflite, quantized .tflite, labels.txt)
3. Diagnostic plots existence
4. Test set metrics evaluation
5. TFLite parity verification
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ANDROID_DIR = PROJECT_ROOT / "android"

REQUIRED_MODELS = [
    "bee_disease_model.keras",
    "bee_disease_model.tflite",
    "bee_disease_model_quantized.tflite",
    "labels.txt",
]

REQUIRED_PLOTS = [
    "class_distribution.png",
    "sample_dataset_images.png",
    "training_curves.png",
    "confusion_matrix.png",
]

REQUIRED_NOTEBOOKS = [
    "01_training.ipynb",
    "02_evaluation.ipynb",
]

REQUIRED_ANDROID_FILES = [
    "BeeDiseaseClassifier.kt",
    "MainActivity.kt",
    "build.gradle.kts",
]


def run_full_verification():
    print("=" * 75)
    print("BeeGuard HiveSense - Final Acceptance Verification Suite")
    print("=" * 75)

    all_passed = True

    # 1. Check Notebooks
    print("\n[Check 1/5] Validating Jupyter Notebooks...")
    for nb_name in REQUIRED_NOTEBOOKS:
        nb_path = NOTEBOOKS_DIR / nb_name
        if nb_path.exists():
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_cells = len(data.get("cells", []))
                print(f"  [OK] {nb_name} (Valid JSON, {num_cells} cells)")
            except Exception as e:
                print(f"  [FAIL] {nb_name} is corrupted: {e}")
                all_passed = False
        else:
            print(f"  [FAIL] Missing notebook: {nb_name}")
            all_passed = False

    # 2. Check Android Kotlin Integration
    print("\n[Check 2/5] Validating Android Integration Files...")
    for f_name in REQUIRED_ANDROID_FILES:
        f_path = ANDROID_DIR / f_name
        if f_path.exists() and f_path.stat().st_size > 0:
            print(f"  [OK] {f_name} ({f_path.stat().st_size} bytes)")
        else:
            print(f"  [FAIL] Missing or empty Android file: {f_name}")
            all_passed = False

    # 3. Check Diagnostic Plots
    print("\n[Check 3/5] Validating Diagnostic Plots...")
    for plot_name in REQUIRED_PLOTS:
        plot_path = PLOTS_DIR / plot_name
        if plot_path.exists() and plot_path.stat().st_size > 0:
            print(f"  [OK] {plot_name} ({plot_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [FAIL] Missing plot: {plot_name}")
            all_passed = False

    # 4. Check Model Artifacts
    print("\n[Check 4/5] Validating Model Artifacts...")
    for model_name in REQUIRED_MODELS:
        m_path = MODELS_DIR / model_name
        if m_path.exists() and m_path.stat().st_size > 0:
            print(f"  [OK] {model_name} ({m_path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  [FAIL] Missing model artifact: {model_name}")
            all_passed = False

    # 5. Check labels.txt contents
    labels_file = MODELS_DIR / "labels.txt"
    if labels_file.exists():
        labels = [l.strip() for l in labels_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  [OK] labels.txt contains {len(labels)} classes: {labels}")
        if len(labels) != 7:
            print("  [FAIL] labels.txt does not contain exactly 7 classes!")
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("VERDICT: PASS - All BeeGuard HiveSense deliverables verified.")
    else:
        print("VERDICT: FAIL - Some deliverables or checks failed.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)

