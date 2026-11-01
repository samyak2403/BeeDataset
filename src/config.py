"""
BeeGuard HiveSense - Honeybee Disease Detection CNN Training
Configuration & Hyperparameters
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "BeeGuardDataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 7-Class Honeybee Disease Taxonomy
CLASS_NAMES = [
    "Varroa Mites",
    "American Foulbrood",
    "European Foulbrood",
    "Chalkbrood",
    "Sacbrood",
    "Nosema",
    "Healthy",
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Model Architecture Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
DROPOUT_RATE = 0.2
FINE_TUNE_AT_LAYER = 100  # Layer index in MobileNetV2 from which fine-tuning starts

# Training Hyperparameters
BATCH_SIZE = 32
RANDOM_SEED = 42

# Phase 1: Transfer Learning (Feature Extraction with Frozen Base)
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

# Phase 2: Fine-Tuning (Selected Top Layers Unfrozen)
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-5

# Early Stopping & Checkpointing
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-7

# Model Artifact Export Paths
KERAS_MODEL_PATH = MODELS_DIR / "bee_disease_model.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "bee_disease_model.tflite"
TFLITE_QUANT_PATH = MODELS_DIR / "bee_disease_model_quantized.tflite"
LABELS_PATH = MODELS_DIR / "labels.txt"

# Evaluation Plot Paths
TRAINING_HISTORY_PLOT = PLOTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"
SAMPLE_IMAGES_PLOT = PLOTS_DIR / "sample_dataset_images.png"
CLASS_DISTRIBUTION_PLOT = PLOTS_DIR / "class_distribution.png"

