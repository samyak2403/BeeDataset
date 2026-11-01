"""
BeeGuard HiveSense - MobileNetV2 Transfer Learning Architecture
Implements:
- MobileNetV2 pretrained on ImageNet (224x224x3)
- Realistic image augmentation pipeline
- Transfer learning with initial frozen base
- GlobalAveragePooling2D -> Dropout -> Dense(7, softmax)
- Fine-tuning hooks for top convolutional blocks
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers

try:
    from .config import (
        INPUT_SHAPE,
        NUM_CLASSES,
        DROPOUT_RATE,
        FINE_TUNE_AT_LAYER,
        PHASE1_LR,
        PHASE2_LR,
    )
except ImportError:
    from config import (
        INPUT_SHAPE,
        NUM_CLASSES,
        DROPOUT_RATE,
        FINE_TUNE_AT_LAYER,
        PHASE1_LR,
        PHASE2_LR,
    )


def build_data_augmentation() -> tf.keras.Sequential:
    """
    Constructs a robust, realistic data augmentation pipeline suitable for
    apicultural macro imagery under variable field lighting and bee postures.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
            layers.RandomRotation(0.15, fill_mode="nearest", name="aug_random_rotation"),
            layers.RandomZoom(0.15, fill_mode="nearest", name="aug_random_zoom"),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="nearest",
                name="aug_random_translation",
            ),
            layers.RandomContrast(0.15, name="aug_random_contrast"),
        ],
        name="data_augmentation",
    )


def create_bee_disease_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
    learning_rate=PHASE1_LR,
) -> tf.keras.Model:
    """
    Builds the 7-class MobileNetV2 Transfer Learning model.
    Base model initialized with ImageNet weights and frozen for Phase 1.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Data augmentation block (active only during training)
    augmentation = build_data_augmentation()
    x = augmentation(inputs)

    # MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1]
    x = applications.mobilenet_v2.preprocess_input(x)

    # Pretrained MobileNetV2 Backbone
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # Freeze the backbone initially
    base_model.trainable = False

    # Run backbone in inference mode so batch norm stats remain frozen
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_dense_7",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BeeGuard_MobileNetV2")

    # Compile with Adam optimizer, Categorical Crossentropy, and tracking metrics
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def prepare_for_finetuning(
    model: tf.keras.Model,
    fine_tune_at: int = FINE_TUNE_AT_LAYER,
    learning_rate: float = PHASE2_LR,
) -> tf.keras.Model:
    """
    Unfreezes top layers of MobileNetV2 for Phase 2 fine-tuning
    with a reduced learning rate.
    """
    # Locate the MobileNetV2 base layer specifically (ignoring data augmentation)
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 base model could not be found in the provided network.")

    base_model.trainable = True

    # Freeze all layers up to fine_tune_at
    print(f"Fine-tuning enabled: Unfreezing MobileNetV2 layers from layer {fine_tune_at} onwards...")
    total_layers = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at:
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(len(l.trainable_variables) for l in base_model.layers if l.trainable)
    print(f"  MobileNetV2 layers: {total_layers} total | Trainable variables count: {trainable_count}")

    # Recompile with small learning rate to avoid destabilizing pre-trained features
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    m = create_bee_disease_model()
    m.summary()
