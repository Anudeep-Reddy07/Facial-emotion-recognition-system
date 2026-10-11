
"""
Facial Emotion Recognition using CNN
Dataset: FER2013 image-folder format
Input: 48x48 grayscale images
Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 60
NUM_CLASSES = 7
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.20

DATA_DIR = "data/fer2013"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = "models/emotion_cnn_best.keras"
HISTORY_PATH = "results/training_history.npy"

# Explicit order ensures labels match the model's output neurons.
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

tf.random.set_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# 2. LOAD DATA FROM IMAGE FOLDERS
# --------------------------------------------------

def load_fer2013():
    """Load train, validation and test images from folders."""

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}"
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Testing folder not found: {TEST_DIR}"
        )

    print("\n[INFO] Loading images from:", DATA_DIR)

    # Augmentation is applied only to training images.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    # Validation and test images are only normalized.
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        classes=CLASS_NAMES,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    validation_generator = validation_datagen.flow_from_directory(
        TRAIN_DIR,
        classes=CLASS_NAMES,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        classes=CLASS_NAMES,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False,
    )

    print("\n[INFO] Class mapping:")
    print(train_generator.class_indices)

    print("\n[INFO] Dataset summary:")
    print(f"Training images   : {train_generator.samples}")
    print(f"Validation images : {validation_generator.samples}")
    print(f"Testing images    : {test_generator.samples}")

    return train_generator, validation_generator, test_generator


# --------------------------------------------------
# 3. CNN MODEL
# --------------------------------------------------

def build_model():
    """Build and compile the CNN emotion classifier."""

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Convolutional block 1
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional block 2
        layers.Conv2D(
            128, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            128, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional block 3
        layers.Conv2D(
            256, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            256, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional block 4
        layers.Conv2D(
            512, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            512, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classification head
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        layers.Dropout(0.3),

        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# 4. TRAINING CALLBACKS
# --------------------------------------------------

def get_callbacks():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    tensorboard = TensorBoard(
        log_dir="results/logs",
        histogram_freq=0,
    )

    return [
        early_stopping,
        checkpoint,
        reduce_lr,
        tensorboard,
    ]


# --------------------------------------------------
# 5. PLOT TRAINING ACCURACY AND LOSS
# --------------------------------------------------

def plot_history(history):
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history.history["loss"],
        label="Training Loss",
    )
    axes[1].plot(
        history.history["val_loss"],
        label="Validation Loss",
    )
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=150)
    plt.close(fig)

    print("[INFO] Saved results/training_history.png")


# --------------------------------------------------
# 6. EVALUATE ON THE TEST SET
# --------------------------------------------------

def evaluate_model(model, test_generator):
    print("\n[INFO] Evaluating on the test dataset...")

    test_generator.reset()
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        verbose=1,
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    # Reset before predicting so all test samples are included
    # in the same order as test_generator.classes.
    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_generator.classes

    print("\n[INFO] Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTION_LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel("True Emotion")
    plt.xlabel("Predicted Emotion")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("[INFO] Saved results/confusion_matrix.png")

    return y_true, y_pred


# --------------------------------------------------
# 7. DISPLAY SAMPLE PREDICTIONS
# --------------------------------------------------

def visualise_predictions(model, test_generator, n=16):
    test_generator.reset()

    # Read the first test batch.
    images, labels = next(test_generator)

    count = min(n, len(images))
    predictions = model.predict(images[:count], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    columns = 4
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
        squeeze=False,
    )

    for i, ax in enumerate(axes.flatten()):
        if i >= count:
            ax.axis("off")
            continue

        true_label = int(labels[i])
        predicted_label = int(predicted_labels[i])

        ax.imshow(images[i].squeeze(), cmap="gray")

        title_color = (
            "green" if true_label == predicted_label else "red"
        )

        ax.set_title(
            f"True: {EMOTION_LABELS[true_label]}\n"
            f"Predicted: {EMOTION_LABELS[predicted_label]}",
            color=title_color,
            fontsize=9,
        )
        ax.axis("off")

    plt.suptitle(
        "Sample Predictions (Green = Correct, Red = Incorrect)"
    )
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png", dpi=150)
    plt.close(fig)

    print("[INFO] Saved results/sample_predictions.png")


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 60)
    print("FACIAL EMOTION RECOGNITION - CNN TRAINING")
    print("=" * 60)

    train_generator, validation_generator, test_generator = (
        load_fer2013()
    )

    model = build_model()
    model.summary()

    print("\n[INFO] Starting CNN training...")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Save the history for later analysis.
    np.save(HISTORY_PATH, history.history, allow_pickle=True)
    print(f"[INFO] Saved {HISTORY_PATH}")

    plot_history(history)

    # Load the best checkpoint saved during training.
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

    y_true, y_pred = evaluate_model(model, test_generator)
    visualise_predictions(model, test_generator)

    print("\n" + "=" * 60)
    print("[DONE] Training and evaluation completed.")
    print(f"Best model: {MODEL_PATH}")
    print("Graphs and reports: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
