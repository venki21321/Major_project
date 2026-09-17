"""Train three complementary image models and save an inference ensemble.

Run from the project directory:
    .\venv\Scripts\python.exe train_ensemble.py

Training three ImageNet backbones is intentionally much slower than train.py.
Models are trained one at a time so a normal laptop does not hold all three in
memory simultaneously.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import IMAGE_SIZE, compile_model, make_datasets

MODEL_DIR = Path("models")
META_PATH = MODEL_DIR / "model_meta.json"
ARCHITECTURES = ("mobilenet_v2", "efficientnet_b0", "densenet121")


def build_model(name):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(
            factor=0.05, value_range=(0.0, 255.0)),
    ], name="web_image_augmentation")

    if name == "mobilenet_v2":
        constructor = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "efficientnet_b0":
        constructor = tf.keras.applications.EfficientNetB0
        # EfficientNet includes its own input rescaling.
        preprocess = lambda value: value
    elif name == "densenet121":
        constructor = tf.keras.applications.DenseNet121
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Unsupported architecture: {name}")

    base = constructor(
        include_top=False, weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="real_probability")(x)
    return tf.keras.Model(inputs, outputs), base


def labels_from(dataset):
    return np.concatenate([labels.numpy().ravel() for _, labels in dataset])


def select_balanced_threshold(scores, labels):
    """Choose the threshold with the best mean of fake and real recall."""
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = (scores >= threshold).astype(np.float32)
        fake_recall = float((prediction[labels == 0] == 0).mean())
        real_recall = float((prediction[labels == 1] == 1).mean())
        balanced_accuracy = (fake_recall + real_recall) / 2
        if balanced_accuracy > best[0]:
            best = (balanced_accuracy, float(threshold),
                    fake_recall, real_recall)
    return best


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, validation, class_names = make_datasets()
    labels = labels_from(validation)
    paths = []
    validation_scores = []

    for name in ARCHITECTURES:
        path = MODEL_DIR / f"deepfake_{name}.keras"
        if path.exists():
            print(f"\nResuming: using completed {path}")
            best = tf.keras.models.load_model(path)
        else:
            print(f"\nTraining {name}")
            last_error = None
            for attempt in range(1, 4):
                try:
                    model, base = build_model(name)
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 3:
                        raise
                    print(
                        f"Could not load/download {name} weights "
                        f"(attempt {attempt}/3): {error}"
                    )
                    print("Retrying in 10 seconds. Check the internet connection.")
                    time.sleep(10)
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    path, monitor="val_auc", mode="max",
                    save_best_only=True, verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=3,
                    restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.3, patience=2,
                    min_lr=1e-7, verbose=1),
            ]
            compile_model(model, 1e-3)
            model.fit(train, validation_data=validation, epochs=5,
                      callbacks=callbacks)

            base.trainable = True
            for layer in base.layers[:-25]:
                layer.trainable = False
            compile_model(model, 1e-5)
            model.fit(train, validation_data=validation, initial_epoch=5,
                      epochs=12, callbacks=callbacks)
            best = tf.keras.models.load_model(path)

        validation_scores.append(
            best.predict(validation, verbose=0).ravel())
        paths.append(path.as_posix())
        del best
        if "model" in locals():
            del model, base
        tf.keras.backend.clear_session()
        gc.collect()

    scores = np.stack(validation_scores, axis=1)
    ensemble_score = scores.mean(axis=1)
    balanced_accuracy, threshold, fake_recall, real_recall = (
        select_balanced_threshold(ensemble_score, labels))
    prediction = (ensemble_score >= threshold).astype(np.float32)
    accuracy = float((prediction == labels).mean())

    metadata = {
        "class_names": class_names,
        "positive_class": "real",
        "image_size": list(IMAGE_SIZE),
        "ensemble_models": paths,
        "ensemble_method": "mean_probability",
        "decision_threshold": threshold,
        "uncertain_margin": 0.12,
        "max_disagreement": 0.18,
        "validation_accuracy": accuracy,
        "validation_balanced_accuracy": balanced_accuracy,
        "validation_fake_recall": fake_recall,
        "validation_real_recall": real_recall,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSaved ensemble metadata:", META_PATH)
    print(f"Threshold={threshold:.3f}, accuracy={accuracy:.3f}, "
          f"balanced accuracy={balanced_accuracy:.3f}, "
          f"fake recall={fake_recall:.3f}, real recall={real_recall:.3f}")


if __name__ == "__main__":
    main()
