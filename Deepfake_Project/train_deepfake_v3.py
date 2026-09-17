"""Train the V3 image model with hard-example mining.

Run from the project directory:
    .\venv\Scripts\python.exe -u train_deepfake_v3.py
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET = Path("dataset")
MODEL_PATH = Path("models/deepfake_v3.keras")
META_PATH = Path("models/deepfake_v3_meta.json")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def files_and_labels():
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths, labels = [], []
    for label, name in enumerate(("fake", "real")):
        found = sorted(str(p) for p in (DATASET / name).iterdir()
                       if p.is_file() and p.suffix.lower() in extensions)
        if not found:
            raise FileNotFoundError(f"No images found in {DATASET / name}")
        paths.extend(found)
        labels.extend([label] * len(found))
    return np.asarray(paths), np.asarray(labels, dtype=np.float32)


def make_dataset(paths, labels, training=False, weights=None):
    if weights is None:
        weights = np.ones(len(paths), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    if training:
        ds = ds.shuffle(min(len(paths), 5000), seed=SEED,
                        reshuffle_each_iteration=True)

    def decode(path, label, weight):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                   expand_animations=False)
        image.set_shape([None, None, 3])
        # Preserve facial geometry instead of stretching every image.
        image = tf.image.resize_with_pad(image, *IMAGE_SIZE, antialias=True)
        return tf.cast(image, tf.float32), label, weight

    return (ds.map(decode, num_parallel_calls=2)
              .batch(BATCH_SIZE)
              .prefetch(1))


def build_model():
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.035),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.12),
        tf.keras.layers.RandomBrightness(0.06, value_range=(0.0, 255.0)),
    ], name="augmentation")
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3), include_preprocessing=True)
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augment(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid",
                                    name="real_probability")(x)
    return tf.keras.Model(inputs, outputs), backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, label_smoothing=0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")])


def choose_threshold(scores, labels):
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.linspace(0.2, 0.8, 121):
        predicted = scores >= threshold
        fake_recall = float(np.mean(~predicted[labels == 0]))
        real_recall = float(np.mean(predicted[labels == 1]))
        balanced = (fake_recall + real_recall) / 2
        if balanced > best[0]:
            best = balanced, float(threshold), fake_recall, real_recall
    return best


def main():
    paths, labels = files_and_labels()
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1765, random_state=SEED,
        stratify=train_labels)

    train = make_dataset(train_paths, train_labels, training=True)
    validation = make_dataset(val_paths, val_labels)
    test = make_dataset(test_paths, test_labels)
    model, backbone = build_model()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=0.3, min_lr=1e-7),
    ]

    compile_model(model, 8e-4)
    model.fit(train, validation_data=validation, epochs=6,
              callbacks=callbacks)

    backbone.trainable = True
    for layer in backbone.layers[:-35]:
        layer.trainable = False
    compile_model(model, 8e-6)
    model.fit(train, validation_data=validation, initial_epoch=6, epochs=14,
              callbacks=callbacks)

    # Give four times the normal weight to examples the first pass gets wrong.
    model = tf.keras.models.load_model(MODEL_PATH)
    ordered_train = make_dataset(train_paths, train_labels)
    train_scores = model.predict(ordered_train, verbose=1).ravel()
    hard = (train_scores >= 0.5) != train_labels.astype(bool)
    hard_weights = np.where(hard, 4.0, 1.0).astype(np.float32)
    print(f"Hard examples: {int(hard.sum())}/{len(hard)}")
    hard_train = make_dataset(train_paths, train_labels, training=True,
                              weights=hard_weights)
    compile_model(model, 3e-6)
    model.fit(hard_train, validation_data=validation, epochs=3,
              callbacks=callbacks)

    model = tf.keras.models.load_model(MODEL_PATH)
    val_scores = model.predict(validation, verbose=0).ravel()
    balanced, threshold, _, _ = choose_threshold(
        val_scores, val_labels.astype(int))
    test_scores = model.predict(test, verbose=0).ravel()
    test_predicted = test_scores >= threshold
    fake_recall = float(np.mean(~test_predicted[test_labels == 0]))
    real_recall = float(np.mean(test_predicted[test_labels == 1]))
    test_balanced = (fake_recall + real_recall) / 2
    metadata = {
        "version": "v3", "class_names": ["fake", "real"],
        "positive_class": "real", "image_size": list(IMAGE_SIZE),
        "decision_threshold": threshold, "uncertain_margin": 0.08,
        "minimum_confidence": 0.65,
        "validation_balanced_accuracy": balanced,
        "test_balanced_accuracy": test_balanced,
        "test_fake_recall": fake_recall,
        "test_real_recall": real_recall,
        "hard_examples": int(hard.sum()),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
