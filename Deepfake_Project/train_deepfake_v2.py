"""Clean V2 deepfake trainer with stratified splits and calibrated decisions."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET = Path("dataset")
MODEL_PATH = Path("models/deepfake_v2.keras")
META_PATH = Path("models/deepfake_v2_meta.json")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 42

tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def files_and_labels():
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths, labels = [], []
    for label, name in enumerate(("fake", "real")):  # fake=0, real=1
        found = [str(p) for p in (DATASET / name).iterdir()
                 if p.is_file() and p.suffix.lower() in extensions]
        if not found:
            raise FileNotFoundError(f"No images found in {DATASET / name}")
        paths.extend(found)
        labels.extend([label] * len(found))
    return np.array(paths), np.array(labels, dtype=np.float32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 2_000), seed=SEED, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    return ds.map(decode, num_parallel_calls=2).batch(BATCH_SIZE).prefetch(1)


def best_threshold(real_scores, labels):
    # Maximise balanced accuracy on a held-out validation set.
    candidates = np.linspace(0.20, 0.80, 121)
    best, best_score = 0.5, -1.0
    for threshold in candidates:
        predicted = (real_scores >= threshold).astype(int)
        fake_recall = np.mean(predicted[labels == 0] == 0)
        real_recall = np.mean(predicted[labels == 1] == 1)
        score = (fake_recall + real_recall) / 2
        if score > best_score:
            best, best_score = float(threshold), float(score)
    return best, best_score


def main():
    paths, labels = files_and_labels()
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1765, random_state=SEED, stratify=train_labels)
    train = make_dataset(train_paths, train_labels, training=True)
    validation = make_dataset(val_paths, val_labels)
    test = make_dataset(test_paths, test_labels)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.08),
    ], name="augmentation")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augment(inputs)
    x = backbone(x, training=False)  # EfficientNet includes its own rescaling.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="real_probability")(x)
    model = tf.keras.Model(inputs, output)

    def compile_model(rate):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(rate),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-7),
    ]
    compile_model(1e-3)
    model.fit(train, validation_data=validation, epochs=5, callbacks=callbacks)
    backbone.trainable = True
    for layer in backbone.layers[:-25]:
        layer.trainable = False
    compile_model(1e-5)
    model.fit(train, validation_data=validation, initial_epoch=5, epochs=13, callbacks=callbacks)

    model = tf.keras.models.load_model(MODEL_PATH)
    val_scores = model.predict(validation, verbose=0).ravel()
    threshold, validation_balanced_accuracy = best_threshold(val_scores, val_labels.astype(int))
    test_scores = model.predict(test, verbose=0).ravel()
    test_predicted = (test_scores >= threshold).astype(int)
    test_fake_recall = float(np.mean(test_predicted[test_labels == 0] == 0))
    test_real_recall = float(np.mean(test_predicted[test_labels == 1] == 1))
    metadata = {
        "version": "v2", "class_names": ["fake", "real"], "image_size": IMAGE_SIZE,
        "decision_threshold": threshold, "uncertain_margin": 0.07,
        "minimum_confidence": 0.65,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "test_balanced_accuracy": (test_fake_recall + test_real_recall) / 2,
        "test_fake_recall": test_fake_recall, "test_real_recall": test_real_recall,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
