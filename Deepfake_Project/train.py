"""Train a robust real-vs-fake image classifier.

The model output is P(real), and that convention is saved in metadata so
inference cannot accidentally reverse the two classes.
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 42
# Smaller images and batches let this train on a normal laptop CPU/RAM.
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/deepfake_model.keras")
META_PATH = Path("models/model_meta.json")
tf.keras.utils.set_random_seed(SEED)
# Avoid TensorFlow creating large CPU-memory workspaces on lower-memory PCs.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def make_datasets():
    """Create a per-class (stratified) split.

    `validation_split` on a directory can select one contiguous block of
    alphabetically ordered paths, leaving validation with only one class.
    """
    class_names = ["fake", "real"]
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    rng = np.random.default_rng(SEED)
    for label, class_name in enumerate(class_names):
        files = [str(path) for path in (DATASET_DIR / class_name).iterdir()
                 if path.is_file() and path.suffix.lower() in extensions]
        if not files:
            raise ValueError(f"No images found in dataset/{class_name}")
        rng.shuffle(files)
        split_at = max(1, int(len(files) * 0.2))
        val_paths.extend(files[:split_at])
        val_labels.extend([label] * split_at)
        train_paths.extend(files[split_at:])
        train_labels.extend([label] * (len(files) - split_at))

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                   expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    def dataset(paths, labels, training):
        result = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            result = result.shuffle(min(len(paths), 2_000), seed=SEED,
                                    reshuffle_each_iteration=True)
        return result.map(decode, num_parallel_calls=2).batch(BATCH_SIZE).prefetch(1)

    return dataset(train_paths, train_labels, True), dataset(val_paths, val_labels, False), class_names


def select_fake_threshold(model, validation, real_index):
    """Limit false-fake results: retain 98% of validation real images as real."""
    scores = model.predict(validation, verbose=0).ravel()  # score = P(real)
    labels = np.concatenate([y.numpy().ravel() for _, y in validation])
    real_scores = scores[labels == real_index]
    return float(np.clip(np.quantile(real_scores, 0.02), 0.05, 0.5))


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="real_recall")],
    )


def main():
    train, validation, class_names = make_datasets()
    real_index = class_names.index("real")
    print("Class mapping:", dict(enumerate(class_names)))

    # Images are streamed from disk; caching all 22k images can exhaust RAM.

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    # Required by ImageNet MobileNetV2: convert uint8 pixels to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="real_probability")(x)
    model = tf.keras.Model(inputs, outputs)

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    callbacks = [
        # AUC evaluates both real and fake examples. Monitoring only real recall
        # can save a useless model that calls every image "real".
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc",
                                           mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=2, min_lr=1e-7, verbose=1),
    ]

    compile_model(model, 1e-3)
    model.fit(train, validation_data=validation, epochs=5, callbacks=callbacks)

    # Fine-tune a small tail of the backbone with a much lower learning rate.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    model.fit(train, validation_data=validation, initial_epoch=5, epochs=17,
              callbacks=callbacks)

    model = tf.keras.models.load_model(MODEL_PATH)
    threshold = select_fake_threshold(model, validation, real_index)
    META_PATH.write_text(json.dumps({"class_names": class_names,
                                     "positive_class": "real",
                                     "fake_threshold": threshold,
                                     "image_size": IMAGE_SIZE}, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH}; fake threshold={threshold:.3f}")


if __name__ == "__main__":
    main()
