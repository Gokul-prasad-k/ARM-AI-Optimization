import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import get_dataset as kws_data


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parent

NETWORK1_MODEL = ROOT / "split_model" / "network1"
DATA_DIR = ROOT.parents[1] / "mlcommons_data"

OUTPUT_MODEL = ROOT / "ee1_probe"

# ============================================================
# SETTINGS
# ============================================================

NUM_CLASSES = 12
BATCH_SIZE = 100
EPOCHS = 30


# ============================================================
# DATASET
# ============================================================

os.environ["HOME"] = str(ROOT.parents[1])
os.environ["PWD"] = str(ROOT.parents[1])

class Args:
    pass


args = Args()
args.data_dir = DATA_DIR
args.bg_path = r"D:\1vidamuyarchi"
args.background_volume = 0.1
args.background_frequency = 0.8
args.silence_percentage = 10.0
args.unknown_percentage = 10.0
args.time_shift_ms = 100.0

args.sample_rate = 16000
args.clip_duration_ms = 1000
args.window_size_ms = 30.0
args.window_stride_ms = 20.0
args.feature_type = "mfcc"
args.dct_coefficient_count = 10

args.num_train_samples = -1
args.num_val_samples = -1
args.num_test_samples = 4890
args.batch_size = BATCH_SIZE


print("=" * 60)
print("LOADING DATA")
print("=" * 60)

ds_train, ds_test, ds_val = kws_data.get_training_data(args)

print("Dataset loaded.")


# ============================================================
# LOAD NETWORK 1
# ============================================================

print()
print("=" * 60)
print("LOADING NETWORK 1")
print("=" * 60)

network1 = keras.models.load_model(NETWORK1_MODEL)

print("Network 1 output:", network1.output_shape)


# ============================================================
# FREEZE NETWORK 1
# ============================================================

network1.trainable = False

for layer in network1.layers:
    layer.trainable = False


# ============================================================
# BUILD EE1
# ============================================================

print()
print("=" * 60)
print("BUILDING EE1 PROBE")
print("=" * 60)

ee_input = keras.Input(
    shape=network1.output_shape[1:],
    name="ee1_input"
)

x = keras.layers.GlobalAveragePooling2D()(ee_input)

ee_output = keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="ee1"
)(x)

ee1 = keras.Model(
    inputs=ee_input,
    outputs=ee_output,
    name="ee1_probe"
)

ee1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

ee1.summary()


# ============================================================
# CREATE FEATURE DATASETS
# ============================================================

print()
print("=" * 60)
print("GENERATING FROZEN FEATURES")
print("=" * 60)


def make_feature_dataset(dataset):

    def generator():

        for samples, labels in dataset:

            features = network1(
                samples,
                training=False
            ).numpy()

            for feature, label in zip(features, labels.numpy()):
                yield feature, label


    output_signature = (
        tf.TensorSpec(
            shape=(25, 5, 64),
            dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int64
        )
    )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_features = make_feature_dataset(ds_train)
val_features = make_feature_dataset(ds_val)
test_features = make_feature_dataset(ds_test)


# ============================================================
# TRAIN EE1
# ============================================================

print()
print("=" * 60)
print("TRAINING EE1")
print("=" * 60)

ee1.fit(
    train_features,
    epochs=EPOCHS
)


# ============================================================
# TEST EE1
# ============================================================

print()
print("=" * 60)
print("EE1 TEST ACCURACY")
print("=" * 60)

results = ee1.evaluate(
    test_features,
    return_dict=True
)

print()
print("EE1 test accuracy:",
      results["sparse_categorical_accuracy"] * 100,
      "%")


# ============================================================
# SAVE
# ============================================================

ee1.save(OUTPUT_MODEL)

print()
print("Saved EE1 probe to:")
print(OUTPUT_MODEL)