import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import kws_util
import get_dataset as kws_data


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parent

ORIGINAL_MODEL = ROOT / "trained_models" / "kws_ref_model"

NETWORK1_MODEL = ROOT / "split_model" / "network1"
NETWORK2_MODEL = ROOT / "split_model" / "network2"

DATA_DIR = ROOT / "mlcommons_data"


# ============================================================
# SETTINGS
# ============================================================

NUM_TEST_SAMPLES = 4890


# ============================================================
# LOAD DATASET USING MLCOMMONS PIPELINE
# ============================================================

print("=" * 60)
print("LOADING MLCOMMONS TEST DATA")
print("=" * 60)

# kws_util expects HOME/PWD to exist on Windows
os.environ["HOME"] = r"D:\1vidamuyarchi"
os.environ["PWD"] = r"D:\1vidamuyarchi"

# Build the same arguments expected by get_training_data()
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
args.num_test_samples = NUM_TEST_SAMPLES

args.batch_size = 100

print("Getting dataset...")

ds_train, ds_test, ds_val = kws_data.get_training_data(args)

print("Dataset loaded.")


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

print("Original model:")
original = keras.models.load_model(ORIGINAL_MODEL)

print("Network 1:")
network1 = keras.models.load_model(NETWORK1_MODEL)

print("Network 2:")
network2 = keras.models.load_model(NETWORK2_MODEL)

print()
print("Models loaded successfully.")


# ============================================================
# RUN VALIDATION
# ============================================================

original_outputs = []
split_outputs = []
labels = []

sample_count = 0

print()
print("=" * 60)
print("RUNNING ORIGINAL VS SPLIT")
print("=" * 60)

for samples, batch_labels in ds_test:

    # Respect requested sample count
    remaining = NUM_TEST_SAMPLES - sample_count

    if remaining <= 0:
        break

    if samples.shape[0] > remaining:
        samples = samples[:remaining]
        batch_labels = batch_labels[:remaining]

    # --------------------------------------------------------
    # Original model
    # --------------------------------------------------------

    original_pred = original(samples, training=False).numpy()

    # --------------------------------------------------------
    # Forced split:
    #
    # input
    #   ↓
    # Network 1
    #   ↓
    # intermediate tensor
    #   ↓
    # Network 2
    #   ↓
    # final output
    # --------------------------------------------------------

    intermediate = network1(samples, training=False)

    split_pred = network2(intermediate, training=False).numpy()

    original_outputs.append(original_pred)
    split_outputs.append(split_pred)
    labels.append(batch_labels.numpy())

    sample_count += samples.shape[0]

    if sample_count % 500 < samples.shape[0]:
        print(f"Processed: {sample_count}")


# ============================================================
# COMBINE RESULTS
# ============================================================

original_outputs = np.concatenate(original_outputs, axis=0)
split_outputs = np.concatenate(split_outputs, axis=0)
labels = np.concatenate(labels, axis=0)

# Make sure we only compare requested samples
original_outputs = original_outputs[:NUM_TEST_SAMPLES]
split_outputs = split_outputs[:NUM_TEST_SAMPLES]
labels = labels[:NUM_TEST_SAMPLES]


# ============================================================
# ACCURACY
# ============================================================

original_predictions = np.argmax(original_outputs, axis=1)
split_predictions = np.argmax(split_outputs, axis=1)

original_accuracy = np.mean(original_predictions == labels) * 100.0
split_accuracy = np.mean(split_predictions == labels) * 100.0

prediction_agreement = (
    np.mean(original_predictions == split_predictions) * 100.0
)


# ============================================================
# NUMERICAL DIFFERENCE
# ============================================================

absolute_difference = np.abs(
    original_outputs - split_outputs
)

max_difference = np.max(absolute_difference)
mean_difference = np.mean(absolute_difference)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("        FORCED SPLIT VALIDATION")
print("=" * 60)

print(f"Samples                         : {len(labels)}")

print()
print(f"Original accuracy               : {original_accuracy:.4f}%")
print(f"Network 1 -> Network 2 accuracy : {split_accuracy:.4f}%")

print()
print(f"Prediction agreement            : {prediction_agreement:.4f}%")

print()
print(f"Maximum probability difference  : {max_difference:.10f}")
print(f"Mean probability difference     : {mean_difference:.10f}")

print("=" * 60)


# ============================================================
# PASS / FAIL
# ============================================================

if prediction_agreement >= 99.99 and max_difference < 1e-4:
    print()
    print("PASS: Split preserves the original model behaviour.")
else:
    print()
    print("FAIL: Split does NOT exactly reproduce the original model.")