import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

DATA_DIR = REPO_ROOT / "mlcommons_data"

ORIGINAL_MODEL = ROOT / "trained_models" / "kws_ref_model"
NETWORK1_MODEL = ROOT / "split_model" / "network1"
NETWORK2_MODEL = ROOT / "split_model" / "network2"
EE1_MODEL = ROOT / "ee1_probe"

RESULT_CSV = ROOT / "ee1_threshold_results.csv"

NUM_CLASSES = 12
TEST_SAMPLES = 4890

# Thresholds we will evaluate.
THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
    0.97,
    0.99,
]

CLASS_NAMES = [
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
    "silence",
    "unknown",
    "down",
]

# ============================================================
# ENVIRONMENT
# ============================================================

# The MLCommons scripts expect these variables on Windows.
os.environ["HOME"] = str(REPO_ROOT)
os.environ["PWD"] = str(ROOT)

# ============================================================
# IMPORT EXISTING DATASET CODE
# ============================================================

import kws_util
import get_dataset as kws_data


# ============================================================
# HELPERS
# ============================================================

def accuracy(predictions, labels):
    return float(np.mean(predictions == labels))


def probability_difference(a, b):
    diff = np.abs(a - b)
    return float(np.max(diff)), float(np.mean(diff))


def print_class_distribution(labels, name):
    print()
    print(f"--- {name} CLASS DISTRIBUTION ---")

    counts = np.bincount(
        labels.astype(np.int32),
        minlength=NUM_CLASSES
    )

    for i, count in enumerate(counts):
        if count > 0:
            class_name = (
                CLASS_NAMES[i]
                if i < len(CLASS_NAMES)
                else f"class_{i}"
            )

            print(
                f"{i:2d} {class_name:<10} : "
                f"{count:4d} ({100.0 * count / len(labels):6.2f}%)"
            )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("T-RECX FINAL EE1 VALIDATION")
print("=" * 70)

print()
print("TensorFlow:", tf.__version__)
print("Base directory:", BASE_DIR)
print("Dataset:", DATA_DIR)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print()
print("=" * 70)
print("1. LOADING MLCOMMONS DATASET")
print("=" * 70)

# Reuse the exact MLCommons argument parser used previously.
# This keeps preprocessing identical to the original 92.17% result.

import sys

sys.argv = [
    sys.argv[0],

    "--data_dir", DATA_DIR,

    "--num_test_samples", str(TEST_SAMPLES),

    "--feature_type", "mfcc",

    "--sample_rate", "16000",

    "--clip_duration_ms", "1000",

    "--window_size_ms", "30",

    "--window_stride_ms", "20",

    "--dct_coefficient_count", "10",

    "--model_architecture", "ds_cnn",
]

Flags, _ = kws_util.parse_command()

print("Getting dataset...")
ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
print("Dataset loaded.")

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

print()
print("=" * 70)
print("2. LOADING MODELS")
print("=" * 70)

print()
print("Original model:")
original_model = keras.models.load_model(ORIGINAL_MODEL)

print()
print("Network 1:")
network1 = keras.models.load_model(NETWORK1_MODEL)

print()
print("Network 2:")
network2 = keras.models.load_model(NETWORK2_MODEL)

print()
print("EE1:")
ee1 = keras.models.load_model(EE1_MODEL)

print()
print("All models loaded.")

# ------------------------------------------------------------
# MODEL SHAPES
# ------------------------------------------------------------

print()
print("=" * 70)
print("3. MODEL INTERFACES")
print("=" * 70)

print("Original input :", original_model.input_shape)
print("Original output:", original_model.output_shape)

print("Network1 input :", network1.input_shape)
print("Network1 output:", network1.output_shape)

print("Network2 input :", network2.input_shape)
print("Network2 output:", network2.output_shape)

print("EE1 input      :", ee1.input_shape)
print("EE1 output     :", ee1.output_shape)

# Basic interface sanity checks.

assert network1.input_shape[1:] == original_model.input_shape[1:], \
    "Network1 input does not match original model input."

assert network1.output_shape[1:] == network2.input_shape[1:], \
    "Network1 output does not match Network2 input."

assert network1.output_shape[1:] == ee1.input_shape[1:], \
    "Network1 output does not match EE1 input."

assert network2.output_shape[1:] == original_model.output_shape[1:], \
    "Network2 output does not match original output."

print()
print("INTERFACE CHECK: PASS")

# ------------------------------------------------------------
# RUN TEST DATA
# ------------------------------------------------------------

print()
print("=" * 70)
print("4. RUNNING TEST SET")
print("=" * 70)

all_labels = []

original_outputs = []
network1_outputs = []
ee1_outputs = []
network2_outputs = []

total_batches = 0

for samples, labels in ds_test:

    samples_np = samples.numpy()

    # --------------------------------------------------------
    # Original model
    # --------------------------------------------------------

    original_pred = original_model.predict(
        samples_np,
        verbose=0
    )

    # --------------------------------------------------------
    # Network 1
    # --------------------------------------------------------

    intermediate = network1.predict(
        samples_np,
        verbose=0
    )

    # --------------------------------------------------------
    # EE1
    # --------------------------------------------------------

    ee_pred = ee1.predict(
        intermediate,
        verbose=0
    )

    # --------------------------------------------------------
    # Network 2
    #
    # IMPORTANT:
    # We run Network2 for ALL samples here.
    #
    # This lets us calculate every threshold without
    # repeatedly executing the model.
    # --------------------------------------------------------

    final_pred = network2.predict(
        intermediate,
        verbose=0
    )

    all_labels.append(labels.numpy())

    original_outputs.append(original_pred)
    network1_outputs.append(intermediate)
    ee1_outputs.append(ee_pred)
    network2_outputs.append(final_pred)

    total_batches += 1

    if total_batches % 10 == 0:
        print(
            f"Processed batches: {total_batches}"
        )

# ------------------------------------------------------------
# CONCATENATE
# ------------------------------------------------------------

labels = np.concatenate(all_labels, axis=0)

original_outputs = np.concatenate(
    original_outputs,
    axis=0
)

network1_outputs = np.concatenate(
    network1_outputs,
    axis=0
)

ee1_outputs = np.concatenate(
    ee1_outputs,
    axis=0
)

network2_outputs = np.concatenate(
    network2_outputs,
    axis=0
)

print()
print("Total samples processed:", len(labels))

assert len(labels) == TEST_SAMPLES, (
    f"Expected {TEST_SAMPLES} samples, "
    f"got {len(labels)}"
)

# ------------------------------------------------------------
# BASIC PREDICTIONS
# ------------------------------------------------------------

original_classes = np.argmax(
    original_outputs,
    axis=1
)

ee_classes = np.argmax(
    ee1_outputs,
    axis=1
)

network2_classes = np.argmax(
    network2_outputs,
    axis=1
)

# ------------------------------------------------------------
# BASELINE ACCURACY
# ------------------------------------------------------------

print()
print("=" * 70)
print("5. BASELINE VALIDATION")
print("=" * 70)

original_accuracy = accuracy(
    original_classes,
    labels
)

network2_accuracy = accuracy(
    network2_classes,
    labels
)

print(
    f"Original model accuracy : "
    f"{original_accuracy * 100:.4f}%"
)

print(
    f"Network1 -> Network2     : "
    f"{network2_accuracy * 100:.4f}%"
)

# ------------------------------------------------------------
# FORCED SPLIT AGREEMENT
# ------------------------------------------------------------

original_vs_split = np.mean(
    original_classes == network2_classes
)

max_diff, mean_diff = probability_difference(
    original_outputs,
    network2_outputs
)

print()
print("Original vs forced split")
print(
    f"Prediction agreement     : "
    f"{original_vs_split * 100:.4f}%"
)

print(
    f"Maximum probability diff : "
    f"{max_diff:.12f}"
)

print(
    f"Mean probability diff    : "
    f"{mean_diff:.12f}"
)

if original_vs_split < 0.999999:
    print()
    print("WARNING:")
    print("Forced split does NOT perfectly reproduce")
    print("the original model predictions.")

else:
    print()
    print("FORCED SPLIT CHECK: PASS")

# ------------------------------------------------------------
# DATA DISTRIBUTION
# ------------------------------------------------------------

print_class_distribution(
    labels,
    "TEST"
)

# ------------------------------------------------------------
# EE1 STANDALONE ACCURACY
# ------------------------------------------------------------

ee_accuracy = accuracy(
    ee_classes,
    labels
)

print()
print("=" * 70)
print("6. EE1 STANDALONE RESULT")
print("=" * 70)

print(
    f"EE1 accuracy : "
    f"{ee_accuracy * 100:.4f}%"
)

# ------------------------------------------------------------
# EE CONFIDENCE
# ------------------------------------------------------------

ee_confidence = np.max(
    ee1_outputs,
    axis=1
)

print()
print("=" * 70)
print("7. EE1 CONFIDENCE DISTRIBUTION")
print("=" * 70)

for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:

    value = np.percentile(
        ee_confidence,
        p
    )

    print(
        f"{p:3d}th percentile : "
        f"{value:.6f}"
    )

# ------------------------------------------------------------
# THRESHOLD ANALYSIS
# ------------------------------------------------------------

print()
print("=" * 70)
print("8. EARLY-EXIT THRESHOLD ANALYSIS")
print("=" * 70)

print()
print(
    "Threshold | Exit % | "
    "Exited Acc | Fallback Acc | "
    "Cascade Acc | Agreement"
)

print("-" * 78)

results = []

for threshold in THRESHOLDS:

    # Samples that leave the network early.
    exit_mask = ee_confidence >= threshold

    # Samples sent to Network2.
    fallback_mask = ~exit_mask

    exit_count = int(np.sum(exit_mask))
    fallback_count = int(np.sum(fallback_mask))

    exit_percentage = (
        100.0 * exit_count / len(labels)
    )

    # --------------------------------------------------------
    # Accuracy of samples that actually exit at EE1.
    # --------------------------------------------------------

    if exit_count > 0:

        exited_accuracy = accuracy(
            ee_classes[exit_mask],
            labels[exit_mask]
        )

    else:

        exited_accuracy = np.nan

    # --------------------------------------------------------
    # Accuracy of samples sent to Network2.
    # --------------------------------------------------------

    if fallback_count > 0:

        fallback_accuracy = accuracy(
            network2_classes[fallback_mask],
            labels[fallback_mask]
        )

    else:

        fallback_accuracy = np.nan

    # --------------------------------------------------------
    # Build actual cascade output.
    # --------------------------------------------------------

    cascade_classes = network2_classes.copy()

    cascade_classes[exit_mask] = (
        ee_classes[exit_mask]
    )

    cascade_accuracy = accuracy(
        cascade_classes,
        labels
    )

    # --------------------------------------------------------
    # Agreement with original model.
    # --------------------------------------------------------

    cascade_agreement = np.mean(
        cascade_classes == original_classes
    )

    print(
        f"{threshold:9.2f} | "
        f"{exit_percentage:6.2f}% | "
        f"{exited_accuracy * 100 if not np.isnan(exited_accuracy) else float('nan'):10.2f}% | "
        f"{fallback_accuracy * 100 if not np.isnan(fallback_accuracy) else float('nan'):12.2f}% | "
        f"{cascade_accuracy * 100:11.2f}% | "
        f"{cascade_agreement * 100:9.2f}%"
    )

    # --------------------------------------------------------
    # Store result.
    # --------------------------------------------------------

    results.append({
        "threshold": threshold,
        "exit_count": exit_count,
        "fallback_count": fallback_count,
        "exit_percentage": exit_percentage,
        "exited_accuracy": (
            exited_accuracy * 100
            if not np.isnan(exited_accuracy)
            else ""
        ),
        "fallback_accuracy": (
            fallback_accuracy * 100
            if not np.isnan(fallback_accuracy)
            else ""
        ),
        "cascade_accuracy": cascade_accuracy * 100,
        "cascade_agreement": cascade_agreement * 100,
    })

# ------------------------------------------------------------
# ESTIMATED COMPUTE SAVING
# ------------------------------------------------------------

print()
print("=" * 70)
print("9. ESTIMATED COMPUTATION")
print("=" * 70)

print()
print(
    "This is an architectural estimate, NOT STM32 timing."
)

print(
    "Assuming the current split has 5 equal-cost "
    "DS-CNN blocks:"
)

print(
    "Network1 = first block"
)

print(
    "Network2 = remaining 4 blocks"
)

print()

for result in results:

    threshold = result["threshold"]

    exit_fraction = (
        result["exit_percentage"] / 100.0
    )

    # Approximate fraction of backbone blocks executed.
    #
    # Early exit:
    #   1 block
    #
    # Fallback:
    #   5 blocks
    #
    # Therefore:
    #   average blocks =
    #       exit_fraction*1 +
    #       fallback_fraction*5
    #

    average_blocks = (
        exit_fraction * 1.0
        + (1.0 - exit_fraction) * 5.0
    )

    compute_fraction = (
        average_blocks / 5.0
    )

    estimated_speedup = (
        1.0 / compute_fraction
    )

    print(
        f"Threshold {threshold:.2f}: "
        f"exit={result['exit_percentage']:.2f}% | "
        f"estimated compute={compute_fraction * 100:.2f}% | "
        f"estimated speedup={estimated_speedup:.3f}x"
    )

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

print()
print("=" * 70)
print("10. SAVING RESULTS")
print("=" * 70)

with open(
    RESULT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "threshold",
            "exit_count",
            "fallback_count",
            "exit_percentage",
            "exited_accuracy",
            "fallback_accuracy",
            "cascade_accuracy",
            "cascade_agreement",
        ]
    )

    writer.writeheader()

    writer.writerows(results)

print(
    "Saved:"
)

print(
    RESULT_CSV
)

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Samples                 : {len(labels)}"
)

print(
    f"Original accuracy       : "
    f"{original_accuracy * 100:.4f}%"
)

print(
    f"Forced split accuracy   : "
    f"{network2_accuracy * 100:.4f}%"
)

print(
    f"Forced split agreement  : "
    f"{original_vs_split * 100:.4f}%"
)

print(
    f"EE1 standalone accuracy : "
    f"{ee_accuracy * 100:.4f}%"
)

print()
print(
    "The table above is the important result."
)

print(
    "DO NOT train another EE or change the split "
    "until we inspect these numbers."
)

print("=" * 70) 