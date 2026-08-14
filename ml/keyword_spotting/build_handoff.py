import os
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

import kws_util
import get_dataset as kws_data


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

OUTPUT_DIR = ROOT / "FINAL_HANDOFF"
NUM_TEST_VECTORS = 10

# Current development operating point.
# Keep configurable for STM32 benchmarking.
DEFAULT_THRESHOLD = 0.90

# Numerical tolerance for float32 TFLite vs Keras comparison.
ATOL = 1e-5
RTOL = 1e-4


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def clean_output():
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing previous handoff: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "test_vectors"))


def export_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_model = converter.convert()

    with open(path, "wb") as f:
        f.write(tflite_model)

    print(f"Exported: {path}")
    print(f"Size: {len(tflite_model):,} bytes")


def inspect_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    return {
        "input": {
            "name": inputs[0]["name"],
            "shape": inputs[0]["shape"].tolist(),
            "dtype": str(inputs[0]["dtype"]),
        },
        "output": {
            "name": outputs[0]["name"],
            "shape": outputs[0]["shape"].tolist(),
            "dtype": str(outputs[0]["dtype"]),
        },
    }


def run_tflite(interpreter, x):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    interpreter.set_tensor(inp["index"], x.astype(np.float32))
    interpreter.invoke()

    return interpreter.get_tensor(out["index"]).copy()


def load_models():
    section("1. LOADING MODELS")

    print("Original:")
    original = keras.models.load_model(ORIGINAL_MODEL)

    print("Network 1:")
    network1 = keras.models.load_model(NETWORK1_MODEL)

    print("Network 2:")
    network2 = keras.models.load_model(NETWORK2_MODEL)

    print("EE1:")
    ee1 = keras.models.load_model(EE1_MODEL)

    print("\nAll models loaded.")

    return original, network1, ee1, network2


def check_interfaces(original, network1, ee1, network2):
    section("2. INTERFACE VALIDATION")

    expected = {
        "original_input": [None, 49, 10, 1],
        "original_output": [None, 12],
        "network1_input": [None, 49, 10, 1],
        "network1_output": [None, 25, 5, 64],
        "ee1_input": [None, 25, 5, 64],
        "ee1_output": [None, 12],
        "network2_input": [None, 25, 5, 64],
        "network2_output": [None, 12],
    }

    actual = {
        "original_input": original.input_shape,
        "original_output": original.output_shape,
        "network1_input": network1.input_shape,
        "network1_output": network1.output_shape,
        "ee1_input": ee1.input_shape,
        "ee1_output": ee1.output_shape,
        "network2_input": network2.input_shape,
        "network2_output": network2.output_shape,
    }

    passed = True

    for name in expected:
        a = list(actual[name])
        e = expected[name]

        print(f"{name:20s}: {a}")

        if a != e:
            print(f"  ERROR expected: {e}")
            passed = False

    if not passed:
        raise RuntimeError("INTERFACE VALIDATION FAILED")

    print("\nINTERFACE VALIDATION: PASS")


def load_test_data():
    section("3. LOADING MLCOMMONS TEST DATA")

    # kws_util.py expects HOME and PWD on Windows.
    os.environ["HOME"] = str(REPO_ROOT)
    os.environ["PWD"] = str(ROOT)

    # Construct flags using the project's own argument parser.
    import sys

    original_argv = sys.argv
    sys.argv = [
        "build_handoff.py",
        "--data_dir", DATA_DIR,
        "--num_test_samples", "-1",
        "--feature_type", "mfcc",
        "--sample_rate", "16000",
        "--clip_duration_ms", "1000",
        "--window_size_ms", "30",
        "--window_stride_ms", "20",
        "--dct_coefficient_count", "10",
        "--model_architecture", "ds_cnn",
    ]

    try:
        flags, _ = kws_util.parse_command()
    finally:
        sys.argv = original_argv

    _, ds_test, _ = kws_data.get_training_data(flags)

    print("Dataset loaded.")

    return ds_test


def collect_vectors(ds_test):
    section("4. COLLECTING GOLDEN TEST VECTORS")

    xs = []
    ys = []

    for samples, labels in ds_test:
        xs.append(samples.numpy())
        ys.append(labels.numpy())

        total = sum(len(x) for x in xs)

        if total >= NUM_TEST_VECTORS:
            break

    x = np.concatenate(xs, axis=0)[:NUM_TEST_VECTORS]
    y = np.concatenate(ys, axis=0)[:NUM_TEST_VECTORS]

    print(f"Collected {len(x)} deterministic test samples.")
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {y.shape}")

    return x.astype(np.float32), y.astype(np.int32)


def validate_split(original, network1, network2, x):
    section("5. VERIFYING NETWORK SPLIT")

    original_out = original.predict(x, verbose=0)

    h = network1.predict(x, verbose=0)
    split_out = network2.predict(h, verbose=0)

    max_diff = float(np.max(np.abs(original_out - split_out)))
    mean_diff = float(np.mean(np.abs(original_out - split_out)))

    original_pred = np.argmax(original_out, axis=1)
    split_pred = np.argmax(split_out, axis=1)

    agreement = float(np.mean(original_pred == split_pred) * 100.0)

    print(f"Samples:                  {len(x)}")
    print(f"Prediction agreement:     {agreement:.6f}%")
    print(f"Maximum probability diff: {max_diff:.10f}")
    print(f"Mean probability diff:    {mean_diff:.10f}")

    if agreement != 100.0 or max_diff > ATOL + RTOL:
        raise RuntimeError("NETWORK SPLIT VALIDATION FAILED")

    print("\nNETWORK SPLIT VALIDATION: PASS")

    return original_out, h, split_out


def validate_tflite_models(tflite_paths, x, h, ee1):
    section("6. VERIFYING EXPORTED TFLITE MODELS")

    results = {}

    for name, path in tflite_paths.items():
        print(f"\n{name}:")
        info = inspect_tflite(path)

        print("  Input :", info["input"])
        print("  Output:", info["output"])

        results[name] = info

    # Run TFLite models.
    n1_interpreter = tf.lite.Interpreter(
        model_path=tflite_paths["network1"]
    )
    ee1_interpreter = tf.lite.Interpreter(
        model_path=tflite_paths["ee1"]
    )
    n2_interpreter = tf.lite.Interpreter(
        model_path=tflite_paths["network2"]
    )

    n1_interpreter.allocate_tensors()
    ee1_interpreter.allocate_tensors()
    n2_interpreter.allocate_tensors()

    tflite_h = []
    tflite_ee = []
    tflite_n2 = []

    for sample in x:
        sample = sample[np.newaxis, ...]

        h1 = run_tflite(n1_interpreter, sample)
        ee_out = run_tflite(ee1_interpreter, h1)
        n2_out = run_tflite(n2_interpreter, h1)

        tflite_h.append(h1[0])
        tflite_ee.append(ee_out[0])
        tflite_n2.append(n2_out[0])

    tflite_h = np.asarray(tflite_h)
    tflite_ee = np.asarray(tflite_ee)
    tflite_n2 = np.asarray(tflite_n2)

    keras_h = h
    keras_ee = ee1.predict(h, verbose=0)
    # keras_n2 = n2_interpreter  # placeholder to avoid accidental use

    # Compare N1.
    n1_diff = float(np.max(np.abs(keras_h - tflite_h)))

    # Compare EE1.
    ee_diff = float(np.max(np.abs(keras_ee - tflite_ee)))

    print("\nTFLite vs Keras:")
    print(f"Network1 max diff: {n1_diff:.10f}")
    print(f"EE1 max diff:      {ee_diff:.10f}")

    if n1_diff > ATOL + RTOL:
        raise RuntimeError("TFLITE NETWORK1 VALIDATION FAILED")

    if ee_diff > ATOL + RTOL:
        raise RuntimeError("TFLITE EE1 VALIDATION FAILED")

    print("\nTFLITE EXPORT VALIDATION: PASS")

    return tflite_h, tflite_ee, tflite_n2, results


def generate_vectors(
    x,
    labels,
    tflite_h,
    tflite_ee,
    tflite_n2,
):
    section("7. GENERATING GOLDEN TEST VECTORS")

    vector_dir = os.path.join(OUTPUT_DIR, "test_vectors")

    for i in range(len(x)):
        sample_dir = os.path.join(
            vector_dir, f"sample_{i:02d}"
        )
        os.makedirs(sample_dir)

        np.save(
            os.path.join(sample_dir, "mfcc.npy"),
            x[i]
        )

        np.save(
            os.path.join(sample_dir, "network1_output.npy"),
            tflite_h[i]
        )

        np.save(
            os.path.join(sample_dir, "ee1_output.npy"),
            tflite_ee[i]
        )

        np.save(
            os.path.join(sample_dir, "network2_output.npy"),
            tflite_n2[i]
        )

        ee_probs = tflite_ee[i]
        n2_probs = tflite_n2[i]

        ee_confidence = float(np.max(ee_probs))
        ee_prediction = int(np.argmax(ee_probs))

        if ee_confidence >= DEFAULT_THRESHOLD:
            cascade_prediction = ee_prediction
            exited = True
        else:
            cascade_prediction = int(np.argmax(n2_probs))
            exited = False

        metadata = {
            "sample_index": i,
            "true_label": int(labels[i]),
            "ee1_prediction": ee_prediction,
            "ee1_confidence": ee_confidence,
            "threshold": DEFAULT_THRESHOLD,
            "early_exit": exited,
            "network2_prediction": int(np.argmax(n2_probs)),
            "cascade_prediction": cascade_prediction,
        }

        with open(
            os.path.join(sample_dir, "metadata.json"),
            "w"
        ) as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved {len(x)} golden test vectors.")
    print(f"Directory: {vector_dir}")


def create_metadata(tflite_info):
    section("8. CREATING HANDOFF METADATA")

    metadata = {
        "project": "T-RECX",
        "task": "Keyword Spotting Early Exit",
        "framework": {
            "tensorflow": tf.__version__,
            "tflite_dtype": "float32",
        },
        "input": {
            "feature": "MFCC",
            "sample_rate": 16000,
            "clip_duration_ms": 1000,
            "window_size_ms": 30,
            "window_stride_ms": 20,
            "dct_coefficient_count": 10,
            "shape": [1, 49, 10, 1],
        },
        "models": {
            "network1": {
                "file": "network1.tflite",
                "input_shape": [1, 49, 10, 1],
                "output_shape": [1, 25, 5, 64],
            },
            "ee1": {
                "file": "ee1.tflite",
                "input_shape": [1, 25, 5, 64],
                "output_shape": [1, 12],
            },
            "network2": {
                "file": "network2.tflite",
                "input_shape": [1, 25, 5, 64],
                "output_shape": [1, 12],
            },
        },
        "cascade": {
            "default_threshold": DEFAULT_THRESHOLD,
            "logic": (
                "Run Network1. Run EE1 on Network1 output. "
                "If max(EE1 output) >= threshold, return argmax(EE1). "
                "Otherwise run Network2 on the same Network1 output "
                "and return argmax(Network2)."
            ),
        },
        "important_buffer_requirement": (
            "Network1 output [1,25,5,64] is shared by EE1 and Network2. "
            "The buffer must remain valid and unmodified until the "
            "conditional Network2 execution has completed."
        ),
        "known_reference_results": {
            "baseline_accuracy_percent": 92.1677,
            "threshold_0.90_exit_percent": 24.62,
            "threshold_0.90_cascade_accuracy_percent": 91.68,
            "threshold_0.90_prediction_agreement_percent": 99.45,
            "threshold_0.90_mac_estimated_speedup": 1.121,
        },
        "important_note": (
            "The 0.90 threshold was selected during exploratory "
            "test-set analysis. It is not a clean held-out threshold."
        ),
    }

    path = os.path.join(
        OUTPUT_DIR, "metadata.json"
    )

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    section("T-RECX FINAL HANDOFF BUILD")

    print(f"Base directory : {BASE_DIR}")
    print(f"Dataset        : {DATA_DIR}")
    print(f"Output         : {OUTPUT_DIR}")
    print(f"Threshold      : {DEFAULT_THRESHOLD}")

    clean_output()

    original, network1, ee1, network2 = load_models()

    check_interfaces(
        original,
        network1,
        ee1,
        network2,
    )

    ds_test = load_test_data()

    x, labels = collect_vectors(ds_test)

    original_out, h, split_out = validate_split(
        original,
        network1,
        network2,
        x,
    )

    # --------------------------------------------------------
    # Export TFLite
    # --------------------------------------------------------

    section("6. EXPORTING FLOAT32 TFLITE MODELS")

    tflite_paths = {
        "network1": os.path.join(
            OUTPUT_DIR, "network1.tflite"
        ),
        "ee1": os.path.join(
            OUTPUT_DIR, "ee1.tflite"
        ),
        "network2": os.path.join(
            OUTPUT_DIR, "network2.tflite"
        ),
    }

    export_tflite(network1, tflite_paths["network1"])
    export_tflite(ee1, tflite_paths["ee1"])
    export_tflite(network2, tflite_paths["network2"])

    # --------------------------------------------------------
    # Validate TFLite
    # --------------------------------------------------------

    tflite_h, tflite_ee, tflite_n2, tflite_info = (
        validate_tflite_models(
            tflite_paths,
            x,
            h,
            ee1,
        )
    )
    # Compare Network2 TFLite with Keras Network2.
    n2_keras = network2.predict(h, verbose=0)

    n2_diff = float(
        np.max(np.abs(n2_keras - tflite_n2))
    )

    print(f"Network2 max diff: {n2_diff:.10f}")

    if n2_diff > ATOL + RTOL:
        raise RuntimeError(
            "TFLITE NETWORK2 VALIDATION FAILED"
        )

    print("Network2 TFLite validation: PASS")

    generate_vectors(
        x,
        labels,
        tflite_h,
        tflite_ee,
        tflite_n2,
    )

    create_metadata(tflite_info)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    section("FINAL HANDOFF RESULT")

    print("PASS: Models loaded")
    print("PASS: Interfaces verified")
    print("PASS: Network split verified")
    print("PASS: Float32 TFLite exports verified")
    print("PASS: Golden test vectors generated")
    print("PASS: Handoff metadata generated")

    print()
    print("READY TO SHIP")
    print()
    print(f"Handoff folder:")
    print(OUTPUT_DIR)

    print()
    print("Files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for file in files:
            print(f"{indent}  {file}")


if __name__ == "__main__":
    main()