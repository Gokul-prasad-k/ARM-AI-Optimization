import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================


ROOT = Path(__file__).resolve().parent   # ml/keyword_spotting

MODEL_PATH = ROOT / "trained_models" / "kws_ref_model"
OUT_DIR = ROOT / "split_model"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_LAYER_NAME = "activation_4"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

print("=" * 70)
print("LOADING ORIGINAL MLCOMMONS KWS MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")
print("Input :", model.input.shape)
print("Output:", model.output.shape)


# ============================================================
# VERIFY SPLIT LAYER
# ============================================================

split_layer = model.get_layer(SPLIT_LAYER_NAME)

print("\n" + "=" * 70)
print("SPLIT POINT")
print("=" * 70)

print("Layer :", split_layer.name)
print("Type  :", split_layer.__class__.__name__)
print("Shape :", split_layer.output.shape)


# ============================================================
# BUILD NETWORK 1
# ============================================================
#
# Original input
#       |
#       v
#    split layer
#       |
#       +----> split tensor
#
# Network 1 stops exactly at activation_4.
# ============================================================

network1 = tf.keras.Model(
    inputs=model.input,
    outputs=split_layer.output,
    name="kws_network1"
)


# ============================================================
# BUILD NETWORK 2
# ============================================================
#
# Network 2 starts with exactly the shape produced by Network 1.
#
# We deliberately reuse the ORIGINAL trained layer objects.
# Therefore their trained weights are preserved.
# ============================================================

split_shape = tuple(split_layer.output.shape[1:])

network2_input = tf.keras.Input(
    shape=split_shape,
    name="kws_split_input"
)

x = network2_input

# Layers AFTER activation_4
start_found = False

for layer in model.layers:

    if layer.name == SPLIT_LAYER_NAME:
        start_found = True
        continue

    if not start_found:
        continue

    # Skip InputLayer if encountered
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue

    x = layer(x)

network2 = tf.keras.Model(
    inputs=network2_input,
    outputs=x,
    name="kws_network2"
)


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("NETWORK 1")
print("=" * 70)

network1.summary()

print("\n" + "=" * 70)
print("NETWORK 2")
print("=" * 70)

network2.summary()


# ============================================================
# VERIFY OUTPUT SHAPES
# ============================================================

assert network1.output_shape == (None, *split_shape), (
    f"Unexpected Network 1 output shape: {network1.output_shape}"
)

assert network2.output_shape == model.output_shape, (
    f"Network 2 output {network2.output_shape} "
    f"does not match original {model.output_shape}"
)

print("\nShape checks PASSED.")


# ============================================================
# SAVE KERAS MODELS
# ============================================================

network1_path = os.path.join(OUT_DIR, "network1")
network2_path = os.path.join(OUT_DIR, "network2")

network1.save(network1_path)
network2.save(network2_path)

print("\nSaved:")
print(network1_path)
print(network2_path)


# ============================================================
# CONVERT TO FLOAT32 TFLITE
# ============================================================

def convert_to_tflite(keras_model, output_path):

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Explicitly keep this experiment FLOAT32.
    # No quantization yet.
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print("Created:", output_path)


network1_tflite = os.path.join(OUT_DIR, "network1.tflite")
network2_tflite = os.path.join(OUT_DIR, "network2.tflite")

convert_to_tflite(network1, network1_tflite)
convert_to_tflite(network2, network2_tflite)


# ============================================================
# FINAL STRUCTURAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("Original input      :", model.input_shape)
print("Split layer         :", SPLIT_LAYER_NAME)
print("Split tensor        :", split_layer.output_shape)
print("Network 1 output    :", network1.output_shape)
print("Network 2 input     :", network2.input_shape)
print("Network 2 output    :", network2.output_shape)
print("Original output     :", model.output_shape)

print("\nFiles:")
print(" ", network1_tflite)
print(" ", network2_tflite)

print("\nNO EARLY EXIT HAS BEEN ADDED.")
print("This is a forced split validation model.")