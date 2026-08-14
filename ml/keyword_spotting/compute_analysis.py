import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


from pathlib import Path

ROOT = Path(__file__).resolve().parent

NETWORK1_PATH = ROOT / "split_model" / "network1"
NETWORK2_PATH = ROOT / "split_model" / "network2"
EE1_PATH = ROOT / "ee1_probe"

print("=" * 70)
print("T-RECX ACTUAL COMPUTE ANALYSIS")
print("=" * 70)

print("\nLoading models...")

network1 = keras.models.load_model(NETWORK1_PATH)
network2 = keras.models.load_model(NETWORK2_PATH)
ee1 = keras.models.load_model(EE1_PATH)

print("Models loaded.")


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_params(model):
    return sum(
        np.prod(w.shape)
        for w in model.weights
    )


print("\n" + "=" * 70)
print("PARAMETERS")
print("=" * 70)

print(
    "Network 1 parameters :",
    count_params(network1)
)

print(
    "Network 2 parameters :",
    count_params(network2)
)

print(
    "EE1 parameters        :",
    count_params(ee1)
)


# ============================================================
# MAC CALCULATION
# ============================================================

def conv2d_macs(layer):
    input_shape = layer.input_shape
    output_shape = layer.output_shape

    kernel_h, kernel_w, in_channels, out_channels = (
        layer.kernel.shape
    )

    batch, out_h, out_w, out_c = output_shape

    return (
        out_h *
        out_w *
        out_c *
        kernel_h *
        kernel_w *
        in_channels
    )


def depthwise_macs(layer):
    output_shape = layer.output_shape

    kernel_h, kernel_w, in_channels, channel_multiplier = (
        layer.depthwise_kernel.shape
    )

    batch, out_h, out_w, out_c = output_shape

    return (
        out_h *
        out_w *
        out_c *
        kernel_h *
        kernel_w
    )


def dense_macs(layer):
    weights = layer.kernel.shape

    return (
        weights[0] *
        weights[1]
    )


print("\n" + "=" * 70)
print("NETWORK 1 OPERATIONS")
print("=" * 70)

network1_macs = 0

for layer in network1.layers:

    macs = 0

    if isinstance(layer, keras.layers.Conv2D):
        macs = conv2d_macs(layer)

    elif isinstance(layer, keras.layers.DepthwiseConv2D):
        macs = depthwise_macs(layer)

    elif isinstance(layer, keras.layers.Dense):
        macs = dense_macs(layer)

    if macs:

        network1_macs += macs

        print(
            f"{layer.name:30s}"
            f"{layer.__class__.__name__:22s}"
            f"{macs:12,d}"
        )


print("\nNetwork 1 total MACs:")
print(f"{network1_macs:,}")


print("\n" + "=" * 70)
print("NETWORK 2 OPERATIONS")
print("=" * 70)

network2_macs = 0

for layer in network2.layers:

    macs = 0

    if isinstance(layer, keras.layers.Conv2D):
        macs = conv2d_macs(layer)

    elif isinstance(layer, keras.layers.DepthwiseConv2D):
        macs = depthwise_macs(layer)

    elif isinstance(layer, keras.layers.Dense):
        macs = dense_macs(layer)

    if macs:

        network2_macs += macs

        print(
            f"{layer.name:30s}"
            f"{layer.__class__.__name__:22s}"
            f"{macs:12,d}"
        )


print("\nNetwork 2 total MACs:")
print(f"{network2_macs:,}")


# ============================================================
# EE1
# ============================================================

print("\n" + "=" * 70)
print("EE1 OPERATIONS")
print("=" * 70)

ee1_macs = 0

for layer in ee1.layers:

    macs = 0

    if isinstance(layer, keras.layers.Dense):
        macs = dense_macs(layer)

    if macs:

        ee1_macs += macs

        print(
            f"{layer.name:30s}"
            f"{layer.__class__.__name__:22s}"
            f"{macs:12,d}"
        )


print("\nEE1 total MACs:")
print(f"{ee1_macs:,}")


# ============================================================
# TOTAL
# ============================================================

total_macs = network1_macs + network2_macs

print("\n" + "=" * 70)
print("FINAL COMPUTE MODEL")
print("=" * 70)

print(
    f"Network 1              : {network1_macs:,} MACs"
)

print(
    f"Network 2              : {network2_macs:,} MACs"
)

print(
    f"Full model             : {total_macs:,} MACs"
)

print(
    f"EE1                    : {ee1_macs:,} MACs"
)

print(
    f"Early-exit path        : "
    f"{network1_macs + ee1_macs:,} MACs"
)

print(
    "\nEarly-exit compute ratio:"
)

print(
    (
        (network1_macs + ee1_macs)
        / total_macs
    )
    * 100,
    "%"
)


# ============================================================
# THRESHOLD TABLE USING ACTUAL COMPUTE
# ============================================================

exit_rates = {
    0.50: 79.55,
    0.60: 66.05,
    0.70: 54.07,
    0.75: 47.22,
    0.80: 40.27,
    0.85: 33.33,
    0.90: 24.62,
    0.92: 20.37,
    0.95: 13.50,
    0.97: 8.43,
    0.99: 1.96,
}


print("\n" + "=" * 70)
print("EXPECTED COMPUTE BY THRESHOLD")
print("=" * 70)

print(
    "\nThreshold | Exit % | Avg MACs | "
    "Compute % | Speedup"
)

print("-" * 65)

for threshold, exit_percent in exit_rates.items():

    p = exit_percent / 100.0

    early_cost = network1_macs + ee1_macs

    average_macs = (
        p * early_cost
        +
        (1 - p) * total_macs
    )

    compute_percent = (
        average_macs /
        total_macs
    ) * 100

    speedup = (
        total_macs /
        average_macs
    )

    print(
        f"{threshold:9.2f} | "
        f"{exit_percent:6.2f}% | "
        f"{average_macs:9,.0f} | "
        f"{compute_percent:8.2f}% | "
        f"{speedup:7.3f}x"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)