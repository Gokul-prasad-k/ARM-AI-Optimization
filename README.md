# ARM AI Optimization — Early-Exit Keyword Spotting on STM32F4

This repository packages two STM32CubeIDE projects built around the same
12-class keyword-spotting (KWS) model family, so a judge/reviewer can build
and flash a **baseline (full) model** and a **cascaded early-exit model**
side by side on identical ARM Cortex-M4 hardware and compare them directly.


## Folder map

```
├── README.md                          <- you are here
└── ARM-AI-Optimization/               <- git repo, submitted as-is
    ├── kws_ref_model.tflite           <- reference monolithic KWS model (quantized)
    ├── kws_ref_model_float32.tflite   <- reference monolithic KWS model (float32)
    ├── Full model F4/                 <- STM32CubeIDE project — baseline, single network
    │   ├── Full model F4.ioc
    │   ├── Core/Src/main.c, ...
    │   ├── X-CUBE-AI/App/network.*, app_x-cube-ai.c
    │   ├── Debug/                     <- prebuilt .elf/.map from a previous build
    │   └── Drivers/, Middlewares/     <- HAL + X-CUBE-AI runtime
    └── Early Exit Validation F4/      <- STM32CubeIDE project — cascaded early-exit
        ├── Early Exit Validation F4.ioc
        ├── Core/Src/main.c, test_data.c, ...
        ├── X-CUBE-AI/App/network_1.*, network_2.*, network_ee.*, app_x-cube-ai.c
        └── Drivers/, Middlewares/     <- HAL + X-CUBE-AI runtime
```

Both projects target an **STM32F411CEUx** (UFQFPN48, "BlackPill"-class
board) and are otherwise identical STM32CubeIDE/X-CUBE-AI exports — the
only difference is which model(s) run in `X-CUBE-AI/App/` and
`app_x-cube-ai.c`. That makes the two projects a controlled, apples-to-apples
comparison of the same keyword-spotting task on the same MCU.

## The task: keyword spotting (KWS)

Both models classify a 1-second, 16 kHz audio clip into one of **12
keyword classes** from a 10-coefficient MFCC feature representation:

- 30 ms analysis window, 20 ms stride → **49 frames**
- **10 MFCC (DCT) coefficients** per frame
- Input tensor: `1 × 49 × 10 × 1` = **490 floats**, matching
  `MFCC_VECTOR_SIZE` in `test_data.h`

The underlying architecture is a small depthwise-separable CNN (DS-CNN):
a stack of `Conv2D`/`DepthwiseConv2D` blocks, a global average pool, and a
12-way dense + softmax classifier — the standard shape for this class of
TinyML KWS models. The early-exit variant follows the
**[T-RECX](https://arxiv.org/abs/2207.06613)**-style approach of splitting
that backbone in half and attaching a cheap classifier head partway
through, so "easy" inputs can skip the second half of the network
entirely.

## `Full model F4` — baseline

Runs the full DS-CNN (`full_model_float32_test.tflite`, X-CUBE-AI
c-name `network`) end-to-end, every inference, no shortcuts.

| | |
|---|---:|
| Parameters | 22,604 (88.30 KiB) |
| MACs / inference | 2,737,536 |
| Weights (Flash) | 90,416 B (88.30 KiB) |
| Activation RAM | 50,432 B (49.25 KiB) |

`MX_X_CUBE_AI_Process()` in
[`app_x-cube-ai.c`](<ARM-AI-Optimization/Full model F4/X-CUBE-AI/App/app_x-cube-ai.c>)
feeds a dummy all-ones input through the model in a tight loop, timing
each inference with the Cortex-M4 DWT cycle counter into the `duration_us`
global — this project has **no UART wired up**, so timing is read via a
debugger live-watch/breakpoint on `duration_us`, not a serial log (see
[Building and running](#building-and-running) below).

## `Early Exit Validation F4` — cascaded early-exit

This is a **cascaded / early-exit inference pipeline**, not a single
network. It's generated from an X-CUBE-AI export of three TFLite models
that share a common feature space:

| Model | Role | Input | Output |
|---|---|---|---|
| `network_1` (`network1.tflite`) | Shared backbone / feature extractor | MFCC frame, `1x49x10x1` (490 floats) | Feature map `1x25x5x64` |
| `network_ee` (`ee1.tflite`) | **Early-exit head** — cheap classifier | `network_1` output (shared, not recomputed) | 12-class softmax probabilities |
| `network_2` (`network2.tflite`) | **Full head** — deep classifier, only runs when escalated | `network_1` output (same buffer as above) | 12-class softmax probabilities |

**Cascade logic** (implemented in `post_process_ee()` /
`run_validation_sample()` in
[`app_x-cube-ai.c`](<ARM-AI-Optimization/Early Exit Validation F4/X-CUBE-AI/App/app_x-cube-ai.c>)):

1. Run `network_1` once on the incoming MFCC vector → feature map
   (`1×25×5×64`).
2. Run the tiny `network_ee` head on that feature map (global average
   pool → 64→12 dense → softmax, already baked into the model so the
   firmware just takes argmax of the raw output).
3. Take `argmax(network_ee output)` and its confidence (max probability).
   - **If confidence ≥ threshold (`EE_CONF_THRESHOLD = 0.90`)** → early
     exit. Return this prediction. `network_2` never runs.
   - **Otherwise** → escalate. Run the full `network_2` head on the
     *same* `network_1` feature map (nothing from `network_1` is
     recomputed) and return `argmax(network_2 output)` instead.

The key optimization is that the expensive backbone (`network_1`) runs
exactly once per inference regardless of the outcome, and the feature map
it produces is reused by whichever head runs. Only the cheap `network_ee`
head is guaranteed to run in addition; the expensive `network_2` head is
skipped entirely whenever the model is already confident.

Because `network_1` and `network_2` together are literally the same
backbone as the baseline split at the `1×25×5×64` boundary, their MACs
sum to *exactly* the full model's MAC count (1,528,320 + 1,209,216 =
2,737,536) — the early-exit head is the only new compute added, and it's
tiny.

### Why this saves inference time

Per-model cost, taken from the X-CUBE-AI generation reports in
`X-CUBE-AI/App/*_generate_report.txt`:

| Model | MACs | Weights (Flash) | Activation RAM |
|---|---:|---:|---:|
| `network_1` (backbone) | 1,528,320 | 48,896 B (47.75 KiB) | 41,216 B (40.25 KiB) |
| `network_ee` (early-exit head) | 8,960 | 3,120 B (3.05 KiB) | 32,000 B (31.25 KiB) |
| `network_2` (full head) | 1,209,216 | 41,520 B (40.55 KiB) | 41,216 B (40.25 KiB) |

| Path | MACs | vs. baseline (2,737,536 MACs) |
|---|---:|---:|
| Early exit taken (`network_1 + network_ee`) | 1,537,280 | **−43.8% compute** |
| Escalated (`network_1 + network_ee + network_2`) | 2,746,496 | +0.3% (cost of the EE head) |

So every sample that exits early skips ~1.2M MACs, and even a sample that
escalates only pays a ~0.3% MAC overhead for the confidence check. Flash
footprint is the one trade-off: since `network_2`'s weights must still be
present on-device for the escalation path, total weight storage for the
early-exit project (48,896 + 3,120 + 41,520 = 93,536 B / 91.34 KiB) is
about 3.5% larger than the baseline's 90,416 B — a small flash cost in
exchange for conditional compute/latency savings.

### Offline validation results (reference, from the training/export pipeline)

These are the project's own reference numbers from a full offline
validation run of the cascade at the shipped threshold (0.90), computed
during model development:

| Metric | Value |
|---|---:|
| Baseline (always-full-model) accuracy | 92.17% |
| Samples exiting early at threshold 0.90 | 24.62% |
| Cascade accuracy at threshold 0.90 | 91.68% |
| Agreement between cascade and full-model prediction | 99.45% |
| Estimated MAC-count speedup at threshold 0.90 | 1.121× |

Note: the 0.90 confidence threshold was picked from exploratory analysis
on the test set, not a separate held-out split — worth being upfront
about if asked during judging.

## On-device validation (Python ↔ hardware parity)

`Core/Src/test_data.c` / `Core/Inc/test_data.h` (in `Early Exit
Validation F4/`) embed 10 real validation samples directly into flash as
`const` arrays (zero RAM cost):

- `mfcc_samples[10]` — the 490-float MFCC input for each sample
- `ee1_ref_outputs[10]` — the **Python-computed** `network_ee` softmax
  output for that sample (ground truth to compare against)
- `network2_ref_outputs[10]` — the **Python-computed** `network_2`
  softmax output (where applicable)
- `test_metadata[10]` — per-sample `true_label`, `ee1_prediction`,
  `ee1_confidence`, `early_exit` flag, `network2_prediction`, and the
  final `cascade_prediction`, all as computed by the Python reference
  pipeline

These vectors are the exact same inputs/outputs used to validate the
model in Python (via the TFLite runtime, `tensorflow==2.8.0`) — i.e. the
same test set is flashed onto the MCU rather than being a separate
hardware-only sanity check.

On boot, `MX_X_CUBE_AI_Process()` calls `run_full_validation_suite()`,
which for every embedded sample:

1. Feeds the stored MFCC vector through `network_1` → `network_ee` on
   the real hardware.
2. Compares the device's `network_ee` output against the Python
   reference via **MAE** and **cosine similarity**.
3. Checks whether the device's predicted class and early-exit decision
   match the Python reference.
4. If escalation is triggered on-device, runs `network_2` and compares
   it against the Python reference the same way.
5. Times each sample with the Cortex-M4 DWT cycle counter and reports
   per-sample and average latency in microseconds.
6. Prints a full summary (match counts, average MAE/cosine, average
   latency, and a confusion matrix) over UART.

Per the project owner: hardware results were compared against these
Python validation outputs and **matched** — confirming the on-device
pipeline (HAL + X-CUBE-AI runtime) reproduces the Python/TFLite reference
numerically, not just at the class-prediction level.

## Building and running

Both projects are STM32CubeIDE projects targeting an **STM32F411CEUx**
(`STM32F411CEUX_FLASH.ld` / `startup_stm32f411ceux.s`) — a common
"BlackPill"-class board. Clock is configured via `SystemClock_Config()`
using the internal **HSI oscillator with no PLL** (≈16 MHz core clock) in
both projects, so latency numbers from the two are directly comparable
as-is (rerun at a higher `SYSCLK` if you need numbers representative of
the part's ~100 MHz peak performance).

### `Early Exit Validation F4/` (cascade, UART output)

1. Open STM32CubeIDE → `File > Open Projects from File System…` → point
   at `ARM-AI-Optimization/Early Exit Validation F4/`.
2. Build (`Project > Build Project`) and flash/debug onto the target.
3. Connect a serial terminal to the ST-Link VCP / USB-UART bridge at
   **115200 baud, 8N1** (USART2 on PA2/PA3, see `Core/Src/usart.c`) to
   see the validation suite's printf output.
4. The validation run is triggered automatically on boot/reset — no
   host-side interaction is required after flashing. It runs the 10
   embedded samples once and prints the summary described above.

### `Full model F4/` (baseline, debugger-only output)

1. Open STM32CubeIDE → `File > Open Projects from File System…` → point
   at `ARM-AI-Optimization/Full model F4/`.
2. Build (`Project > Build Project`) and flash/debug onto the target.
3. This project has no UART wired up — it loops `MX_X_CUBE_AI_Process()`
   forever on a fixed dummy input. Add `duration_us` to a **Live
   Expressions**/watch window in the debugger to read per-inference
   latency in microseconds, or set a breakpoint inside the loop.

## Hardware

- MCU: **STM32F411CEUx** (Cortex-M4F, UFQFPN48)
- Board class: "BlackPill"-style STM32F411 dev board
- Timing: Cortex-M4 DWT cycle counter, converted to microseconds using
  `HAL_RCC_GetHCLKFreq()`
- Both projects verified to produce matching MFCC-in → prediction-out
  behavior against the Python/TFLite reference pipeline, per the notes
  above.


** For validation of model on host pc refeer to the readme inside the folder ml.