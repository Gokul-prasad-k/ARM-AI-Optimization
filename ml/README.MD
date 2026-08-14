# Early Exit DS-CNN Extension

An implementation of an **Early Exit DS-CNN** built on **MLCommons Tiny v1.1 Keyword Spotting**. This project splits the original DS-CNN into two independent networks, trains an intermediate Early Exit classifier (EE1), and evaluates the accuracy vs. compute trade-off for edge AI deployment.

---

## Repository Structure

```text
ml/
└── keyword_spotting/
    ├── trained_models/                 # Original MLCommons DS-CNN
    ├── split_model/                    # Network1 & Network2
    ├── ee1_probe/                      # Trained EE1 model
    ├── ee1_threshold_results.csv/      # Validation results
    ├── FINAL_HANDOFF/                  # Deployment package
    ├── split_model.py
    ├── validate_split.py
    ├── training_ee1.py
    ├── final_validation.py
    ├── compute_analysis.py
    ├── build_handoff.py
    ├── requirements-lock.txt
    └── setup_env.ps1

mlcommons_data/                  # Downloaded automatically
```

---

## Quick Start

From the repository root, run:

```powershell
.\ml\keyword_spotting\setup_env.ps1
```

The setup script automatically:

- Creates a Python virtual environment
- Installs all required dependencies
- Downloads the MLCommons Speech Commands dataset (only if it is not already present)

To activate the environment later:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Execution Pipeline

Run the scripts in the following order.

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `split_model.py` | Split the original DS-CNN into Network1 & Network2 |
| 2 | `validate_split.py` | Verify the split reproduces the original model |
| 3 | `training_ee1.py` | Train the Early Exit classifier |
| 4 | `final_validation.py` | Evaluate threshold-based early exit performance |
| 5 | `compute_analysis.py` | Calculate MACs, parameters and theoretical speedup |
| 6 | `build_handoff.py` | Generate deployment-ready TFLite models and test vectors |

---

# Script Reference

## split_model.py

Creates a forced architectural split of the original MLCommons DS-CNN.

### What it does

- Loads the original trained model
- Splits the network at `activation_4`
- Creates **Network1** and **Network2**
- Performs structural validation
- Exports Keras and Float32 TFLite models

### Output

| File | Description |
|------|-------------|
| `split_model/network1/` | First-stage Keras model |
| `split_model/network2/` | Second-stage Keras model |
| `network1.tflite` | Float32 Network1 |
| `network2.tflite` | Float32 Network2 |

### Run

```bash
python split_model.py
```

---

## validate_split.py

Validates that the split architecture is functionally identical to the original MLCommons model.

### What it does

- Loads the original model and both split networks
- Runs the complete MLCommons test dataset
- Compares output probabilities sample-by-sample
- Measures prediction agreement and numerical error
- Confirms that the forced split preserves original model behavior

### Expected Result

- 100% prediction agreement
- Near-zero probability difference
- Split validation **PASS**

### Run

```bash
python validate_split.py
```

---

## training_ee1.py

Trains the first Early Exit (EE1) classifier using frozen intermediate features from Network1.

### What it does

- Loads the MLCommons Speech Commands dataset
- Loads the frozen Network1 model
- Generates intermediate feature tensors
- Trains a Global Average Pooling + Dense classifier
- Evaluates EE1 standalone accuracy
- Saves the trained EE1 model

### Output

| File | Description |
|------|-------------|
| `ee1_probe/` | Trained EE1 Keras model |

### Run

```bash
python training_ee1.py
```

---

## final_validation.py

Primary evaluation script for the complete Early Exit pipeline.

### What it does

- Validates the original DS-CNN and split models
- Evaluates EE1 standalone accuracy
- Simulates multiple confidence thresholds
- Reports exit rate, cascade accuracy and prediction agreement
- Exports threshold evaluation results

### Output

| File | Description |
|------|-------------|
| `ee1_threshold_results.csv` | Threshold evaluation results |

### Run

```bash
python final_validation.py
```

---

## compute_analysis.py

Performs theoretical compute analysis of the Early Exit architecture.

### Reports

- Parameter count
- MAC operations
- Early-exit compute cost
- Average compute reduction
- Estimated speedup across evaluated thresholds

### Run

```bash
python compute_analysis.py
```

> **Note:** The reported speedup is a MAC-based architectural estimate and is **not** measured STM32 runtime.

---

## build_handoff.py

Builds the final deployment package for STM32 integration.

### What it does

- Validates model interfaces
- Exports Float32 TFLite models
- Verifies TFLite outputs against Keras
- Generates deterministic golden test vectors
- Creates deployment metadata
- Packages everything into `FINAL_HANDOFF/`

### Output

| File / Folder | Description |
|--------------|-------------|
| `network1.tflite` | First-stage backbone |
| `ee1.tflite` | Early Exit classifier |
| `network2.tflite` | Final-stage classifier |
| `test_vectors/` | Golden validation vectors |
| `metadata.json` | Deployment metadata |

### Run

```bash
python build_handoff.py
```

---

# Expected Outputs

| Stage | Output |
|--------|--------|
| Model Split | `split_model/` |
| EE1 Training | `ee1_probe/` |
| Threshold Evaluation | `ee1_threshold_results.csv` |
| Compute Analysis | Console report |
| Deployment Package | `FINAL_HANDOFF/` |

---

# Common Issues

## Dataset not found

**Error**

```text
mlcommons_data not found
```

**Solution**

```powershell
.\ml\keyword_spotting\setup_env.ps1
```

---

## TensorFlow not installed

**Error**

```text
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution**

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Split model missing

**Error**

```text
split_model/network1 not found
```

**Solution**

```bash
python split_model.py
```

---

## EE1 model missing

**Error**

```text
ee1_probe not found
```

**Solution**

```bash
python training_ee1.py
```

---

## Interface validation failed

This usually means `Network1`, `Network2`, and `EE1` were generated from different model versions.

Rebuild the pipeline in this order:

```text
split_model.py
      ↓
validate_split.py
      ↓
training_ee1.py
      ↓
final_validation.py
      ↓
compute_analysis.py
      ↓
build_handoff.py
```

---

## Notes

- All scripts use **relative paths**.
- The Speech Commands dataset is downloaded only once and reused by every script.
- The preprocessing pipeline is identical to the original MLCommons Tiny v1.1 implementation for reproducible evaluation.