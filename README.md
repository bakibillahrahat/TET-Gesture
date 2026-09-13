# TET-Gesture: Temporal Efficient Training for Neuromorphic Gesture Recognition

> **Research project:** Extending Temporal Efficient Training (TET) to Spiking Neural Networks for event-based visual recognition, with a focus on the IBM DVS128 Gesture benchmark.

## Tech Stack

- **Python 3.12+**
- **PyTorch 2.x** — deep learning, training, autograd
- **CUDA / NVIDIA GPU** — accelerated SNN training (Kaggle Tesla T4/P100 compatible)
- **NumPy** — numerical computation
- **Pandas** — experiment tables and metrics
- **Matplotlib** — training curves, evaluation plots, and result visualization
- **Jupyter Notebook / Kaggle** — primary experimental environment
- **Spiking Neural Networks (SNNs)** — temporal event-based computation
- **Temporal Efficient Training (TET)** — temporal regularization / loss formulation
- **VGG-style SNN architecture** — convolutional spiking backbone

---

## Overview

This repository contains a self-contained implementation of **Temporal Efficient Training (TET)** for neuromorphic vision. The project investigates how temporal regularization can improve the optimization and generalization of spiking neural networks (SNNs) on event-based image and gesture recognition tasks.

The implementation is designed around the principles introduced in:

> Deng, S., Li, Y., Zhang, S., and Gu, S. **Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting.** ICLR 2022.

The main experimental target is **IBM DVS128 Gesture**, an event-based gesture-recognition dataset containing 11 gesture classes. The notebook also includes infrastructure for **CIFAR10-DVS** experiments.

## Research Objective

The central research question is:

> **Can Temporal Efficient Training improve the temporal learning behavior and recognition performance of spiking neural networks for event-based gesture recognition without introducing excessive computational or memory overhead?**

The project compares a conventional SNN training objective against TET-regularized training and provides hooks for controlled ablation experiments.

## Key Components

### 1. Event-based input processing

The pipeline supports event-camera data and converts asynchronous events into temporal tensors suitable for SNN processing. The current implementation uses:

- Two event polarities/channels
- Spatial resizing to **48 × 48**
- **10 simulation time steps**
- Train/validation/test partitioning
- Dataset discovery for common nested dataset layouts
- Binary event-data parsing utilities where required

### 2. VGG-style Spiking Neural Network

The main model is a VGG-inspired convolutional SNN with leaky integrate-and-fire style neuron dynamics. The implementation exposes neuron parameters such as:

- Membrane threshold: `1.0`
- Membrane time constant: `0.5`
- Surrogate-gradient parameterization
- Dropout regularization
- Temporal simulation across multiple steps

### 3. Temporal Efficient Training (TET)

The training pipeline supports both conventional training and TET-based training.

The TET objective follows the temporal regularization idea of encouraging useful representations across simulation time rather than relying only on the final temporal output.

The implementation exposes the TET regularization coefficient:

```text
lambda = 1e-3
```

and allows TET to be enabled or disabled through the central experiment configuration.

### 4. Reproducible experiments

A centralized configuration controls dataset settings, model parameters, optimization, runtime behavior, and experiment switches. Random seeds are explicitly initialized for Python, NumPy, and PyTorch.

Default seed:

```text
42
```

### 5. Training and evaluation

The notebook includes infrastructure for:

- Baseline SNN training
- TET training
- Validation monitoring
- Early stopping
- Learning-rate scheduling
- Gradient accumulation
- Checkpoint management
- Multi-seed experiments
- Ablation experiments
- Prediction export
- Metric and table generation
- Figure generation

---

## Experimental Configuration

The current notebook provides a centralized `CONFIG` dictionary. Important defaults include:

| Parameter | Default |
|---|---:|
| Model | `VGGSNN` |
| Input size | `48 × 48` |
| Time steps | `10` |
| Batch size | `8` |
| Gradient accumulation | `2` |
| Effective batch size | `16` |
| Epochs | `20` |
| Learning rate | `3e-4` |
| Weight decay | `2e-3` |
| Dropout | `0.40` |
| Label smoothing | `0.1` |
| TET λ | `1e-3` |
| Random seed | `42` |
| Precision | FP32 |

> **Note:** The notebook currently defaults to **CIFAR10-DVS** in its central configuration while the repository's primary research direction is **DVS128 Gesture**. Update `dataset_root`, `dataset_name`, `num_classes`, and `class_names` in `CONFIG` when running the DVS128 Gesture experiment.

---

## Datasets

### IBM DVS128 Gesture

The primary target is the event-based **DVS128 Gesture** benchmark with **11 classes**. The dataset contains temporal streams captured by a Dynamic Vision Sensor (DVS) during human gesture demonstrations.

Expected class configuration:

```text
11 classes
```

### CIFAR10-DVS

The implementation also supports the **CIFAR10-DVS** benchmark with 10 object classes:

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

The dataset loader is designed to discover nested dataset structures and construct reproducible train/validation/test partitions.

---

## Repository Structure

```text
TET-Gesture/
├── README.md
├── tet-dvs128-gesture-kaggle.ipynb
└── results/
    ├── figures/
    ├── metrics/
    ├── predictions/
    └── tables/
```

### Main notebook

`tet-dvs128-gesture-kaggle.ipynb` is the primary executable research artifact. It contains the environment setup, reproducibility configuration, data pipeline, SNN implementation, TET training, evaluation, and experiment-output generation.

### Results directory

The notebook is configured to organize generated artifacts into:

- `results/figures/` — plots and visualizations
- `results/metrics/` — numerical evaluation metrics
- `results/predictions/` — model predictions
- `results/tables/` — experiment tables

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bakibillahrahat/TET-Gesture.git
cd TET-Gesture
```

### 2. Recommended environment

The notebook was developed for a GPU-enabled Python environment such as Kaggle with an NVIDIA GPU.

Core packages:

```bash
pip install numpy pandas matplotlib torch torchvision jupyter
```

> For Kaggle execution, most of the required Python stack is already available.

### 3. Configure the dataset

Open:

```text
tet-dvs128-gesture-kaggle.ipynb
```

and update the dataset configuration in `CONFIG`.

For DVS128 Gesture, configure:

```python
CONFIG["dataset_name"] = "DVS128 Gesture"
CONFIG["num_classes"] = 11
CONFIG["dataset_root"] = Path("/path/to/DvsGesture")
```

### 4. Run the notebook

Execute the notebook from top to bottom. The pipeline will:

1. Detect the available accelerator.
2. Initialize deterministic experiment settings.
3. Discover and parse the dataset.
4. Build train/validation/test splits.
5. Construct the VGG-style SNN.
6. Train the conventional baseline.
7. Train the TET configuration.
8. Evaluate both configurations.
9. Save metrics, predictions, tables, checkpoints, and figures when enabled.

---

## Baseline vs. TET

The project is structured around a controlled comparison:

| Experiment | TET | Purpose |
|---|---:|---|
| Baseline SNN | No | Reference conventional SNN training |
| TET-SNN | Yes | Evaluate temporal efficient training |
| Ablation | Configurable | Study sensitivity to TET components |
| Multi-seed | Configurable | Measure robustness across random seeds |

This structure is intended to keep the experimental comparison reproducible and easy to extend.

---

## Research Outputs

The experiment pipeline is designed to produce research-ready artifacts including:

- Training and validation curves
- Accuracy/loss comparisons
- Per-class evaluation metrics
- Confusion matrices
- Baseline vs. TET comparison tables
- Prediction files
- Checkpoints
- Ablation results
- Multi-seed summaries

Numerical claims should be taken from the generated files in `results/` after the corresponding experiment has been executed.

---

## Reproducibility

The project uses a fixed default seed of `42` and provides deterministic seed initialization for Python, NumPy, and PyTorch.

For stronger statistical reporting, the configuration also includes optional seeds:

```text
42, 123, 2026
```

GPU execution can still exhibit implementation- or hardware-dependent nondeterminism despite deterministic seed settings. Reported results should therefore include the execution environment and, where possible, multiple seeds.

---

## Computational Notes

The implementation includes GPU detection and CUDA memory diagnostics. It is intended primarily for GPU execution, particularly Kaggle environments with Tesla T4/P100-class accelerators.

Memory-conscious features include:

- Small micro-batches
- Gradient accumulation
- Optional mixed precision support
- CUDA memory configuration
- Explicit garbage collection
- Checkpoint/output management

The current configuration uses **FP32** by default to preserve numerical stability during the research experiments.

---

## Citation

If you use the TET method, please cite the original work:

```bibtex
@inproceedings{deng2022tet,
  title     = {Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting},
  author    = {Deng, Shikuang and Li, Yuhang and Zhang, Shanghang and Gu, Shi},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

For this repository and any resulting research publication, please cite this implementation as appropriate.

---

## Project Status

**Research / Experimental**

The repository is intended for reproducible experimentation on temporal learning in SNNs. Final performance conclusions should be based on completed baseline, TET, ablation, and multi-seed experiments rather than a single training run.

## Author

**Md. Bakibillah Rahat**  
Computer Science & Engineering  
GitHub: [@bakibillahrahat](https://github.com/bakibillahrahat)

---

## License

See the repository for the applicable license and dataset-specific terms.
