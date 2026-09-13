<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-76B900?logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/SNN-Spiking%20Neural%20Networks-6A1B9A" alt="Spiking Neural Networks" />
  <img src="https://img.shields.io/badge/TET-Temporal%20Efficient%20Training-0A7EA4" alt="Temporal Efficient Training" />
</p>

# TET-Gesture: Temporal Efficient Training for Neuromorphic Gesture Recognition

> **Research project:** Extending Temporal Efficient Training (TET) to Spiking Neural Networks for event-based visual recognition, with a focus on the IBM DVS128 Gesture benchmark.

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

### CIFAR10-DVS

The implementation also supports the **CIFAR10-DVS** benchmark with 10 object classes:

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

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

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bakibillahrahat/TET-Gesture.git
cd TET-Gesture
```

### 2. Recommended environment

The notebook was developed for a GPU-enabled Python environment such as Kaggle with an NVIDIA GPU.

```bash
pip install numpy pandas matplotlib torch torchvision jupyter
```

### 3. Configure DVS128 Gesture

Open `tet-dvs128-gesture-kaggle.ipynb` and update the dataset configuration:

```python
CONFIG["dataset_name"] = "DVS128 Gesture"
CONFIG["num_classes"] = 11
CONFIG["dataset_root"] = Path("/path/to/DvsGesture")
```

### 4. Run the notebook

Execute the notebook from top to bottom. The pipeline will detect the accelerator, initialize reproducible settings, load the dataset, build the VGG-style SNN, train baseline and TET models, evaluate them, and save experiment outputs.

---

## Baseline vs. TET

| Experiment | TET | Purpose |
|---|---:|---|
| Baseline SNN | No | Reference conventional SNN training |
| TET-SNN | Yes | Evaluate temporal efficient training |
| Ablation | Configurable | Study sensitivity to TET components |
| Multi-seed | Configurable | Measure robustness across random seeds |

## Research Outputs

The experiment pipeline is designed to produce:

- Training and validation curves
- Accuracy/loss comparisons
- Per-class evaluation metrics
- Confusion matrices
- Baseline vs. TET comparison tables
- Prediction files
- Checkpoints
- Ablation results
- Multi-seed summaries

Numerical claims should be taken from generated files in `results/` after the corresponding experiment has been executed.

## Reproducibility

The project uses a fixed default seed of `42` and provides deterministic seed initialization for Python, NumPy, and PyTorch. Optional seeds include `42`, `123`, and `2026`.

GPU execution can still exhibit implementation- or hardware-dependent nondeterminism. Where possible, report multiple seeds and the execution environment.

## Computational Notes

The implementation is intended primarily for GPU execution, particularly Kaggle environments with Tesla T4/P100-class accelerators. Memory-conscious features include small micro-batches, gradient accumulation, optional mixed precision, CUDA memory configuration, and explicit garbage collection.

## Citation

```bibtex
@inproceedings{deng2022tet,
  title     = {Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting},
  author    = {Deng, Shikuang and Li, Yuhang and Zhang, Shanghang and Gu, Shi},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

## Project Status

**Research / Experimental**

## Author

**Md. Bakibillah Rahat**  
Computer Science & Engineering  
GitHub: [@bakibillahrahat](https://github.com/bakibillahrahat)

## License

See the repository for the applicable license and dataset-specific terms.
