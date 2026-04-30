# BARZ: Barrier Zones for Adversarial Example Defense

This repository provides the PyTorch implementation for the Barrier Zone (BARZ) defense system, as described in the paper: [Barrier Zones for Adversarial Example Defense](https://ieeexplore.ieee.org/document/9663375), a wrapper on top of it to use it with various ensemble configurations.

BARZ is an ensemble-based defense that utilizes majority voting combined with a rejection threshold. If the models do not reach a specified level of agreement, the input is rejected (assigned to a "noise" class), creating a "Barrier Zone" against adversarial perturbations.

## Project Structure

- `attacks/`: Implementations of various adversarial attacks (ADBA, RayS, Square).
- `BarrierZoneTrainer/`: Core logic for the BARZ defense, including the voting mechanism (`BarrierZoneDefense.py`), model constructors, and training methods.
- `shared/`: Utility functions and global constants.
- `run_barz_*.py`: Execution scripts for running evaluations under different attack scenarios.
- `barz_configs.py`: Configuration manager for model ensembles.
- `barz_utils.py`: Helper functions for system setup and data loading.

## Installation & Setup

### Prerequisites

- PyTorch (>= 1.7.1)
- Torchvision (>= 0.8.2)
- NumPy (>= 1.19.2)

### Configuring Checkpoints and Datasets

Before running evaluations, ensure that your model checkpoints and datasets are correctly configured in `shared/constants.py`:

1.  **Checkpoints:** Update the `CHECKPOINTS` dictionary with the absolute paths to your `.th` or `.pth` model files.
2.  **Datasets:** Update the `VAL_DATASETS` and `TRAIN_DATASETS` dictionaries with the paths to your `.pth` data files.

## Running Evaluations

The repository includes several scripts to evaluate the BARZ defense against different black-box and adaptive attacks.

### 1. ADBA Attack
```bash
python run_barz_adba.py
```
This script evaluates the defense against the Adaptive Black-Box Attack (ADBA) across multiple epsilon values.

### 2. RayS Attack
```bash
python run_barz_rays.py
```
Evaluates robustness against the RayS attack.

### 3. Square Attack
```bash
python run_barz_square.py
```
Evaluates robustness against the Square Attack (L2 and Linf norms).

### Logging
Execution results and logs are typically saved in the `results/` directory (e.g., `results/blackbox_barz_adba_results_eps.log`).

## Configuration

You can customize the ensemble of models used by BARZ by modifying the `config_name` variable in the `main()` function of any `run_barz_*.py` script.

Available configurations (defined in `barz_configs.py`):
- `"base"`: Uses ResNet20, CaiT, and VGG16.
- `"with_svm"`: Includes an SVM-based voter.
- `"ramp_ce_ensemble"`, `"diffusion_linf_ensemble"`, etc.: Various configurations including robustly trained WideResNet models.

## Credit

The original BARZ research and implementation were developed at the University of Connecticut. 

- **Paper:** [BARZ: Barrier Zones for Adversarial Example Defense](https://ieeexplore.ieee.org/document/9663375)
- **Contact:** kaleel.mahmood@uconn.edu

This implementation builds upon several open-source works:
- ResNet implementation adapted from `torchvision`.
- Adaptive Black-Box attack based on the `CleverHans` implementation.
