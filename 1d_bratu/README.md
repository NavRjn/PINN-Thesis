# 1D Bratu Problem - Directory Overview

## Directory Contents
- **main_final_training.py**: Main training script for the standard Bratu PINN experiment. Trains a deep ensemble and saves outputs and checkpoints.
- **main_final_different_lambs.py**: Trains PINNs for different values of the Bratu parameter λ, saving models for each case.
- **main_final_different_initializations.py**: Studies the effect of different random initializations on PINN solutions.
- **main_final_different_iterations.py**: Explores solution evolution over different training iterations.
- **main_final_mhnn.py**: Trains a multi-head PINN (MHNN) variant for the Bratu problem.
- **main_ablation.py**: Ablation study script, testing the effect of different model architectures and initializations.
- **models.py**: Defines core neural network architectures (PNN, MHNN, PNN2) for PINN experiments.
- **utils.py**: Utility functions for loss calculation, parameter flattening/unflattening, and helper routines.
- **fig_case1.py, fig_case2.py, fig_case3.py, fig_case4.py**: Visualization scripts for different λ cases, plotting reference and PINN solutions.
- **fig_ablation.py**: Plots results from ablation experiments, including initialization and architecture effects.
- **fig_initialization.py**: Visualizes the impact of different initializations on solution diversity.
- **fig_mhnn.py**: Plots results for the multi-head PINN experiment.
- **fig_training.py, fig_training_2.py**: Visualizes training progress and loss evolution.
- **checkpoints/**: Contains saved model checkpoints for different experiments and cases.
- **data/**: Contains input data (e.g., data.mat) for experiments.

## Core Functionality
- **Model Definition**: All experiments use variants of the PNN (Physics-Informed Neural Network) defined in models.py. MHNN and PNN2 provide alternative architectures.
- **Training Loop**: Each main_*.py script implements a training loop using Adam optimizer, autograd for PDE residuals, and ensemble learning.
- **Loss Calculation**: The core loss is the PDE residual (Bratu equation), calculated via autograd. Some scripts add regularization or prior terms.
- **Parameter Handling**: flatten/unflatten utilities allow flexible manipulation of model parameters for ensemble and initialization studies.
- **Visualization**: fig_*.py scripts plot reference solutions, PINN predictions, and ensemble diversity.

### Minimal Core Functions for MVP
- `models.PNN` (or `MHNN`): The main PINN model class.
- `utils.loss_function`: Computes the loss for training.
- Training loop (Adam optimizer, autograd for PDE residuals).
- Data loading from `data/data.mat`.
- Plotting with matplotlib for result visualization.

## File/Case Mapping
- **Case 1**: `main_final_training.py` trains PINNs for λ=3.5; `fig_case1.py` plots reference and PINN solutions for this case.
- **Case 2**: `main_final_different_lambs.py` trains for λ=2; `fig_case2.py` visualizes results.
- **Case 3**: `main_final_different_lambs.py` trains for λ=0.5; `fig_case3.py` visualizes results.
- **Case 4**: `main_final_different_lambs.py` trains for λ=3.5 (alternate); `fig_case4.py` visualizes results.
- **Ablation**: `main_ablation.py` and `fig_ablation.py` explore model/initialization effects.
- **Initialization Study**: `main_final_different_initializations.py` and `fig_initialization.py`.
- **Iteration Study**: `main_final_different_iterations.py` and training visualization scripts.
- **MHNN Variant**: `main_final_mhnn.py` and `fig_mhnn.py`.

## models.py & utils.py Function/Object Descriptions
### models.py
- **PNN**: Main PINN model, ensemble-based, with custom initialization and forward pass for Bratu PDE.
- **MHNN**: Multi-head PINN variant, parameter sharing for improved diversity.
- **PNN2**: PINN variant with uniform initialization, used in ablation/initialization studies.

### utils.py
- **loss_function**: Computes negative log-likelihood and prior for Bayesian PINN training.
- **flatten**: Flattens model parameters for ensemble manipulation.
- **unflatten**: Restores flattened parameters to original shapes.

# 1D Bratu Problem - Experiment Log Overview
See exp_logs.md for a summary of differences and versioning across experiments.
