# 1D Bratu Problem

This directory contains the implementation of the 1D Bratu problem within the PINN CLI framework. The 1D Bratu equation is a classic nonlinear boundary value problem often used as a benchmark for numerical methods and physics-informed neural networks.

## Problem Description

The 1D Bratu problem is defined by the following nonlinear ordinary differential equation (ODE):

$$ \frac{d^2u}{dx^2} + \lambda e^u = 0, \quad x \in [0, 1] $$

Subject to the Dirichlet boundary conditions:
$$ u(0) = 0, \quad u(1) = 0 $$

Here, $\lambda$ (lambda) is a constant parameter. Depending on the value of $\lambda$, the equation can have zero, one, or two solutions. This bifurcation behavior makes it a compelling test case for neural network-based PDE solvers, as explored in literature by Karniadakis et al. (e.g., studying multi-modal solutions or capturing specific branches).

## Implementation Details

This module conforms to the `BaseProblemAPI` required by the training framework:

*   **`api.py`**: The entry point for the framework. It defines the `ProblemAPI` class, which handles data generation, loss computation, model initialization, and exposing visualization routines.
*   **`problem.py`**: Contains the core logic for the physics loss (calculating the PDE residuals using PyTorch's `autograd`) and boundary condition enforcement.
*   **`models.py`**: Defines the neural network architecture specific to this problem. Usually, a standard Multi-Layer Perceptron (MLP) or variations like Multi-Head Neural Networks (MHNN) if exploring diverse solutions.
*   **`utils.py`**: Helper functions for exact solution generation (if applicable) or specialized mathematical operations.

## Running the 1D Bratu Problem

To train a model on the 1D Bratu problem, you need to use a configuration file that specifies this module.

```bash
# From the project root
python train.py --config configs/generic.yaml
```
*(Ensure that `problem_name: 1d_bratu` is set in your configuration file)*

## Visualizing Results

After training, you can visualize the model's predictions against the true solution (or plot the PDE residuals) using the built-in visualizer:

```bash
python visualize.py --config configs/generic.yaml
```
