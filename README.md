# Physics-Informed Neural Networks (PINNs) CLI Framework

![License](Polhttps://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)

A modular, extensible Command Line Interface (CLI) framework for training and visualizing Physics-Informed Neural Networks (PINNs). This tool simplifies the process of defining Partial Differential Equation (PDE) problems, managing configurations, orchestrating the training loop, and visualizing the results.

This framework is built for both research exploration and robust implementation, providing a structured approach to solving complex physics problems using neural networks.

## Key Features

*   **Modular Problem API:** Easily add new PDEs by implementing a standard `BaseProblemAPI`.
*   **Automated Training Loops:** Standardized, robust training pipelines built on PyTorch.
*   **Generalized Visualization:** Plotting tools abstracted to work across different problem domains.
*   **Configuration Management:** YAML-based configuration for easy hyperparameter tuning and reproducibility.

## Implemented Problems

The framework currently includes reference implementations for well-known problems in the PINN literature:

### 1. 1D Bratu Problem (`1d_bratu`)
A classic nonlinear boundary value problem. This implementation is based on the methodologies discussed in papers by Karniadakis et al.
*   **Equation:** $u_{xx} + \lambda e^u = 0$
*   [More Info / README](1d_bratu/README.md)

### 2. 2D Gray-Scott System (`gray_scott`)
A complex reaction-diffusion system that exhibits pattern formation. This implementation showcases techniques from the "Geometry Informed Neural Networks (GINN)" paper.
*   **Equations:**
    *   $u_t = D_u \Delta u - uv^2 + F(1-u)$
    *   $v_t = D_v \Delta v + uv^2 - (F+k)v$
*   [More Info / README](gray_scott/README.md)

#### Built-in visualization: Latent space interpolation:


https://github.com/user-attachments/assets/3c78b78c-f648-4df7-b893-ed92796d3920



## Getting Started

### Installation

This project uses Conda for dependency management. The primary dependencies include PyTorch and Plotly.

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/PINN-Thesis.git
    cd PINN-Thesis
    ```

2.  Create the Conda environment from the provided `requirements.yaml` file:
    ```bash
    conda env create -f requirements.yaml
    ```

3.  Activate the environment:
    ```bash
    conda activate ginn_o_311_gpu
    ```

### Basic Usage

The framework is driven by configuration files.

**To train a model:**
```bash
python train.py --config configs/generic.yaml
```

**To visualize results:**
(Assuming the problem API implements the plotting interface)
```bash
python visualize.py --config configs/generic.yaml
```

## Extensibility

This tool is designed to be easily extended. To create a new problem domain:

1.  Run the scaffold script:
    ```bash
    python add_problem.py my_new_pde
    ```
2.  Implement the required methods in the generated `my_new_pde/api.py`, inheriting from `core/BaseProblemAPI.py`.
3.  For detailed instructions, refer to the [API Guide](API_GUIDE.md) and the [Core Architecture Documentation](core/README.md).

## Project Structure

```text
PINN-Thesis/
├── 1d_bratu/        # 1D Bratu problem implementation
├── configs/         # YAML configuration files
├── core/            # The training engine and Base API
├── gray_scott/      # 2D Gray-Scott problem implementation
├── outputs/         # Saved models and plots
├── templates/       # Boilerplate code for new problems
├── train.py         # Main training entry point
├── visualize.py     # Main visualization entry point
├── add_problem.py   # Script to scaffold new PDEs
└── API_GUIDE.md     # Guide for implementing new problems
```
