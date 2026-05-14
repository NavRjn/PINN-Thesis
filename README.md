# Physics-Informed Neural Networks (PINNs) CLI Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)

A modular, extensible Command Line Interface (CLI) framework for training and visualizing Physics-Informed Neural Networks (PINNs). This tool simplifies the process of defining Partial Differential Equation (PDE) problems, managing configurations, orchestrating training runs, and visualizing results.

This framework is designed for both research exploration and robust implementation, providing a structured and extensible system for solving complex physics problems using neural networks.

---

## Key Features

- **CLI-first workflow (Typer-based)**  
  Unified command-line interface for training, visualization, and problem generation.

- **Modular Problem API**  
  Add new PDEs by generating a self-contained problem module via CLI.

- **Automated Training Pipelines**  
  Standardized PyTorch training loop with reproducibility via YAML configs.

- **Unified Visualization System**  
  Cross-problem visualization through a shared API with fallback legacy support.

- **Scaffold Generator**  
  Quickly generate new PDE problem templates.

---

## CLI Usage (Recommended)

After installation (`pip install -e .`), use the `pinn` command:

### Train a model
```bash
pinn train gray_scott.yaml
````

### Visualize latest run

```bash
pinn viz gray_scott --last
```

### Visualize a specific run

```bash
pinn viz gray_scott 20240501_120000
```

### Create a new PDE problem

```bash
pinn add my_new_pde
```

---

## Implemented Problems

### 1. 1D Bratu Problem (`1d_bratu`)

A classical nonlinear boundary value problem.

* **Equation:**
  $u_{xx} + \lambda e^u = 0$

Based on standard PINN benchmark formulations (Karniadakis et al.).

More info: `1d_bratu/README.md`

---

### 2. 2D Gray-Scott System (`gray_scott`)

A reaction–diffusion system exhibiting complex pattern formation.

* **Equations:**

  * $u_t = D_u \Delta u - uv^2 + F(1-u)$
  * $v_t = D_v \Delta v + uv^2 - (F+k)v$

Demonstrates spatiotemporal pattern learning in PINNs.

More info: `gray_scott/README.md`

---

### Built-in visualization

Latent-space interpolation example:

[https://github.com/user-attachments/assets/3c78b78c-f648-4df7-b893-ed92796d3920](https://github.com/user-attachments/assets/3c78b78c-f648-4df7-b893-ed92796d3920)

---

## Installation

This project uses Conda for environment management and is installed as a Python package.

### 1. Clone repository

```bash
git clone https://github.com/yourusername/PINN-Thesis.git
cd PINN-Thesis
```

### 2. Create environment

```bash
conda env create -f requirements.yaml
conda activate ginn_o_311_gpu
```

### 3. Install framework (required for CLI)

```bash
pip install -e .
```

This enables the `pinn` command globally inside the environment.

---

## Project Structure

```text
PINN-Thesis/
├── pinn/                 # CLI package (Typer-based)
│   ├── cli.py
│   └── commands/
│
├── core/                 # Training + visualization engine
├── configs/              # YAML experiment configs
├── templates/            # Problem scaffolding system
├── outputs/              # Saved runs, checkpoints, results
│
├── gray_scott/          # Example PDE (optional)
├── 1d_bratu/            # Example PDE (optional)
│
├── pyproject.toml
├── README.md
└── API_GUIDE.md
```

---

## Extensibility

This framework is based on runtime-extensible PDE "plugins".

### Create a new problem

```bash
pinn add my_new_pde
```

This generates:

```text
my_new_pde/
├── api.py        # Required API for training & visualization
├── problem.py    # PDE definition and loss functions
├── config.yaml   # Default configuration
├── README.md
└── __init__.py
```

### How it works

* Problems are discovered at runtime (not installed via pip)
* Each problem exposes an `API` class in `api.py`
* The CLI dynamically imports the module when needed

---

## Design Philosophy

* Framework is stable
* Problems are dynamic plugins
* Everything is reproducible via YAML configs
* CLI is the single entry point

---
