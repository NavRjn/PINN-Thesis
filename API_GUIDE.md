# GINN Problem API Guide

To integrate a new Physics-Informed Neural Network (PINN) problem into this CLI tool, you must create a directory for your problem containing an `api.py` file. This file acts as the bridge between your specific PDE logic and the centralized training/visualization engine.

## Core Requirements

Your `api.py` must implement two main functions:

### 1. `setup_problem(config, device)`
This function initializes the environment for training.

**Returns:**
- `model`: A `torch.nn.Module`.
- `optimizer`: A `torch.optim` optimizer.
- `loss_fn`: A callable `f(model, batch, config)` that returns `(total_loss, metrics_dict)`.
- `grid_sampler`: A callable that returns a `batch` (data needed for one training step).
- `logger`: (Optional) A logger instance or `None`.

### 2. `post_process_visualize(run_dir, config, device)`
This function is called by the `visualize.py` CLI. It should handle:
- Re-initializing the model architecture based on `config`.
- Loading weights from `run_dir / "checkpoints"`.
- Generating domain-specific plots (Heatmaps for 2D, Line plots for 1D, etc.).

---

## Pattern: The PDE Loss Hook

For PINNs, your `loss_fn` inside `setup_problem` should handle the Automatic Differentiation (AD) or Finite Difference (FD) logic.
```python
def loss_fn(model, batch, config):
    # 1. Forward pass
    u = model(batch)
    
    # 2. Compute Derivatives
    # Use torch.autograd.grad for AD
    u_x = torch.autograd.grad(u, batch, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # 3. Define PDE Residual
    residual = u_x - ... # Your PDE equation
    loss = torch.mean(residual**2)
    
    return loss, {"residual": loss.item()}