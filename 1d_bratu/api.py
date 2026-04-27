import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from pathlib import Path
from core.utils import ProblemSetup

from . import models
from . import utils as bratu_utils


def setup_problem(config, device, logger=None):
    # 1. Model Initialization
    n = config["model"].get("ensemble_size", 100)
    model_type = config["model"].get("name", "PNN")
    units = config["model"].get("units", 50)
    std = config["model"].get("std", 1.0)
    factor = config["model"].get("factor", 1.0)

    if model_type == "PNN":
        model = models.PNN(units=units, n=n, std=std, factor=factor).to(device)
    elif model_type == "PNN2":
        model = models.PNN2(units=units, n=n, R=std).to(device)
    elif model_type == "MHNN":
        model = models.MHNN(units=units, n=n, std=std, factor=factor).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-3))

    # 2. Grid Sampler Hook
    def grid_sampler():
        # Bratu 1D typically uses a fixed grid of 100 points
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        # Tile for ensemble: shape [n, 100, 1]
        x_ensemble = x.repeat(n, 1, 1).to(device)
        x_ensemble.requires_grad_(True)
        return x_ensemble

    # 3. Loss Function Hook (PDE Residual)
    def loss_fn(model, x, config):
        u = model(x)

        # 1st Derivative
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # 2nd Derivative
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        lam = config["physics"].get("lambda", 1.0)
        # Residual: u_xx + lambda * exp(u) = 0
        residual = u_xx + lam * torch.exp(u)
        loss = torch.mean(residual ** 2)

        return loss, {"pde_residual": loss.item()}

    return ProblemSetup(model, optimizer, loss_fn, grid_sampler, logger, device, lambda *x: x)


def post_process_visualize(run_dir, config, device):
    """Unified visualization hook for Bratu 1D."""
    # Load Ground Truth
    data_path = (Path(__file__).parent / "data" / "data.mat").resolve()
    try:
        data = sio.loadmat(str(data_path))
        x_test, u1, u2 = data["x_test"], data["u1"], data["u2"]
    except FileNotFoundError:
        print(f"[WARN] Ground truth data not found at {data_path}")
        return

    ckpt_type = config.get("ckpt_type", "final")

    # Load Model
    # Since legacy Bratu used torch.save(model), we try that first
    ckpt_path = run_dir / "checkpoints" / "model_final.pt"
    if not ckpt_path.exists():
        # Fallback to state_dict if trained via unified
        ckpt_path = run_dir / "checkpoints" / f"{ckpt_type}.pt"
    if not ckpt_path.exists():
        # Fallback to state_dict if trained via unified
        print("falling back to best.pt")
        ckpt_path = run_dir / "checkpoints" / "best.pt"

    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        return


    # Fallback to state_dict (unified)
    problem = setup_problem(config, device)  # To get a model instance
    model = models.PNN(units=config["model"]["units"], n=config["model"]["ensemble_size"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()

    # Inference
    x_tensor = torch.tensor(x_test[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pred = model(x_tensor).cpu().numpy()  # Shape [n, 100, 1]

    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(x_test, u1, "k-", label="u1 (True)")
    plt.plot(x_test, u2, "b-", label="u2 (True)")

    # Plot first few ensemble members
    n_plot = min(u_pred.shape[0], 20)
    for i in range(n_plot):
        plt.plot(x_test, u_pred[i, :, 0], "r--", alpha=0.3)

    plt.title(f"Bratu 1D Ensemble Predictions (λ={config['physics']['lambda']})")
    plt.legend()

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "final_prediction.png")
    plt.close()

    print("Visualization complete. Saved final_prediction.png")