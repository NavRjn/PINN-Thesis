import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

from core.BaseProblemAPI import BaseProblemAPI
from core.utils import ProblemSetup

from . import models
from . import utils as bratu_utils

class API(BaseProblemAPI):
    def __init__(self):
        super().__init__()
        self.metric_keys = ["u_mid", "model_wise_loss"]  # Add more keys as needed for logging

    def setup_problem(self, config, device, logger=None):
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
            return {"x_ensemble": x_ensemble}

        # 3. Loss Function Hook (PDE Residual)
        def loss_fn(model, batch):
            x = batch["x_ensemble"]
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

            model_wise_loss = torch.mean(residual ** 2, dim=(1, 2))

            # one of the metrics tracked in the paper
            u_mid = u[:, 50, :].detach().reshape(-1)  # Shape: [1000]

            # The total loss for backprop must still be a scalar
            total_loss = model_wise_loss.mean()

            metrics = {
                "obj": total_loss.item(),
                self.metric_keys[0]: u_mid.cpu().numpy().tolist(),
                self.metric_keys[1]: model_wise_loss.detach().cpu().numpy().tolist()
            }

            return total_loss, metrics

        self.problem = ProblemSetup(model, optimizer, loss_fn, grid_sampler, logger, device, lambda *x: x)
        return self.problem

    def post_process(self, model, history, run_dir, device):
        """
        Unified visualization for 1D Bratu ensemble results.
        Assumes history contains: 'obj', 'model_wise_loss', and 'u_mid'.
        """

        print("Starting post processing")
        run_dir = Path(run_dir)
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Extract iterations for the x-axis
        iters = sorted(history['obj'].keys())

        print("setup")
        # 1. Total Loss Graph
        plt.figure(figsize=(8, 5))
        total_losses = [history['obj'][i] for i in iters]
        plt.plot(iters, total_losses, color='black', linewidth=2)
        plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("Total Mean Squared Residual")
        plt.title("Bratu 1D: Total Training Loss")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(fig_dir / "loss_total.png", dpi=150)
        plt.close()

        # 2. Model-wise Loss Graph
        plt.figure(figsize=(8, 5))
        # Convert dict to array: [num_iters, ensemble_size]
        model_losses = np.array([history['model_wise_loss'][i] for i in iters])
        plt.plot(iters, model_losses, alpha=0.1, color='blue')  # Light lines for individual models
        plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("Individual Model Loss")
        plt.title("Bratu 1D: Ensemble Loss Convergence")
        plt.savefig(fig_dir / "loss_model_wise.png", dpi=150)
        plt.close()

        # 3. u_mid (0.5) Bifurcation Plot
        plt.figure(figsize=(8, 5))
        # Convert dict to array: [num_iters, ensemble_size]
        u_mids = np.array([history['u_mid'][i] for i in iters])
        plt.plot(iters, u_mids, alpha=0.05, color='red')  # Heavy transparency to see density
        plt.xlabel("Iteration")
        plt.ylabel("u(0.5)")
        plt.title("Bratu 1D: Evolution of Midpoint Predictions (Bifurcation)")
        plt.savefig(fig_dir / "u_mid_evolution.png", dpi=150)
        plt.close()

        # 4. Histogram of Solutions (Final State)
        last_iter = iters[-1]
        final_u_mids = np.array(history['u_mid'][last_iter])

        plt.figure(figsize=(8, 5))
        plt.hist(final_u_mids, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # Paper logic: Categorize by u(0.5) < 3
        u1_count = np.sum(final_u_mids < 3.0)
        u2_count = np.sum(final_u_mids >= 3.0)

        plt.axvline(3.0, color='red', linestyle='--', label=f'u1/u2 Threshold (3.0)')
        plt.xlabel("u(0.5)")
        plt.ylabel("Count")
        plt.title(f"Solution Distribution (u1: {u1_count} | u2: {u2_count})")
        plt.legend()
        plt.savefig(fig_dir / "solution_histogram.png", dpi=150)
        plt.close()

        print(f"[INFO] Bratu visualizations saved to {fig_dir}")

    @classmethod
    def post_process_visualize(cls, run_dir, config, device):
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
        # problem = self.setup_problem(config, device)  # To get a model instance
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