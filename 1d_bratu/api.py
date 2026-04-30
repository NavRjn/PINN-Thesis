import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import plot_latent_histogram
import plotly.graph_objects as go

from core.BaseProblemAPI import BaseProblemAPI

from . import models
from . import utils as bratu_utils

class API(BaseProblemAPI):

    model_map = {
        "PNN": models.PNN,
        "PNN2": models.PNN2,
        "MHNN": models.MHNN
    }

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

        if model_type not in self.model_map:
            raise ValueError(f"Unknown model_type: {model_type}. Available options: {list(self.model_map.keys())}")
        elif model_type == "PNN2":
             model = models.PNN2(units=units, n=n, R=std).to(device)
        else:
             model = self.model_map[model_type](units=units, n=n, std=std, factor=factor).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-3))

        # 2. Grid Sampler Hook
        def grid_sampler():
            # Bratu 1D typically uses a fixed grid of 100 points
            x = torch.linspace(0, 1, 100).reshape(-1, 1)
            # Tile for ensemble: shape [n, 100, 1]
            x_ensemble = x.repeat(n, 1, 1).to(device)
            x_ensemble.requires_grad_(True)

            if config['training'].get("sigma", None) is None:
                lam = torch.tensor(config["physics"].get("lambda", 1.0), device=device)
            else:
                # Sample lambda from a normal distribution for each ensemble member
                lam = torch.normal(
                    mean=config["physics"].get("lambda", 1.0),
                    std=config['training'].get("sigma", 1.0),
                    size=(1,),
                    device=device,
                )

            return {"x_ensemble": x_ensemble, "z": abs(lam)}

        # 3. Loss Function Hook (PDE Residual)
        def loss_fn(model, batch):
            x = batch["x_ensemble"]
            lam = batch["z"]
            u = model(x, lam)

            # 1st Derivative
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]

            # 2nd Derivative
            u_xx = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]

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

        self._init_problem(model, optimizer, loss_fn, grid_sampler, logger, device)

    def post_process(self, history, run_dir):
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


        # plot conditioning vector distribution
        if len(history['z']) > 0:
            plot_latent_histogram(history['z'], run_dir)

        print(f"[INFO] Bratu visualizations saved to {fig_dir}")


    @classmethod
    def post_process_visualize(cls, run_dir, config, device):
        """
        Generates an interactive Plotly HTML visualization with a lambda slider.
        """
        # 1. Load Data (Ground Truth for reference)
        data_dir = (Path(__file__).parent / "data").resolve()
        if (data_dir / "data.npz").exists():
            data = np.load(data_dir / "data.npz")
            x_test, u1, u2 = data["x_test"], data["u1"], data["u2"]
        elif (data_dir / "data.mat").exists():
            data = sio.loadmat(str(data_dir / "data.mat"))
            x_test, u1, u2 = data["x_test"].flatten(), data["u1"].flatten(), data["u2"].flatten()
        else:
            print(f"[WARN] Ground truth data not found at {data_dir}")
            return

        # 2. Load the Parametric Model
        ckpt_type = config.get("ckpt_type", "final")
        ckpt_path = run_dir / "checkpoints" / f"{ckpt_type}.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoints" / "best.pt"

        if not ckpt_path.exists():
            print(f"[ERROR] No checkpoint found at {ckpt_path}")
            return

        model_cfg = config["model"]
        # Initialize the model using the registered map
        model = API.model_map[model_cfg.get("name", "PNN")](
            units=model_cfg["units"],
            n=model_cfg["ensemble_size"]
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 3. Define the Lambda Range for the Slider
        # Defaulting to [0.5, 3.5] to cover the interesting bifurcation region
        lam_start = max(0.0, config.get("z_start_val", 0.0))
        lam_end = config.get("z_end_val", 2.0)
        num_steps = config.get("num_steps", 30)
        lambdas = np.linspace(lam_start, lam_end, num_steps)

        x_tensor = torch.tensor(x_test[None, ...], dtype=torch.float32).to(device)
        x_test = x_test.flatten()
        n_plot = config.get("n", 100)

        frames = []
        # Generate data for each slider step
        for l_val in tqdm(lambdas):
            lam_tensor = torch.tensor(l_val, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Use the new parametric forward(x, lam)
                u_pred = model(x_tensor, lam_tensor).cpu().numpy()  # [n, 100, 1]
                # print(u_pred.shape)

                # --- QUICK DEBUG START ---
                if abs(l_val - 1.0) < 0.01:
                    print(x_test.shape, u_pred[i, :, 0].shape)
                    plt.figure()
                    for i in range(n_plot): plt.plot(x_test, u_pred[i, :, 0], "r--", alpha=0.3)
                    plt.title(f"DEBUG lam={l_val:.2f} shape={u_pred.shape}")
                    plt.grid(True)
                    plt.savefig(run_dir / "figures" / "debug_static_lam1.png")
                    plt.close()
                # --- QUICK DEBUG END ---

            # Each frame contains multiple traces (one for each ensemble member)
            frame_data = []
            for i in range(n_plot):
                frame_data.append(go.Scatter(
                    x=x_test, y=u_pred[i, :, 0],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.2)', width=1, dash='dash'),
                    name=f"Model {i}" if i == 0 else None,
                    showlegend=(i == 0)
                ))

            frames.append(go.Frame(data=frame_data, name=f"lam_{l_val:.2f}"))


        # 4. Build Interactive Figure
        fig = go.Figure(
            data=frames[0].data,  # Initial plot state
            layout=go.Layout(
                title=f"Parametric Bratu 1D: Ensemble Predictions",
                xaxis=dict(title="t (Spatial Coordinate)", range=[0, 1]),
                yaxis=dict(title="u(t) (Temperature)", range=[-0.5, 5.0]),
                template="plotly_white",
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [{"label": "Play", "method": "animate", "args": [None]}]
                }],
                sliders=[{
                    "currentvalue": {"prefix": "λ (Heat Generation): "},
                    "steps": [
                        {
                            "method": "animate",
                            "label": f"{l:.2f}",
                            "args": [[f"lam_{l:.2f}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        } for l in lambdas
                    ]
                }]
            ),
            frames=frames
        )

        # Add ground truth static lines (optional, only if relevant to current lambda)
        fig.add_trace(
            go.Scatter(x=x_test, y=u1, mode='lines', line=dict(color='black'), name="u1 (Stable Ground Truth)"))
        fig.add_trace(
            go.Scatter(x=x_test, y=u2, mode='lines', line=dict(color='blue'), name="u2 (Unstable Ground Truth)"))

        # 5. Save Artifact
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        html_path = fig_dir / "parametric_prediction.html"
        fig.write_html(str(html_path), auto_open=True)

        print(f"Visualization complete. Saved interactive graph to {html_path}")