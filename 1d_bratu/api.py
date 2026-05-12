import torch
import numpy as np
import scipy.io as sio

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tqdm import tqdm

from core.BaseProblemAPI import BaseProblemAPI

from .problem import ProblemDefinition
from .utils import plot_latent_histogram


class API(BaseProblemAPI):

    model_map = ProblemDefinition.model_map

    def __init__(self):
        super().__init__()

        self.problem = None

    # ==========================================================
    # FRAMEWORK ENTRYPOINT
    # ==========================================================
    def setup_problem(self, config, device, logger=None):

        self.problem = ProblemDefinition(config, device)

        self.metric_keys = self.problem.metric_keys

        self._init_problem(
            model=self.problem.model,
            optimizer=self.problem.optimizer,
            loss_fn=self.problem.loss_fn,
            grid_sampler=self.problem.grid_sampler,
            logger=logger,
            device=device
        )

    # ==========================================================
    # POST TRAINING
    # ==========================================================
    def post_process(self, history, run_dir):

        print("Starting post processing")

        run_dir = Path(run_dir)
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        iters = sorted(history["obj"].keys())

        # ------------------------------------------------------
        # Total Loss
        # ------------------------------------------------------

        plt.figure(figsize=(8, 5))

        total_losses = [history["obj"][i] for i in iters]

        plt.plot(
            iters,
            total_losses,
            color="black",
            linewidth=2
        )

        plt.yscale("log")

        plt.xlabel("Iteration")
        plt.ylabel("Total Mean Squared Residual")

        plt.title("Bratu 1D: Total Training Loss")

        plt.grid(True, which="both", ls="-", alpha=0.2)

        plt.savefig(fig_dir / "loss_total.png", dpi=150)

        plt.close()

        # ------------------------------------------------------
        # Model-wise Loss
        # ------------------------------------------------------

        plt.figure(figsize=(8, 5))

        model_losses = np.array([
            history["model_wise_loss"][i]
            for i in iters
        ])

        plt.plot(
            iters,
            model_losses,
            alpha=0.1,
            color="blue"
        )

        plt.yscale("log")

        plt.xlabel("Iteration")
        plt.ylabel("Individual Model Loss")

        plt.title("Bratu 1D: Ensemble Loss Convergence")

        plt.savefig(fig_dir / "loss_model_wise.png", dpi=150)

        plt.close()

        # ------------------------------------------------------
        # Bifurcation Plot
        # ------------------------------------------------------

        plt.figure(figsize=(8, 5))

        u_mids = np.array([
            history["u_mid"][i]
            for i in iters
        ])

        plt.plot(
            iters,
            u_mids,
            alpha=0.05,
            color="red"
        )

        plt.xlabel("Iteration")
        plt.ylabel("u(0.5)")

        plt.title("Bratu 1D: Evolution of Midpoint Predictions")

        plt.savefig(
            fig_dir / "u_mid_evolution.png",
            dpi=150
        )

        plt.close()

        # ------------------------------------------------------
        # Final Histogram
        # ------------------------------------------------------

        last_iter = iters[-1]

        final_u_mids = np.array(history["u_mid"][last_iter])

        plt.figure(figsize=(8, 5))

        plt.hist(
            final_u_mids,
            bins=50,
            color="skyblue",
            edgecolor="black",
            alpha=0.7
        )

        u1_count = np.sum(final_u_mids < 3.0)
        u2_count = np.sum(final_u_mids >= 3.0)

        plt.axvline(
            3.0,
            color="red",
            linestyle="--"
        )

        plt.title(
            f"Solution Distribution "
            f"(u1: {u1_count} | u2: {u2_count})"
        )

        plt.savefig(
            fig_dir / "solution_histogram.png",
            dpi=150
        )

        plt.close()

        if len(history["z"]) > 0:
            plot_latent_histogram(history["z"], run_dir)

        print(f"[INFO] Saved visualizations to {fig_dir}")

    # ==========================================================
    # INTERACTIVE VISUALIZATION
    # ==========================================================


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