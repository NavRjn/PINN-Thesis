import torch
from pathlib import Path
import matplotlib.pyplot as plt
from core.BaseProblemAPI import BaseProblemAPI
from .problem import ProblemDefinition
from .plot import plot_loss_curves
from . import models
from .models import PNN

TEMPLATE_VERSION = "0.1"
PROBLEM_NAME = "{{ problem_name }}"

class API(BaseProblemAPI):


    model_map = {"PNN": models.PNN}

    def __init__(self):
        super().__init__()
        self.problem = None

    def setup_problem(self, config, device, logger=None):
        self.problem = ProblemDefinition(config, device)
        self.metric_keys = self.problem.metric_keys

        model = self._build_model(config).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=config.get("training", {}).get("lr", 1e-3))

        self._init_problem(model, optim, self.problem.loss_fn, self.problem.grid_sampler, logger, device)

    def post_process(self, history, run_dir):
        plot_loss_curves(history, run_dir)

    def _build_model(self, config):
        c = config.get("model", {})
        return self.model_map[c.get("name", "PNN")](
            units=c.get("units", 32),
            layers=c.get("layers", 2)
        )

    @classmethod
    def post_process_visualize(cls, run_dir, config, device):
        run_dir = Path(run_dir)
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Load model
        # -----------------------------
        ckpt_path = run_dir / "checkpoints" / "final.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            print("[WARN] No checkpoint found for visualization")
            return

        model_cfg = config.get("model", {})
        model = PNN(
            units=model_cfg.get("units", 32),
            layers=model_cfg.get("layers", 2)
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # -----------------------------
        # Data
        # -----------------------------
        n = config.get("training", {}).get("n_points", 128)
        x = torch.linspace(0, 1, n, device=device).view(-1, 1)

        with torch.no_grad():
            u = model(x).cpu().numpy()
            x_np = x.cpu().numpy()

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(6, 4))
        plt.plot(x_np, u, label="model u(x)")
        plt.plot(x_np, x_np, "--", label="target x")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("Identity Mapping Fit")
        plt.legend()
        plt.grid(True)

        plt.savefig(fig_dir / "identity_fit.png", dpi=150)
        plt.close()

        print(f"[INFO] Saved visualization to {fig_dir}")