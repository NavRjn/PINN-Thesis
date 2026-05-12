import torch
import torch.nn as nn

from core.BaseProblemAPI import BaseProblemAPI
from .models import init_model
from . import utils
from .plot import plot_loss_curves, plot_batch_fields_fd, plot_resolution_convergence, plot_latent_histogram
from .problem import ProblemDefinition


class API(BaseProblemAPI):

    metric_keys = ProblemDefinition.metric_keys

    def __init__(self):
        super().__init__()
        self.problem = None

    def setup_problem(self, config, device, logger):

        self.problem = ProblemDefinition(config, device)

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        physics_cfg = config.get("physics", {})


        # 1. Initialize Architecture
        arch = model_cfg.get("arch", "SIREN+POSENC")
        use_two_models = model_cfg.get("name", "DualNet") == "DualNet"
        nz, nx = model_cfg.get("nz", 1), model_cfg.get("nx", 2)

        model = self.problem.model or init_model(arch=arch, use_two_models=use_two_models, nz=nz, nx=nx).to(device)
        optimizer = self.problem.optimizer or torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-4))
        self.problem.bind_model(model)

        self._init_problem(model, optimizer, self.problem.loss_fn, self.problem.grid_sampler, logger, device)



    def post_process(self, history, run_dir):
        """Handles problem-specific plotting after training."""
        model = self.model
        device = self.device


        # 1. Retrieve the parameters from the model/config dynamically
        # Do not hardcode nz = 1 if your config used a different value!

        # We can infer nz by looking at the first layer of the model
        # Total input dim = enc_dim + nz
        # Since this can be tricky, it's safer to pass config to post_process
        # But for a quick fix, let's extract dimensions from the model architecture:
        first_layer = None
        if hasattr(model, 'models'):  # DualNet
            first_layer = model.models[0].network[0] if hasattr(model.models[0], 'network') else None
        else:
            first_layer = model.network[0]

        total_input_dim = first_layer.in_features

        # Assuming standard SIREN+POSENC with N_posenc=1 and nx=2 (dim=4)
        # nz = total_input_dim - 4
        nz = total_input_dim - 4
        sigma = 1.0  # Or pull from config
        bounds = [0, 1, 0, 1]

        # 2. Sample with the CORRECT nz dimension
        z_test = utils.sample_z(1, nz, sigma, device)

        # 3. Plotting
        plot_resolution_convergence(model, z_test, bounds, utils, device, run_dir)

        # Optional: plot loss curves if history has the right keys
        if 'obj' in history:
            plot_loss_curves(history, run_dir)
        else:
            self.logger.error(f"No objective loss found! in {history.keys()}")

        plot_latent_histogram(history['z'], run_dir)

        print(f"Gray-Scott post-processing complete in {run_dir}")



    @classmethod
    def post_process_visualize(cls, run_dir, config, device):
        from . import plot
        from . import models
        from . import utils as gs_utils
        from .plot import analyze_latent_space

        # 1. Correctly extract hyperparameters from the nested config
        # Use the 'model' sub-dictionary!
        m_cfg = config.get("model", {})
        t_cfg = config.get("training", {})
        p_cfg = config.get("physics", {})

        # This ensures build_model_from_config sees nz=3, not nz=1
        # If build_model_from_config is still looking at top-level config,
        # we pass m_cfg instead, or fix that function.
        model = models.build_model_from_config(m_cfg).to(device)

        ckpt_type = config.get("ckpt_type", "final")  # Top level from CLI
        ckpt_path = run_dir / "checkpoints" / f"{ckpt_type}.pt"

        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            print(f"Successfully loaded {ckpt_type} weights.")
        else:
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            return

        # 2. Extract values for plotting
        # Bounds and N are now inside 'physics' in the unified config
        bounds = p_cfg.get("bounds", [0, 1, 0, 1])
        nz = m_cfg.get("nz", 1)
        bz = t_cfg.get("bz", 10)  # Not strictly needed here, but good to have
        sigma = config.get("training", {}).get("sigma", 1.0)
        z_test = gs_utils.sample_z(bz, nz, sigma, device)

        plot.plot_resolution_convergence(model, z_test, bounds, gs_utils, device, run_dir)

        # Latent Analysis (The Plotly Animation)
        analyze_latent_space(model, run_dir, config)
