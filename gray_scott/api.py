import torch
import torch.nn as nn
from .models import init_model
from . import utils
from .plot import plot_loss_curves, plot_batch_fields_fd, plot_resolution_convergence, plot_latent_histogram
import json
from . import plot
from . import models
from . import utils as gs_utils


def setup_problem(config, device):
    logger = utils.get_logger()

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    physics_cfg = config.get("physics", {})



    # 1. Initialize Architecture
    arch = model_cfg.get("arch", "SIREN+POSENC")
    use_two_models = model_cfg.get("name", "DualNet") == "DualNet"
    nz, nx = model_cfg.get("nz", 1), model_cfg.get("nx", 2)

    model = init_model(arch=arch, use_two_models=use_two_models, nz=nz, nx=nx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-4))

    # 2. Setup PDE Operators
    # AD operators require model params for functional call
    params = dict(model.named_parameters())
    v_laplacian, vf_x, vf_z = utils.get_ad_operators(model)

    # 3. Domain/Grid Sampling Hook
    method = train_cfg.get("method", "FD")
    bounds = physics_cfg.get("bounds", [0, 1, 0, 1])
    N = physics_cfg.get("grid_N", 64)
    bz = train_cfg.get("bz", 32)
    sigma = train_cfg.get("sigma", 1.0)

    # Precompute base grid for FD
    x_fd, _, _, dx0, dx1 = utils.get_domain_grid(bounds, N, N, device)

    def grid_sampler():
        """Generates the x and z tensors for the current iteration."""
        z = utils.sample_z(bz, nz, sigma, device)

        if method == "AD":
            x = torch.rand(5000, nx, device=device)  # Random points for AD
        else:  # FD
            if train_cfg.get("move_grid", False):
                x = x_fd + torch.rand(1, 1, device=device) * (1 / N)
            else:
                x = x_fd

        x_tp, z_tp = utils.tensor_product_xz(x, z, device)
        return {"x_tp": x_tp, "z_tp": z_tp, "z": z, "x_base": x}

    # 4. Loss Function Hook (The PDE)
    softplus = nn.Softplus(beta=10)

    def loss_fn(model, batch):
        x_tp, z_tp = batch["x_tp"], batch["z_tp"]
        ny = model_cfg.get("ny", 2)

        # Forward Pass
        ys = model(x_tp, z_tp).reshape(bz, -1, ny)
        y0, y1 = ys[..., 0], ys[..., 1]

        # Differentials
        if method == 'AD':
            ys_lap = v_laplacian(params, x_tp, z_tp).reshape(bz, -1, ny)
            y0_lap, y1_lap = ys_lap[..., 0], ys_lap[..., 1]
            y_x = vf_x(params, x_tp, z_tp).reshape(bz, -1, ny, nx)
            grad_norms = y_x[0].square().mean()
        else:  # FD
            Y0, Y1 = y0.reshape(bz, N, N), y1.reshape(bz, N, N)
            y0_lap = utils.laplacian_conv(Y0, dx0, dx1, device).reshape(bz, -1)
            y1_lap = utils.laplacian_conv(Y1, dx0, dx1, device).reshape(bz, -1)

            Y0_x0, Y0_x1 = utils.gradient_conv(Y0, dx0, dx1, device)
            y0_grad_norm = (Y0_x0.square() + Y0_x1.square()).mean()
            grad_norms = y0_grad_norm / 32  # Approximation matching legacy scaling

        # Physics Parameters
        D1, D2 = physics_cfg["D1"] / (N ** 2), physics_cfg["D2"] / (N ** 2)
        Fr, Kr = physics_cfg["Fr"], physics_cfg["Kr"]

        # PDE Residuals
        y011 = y0 * y1 * y1
        res1 = -y011 + Fr * (1 - y0) + D1 * y0_lap
        res2 = y011 - (Fr + Kr) * y1 + D2 * y1_lap

        loss_obj = (res1.square() + res2.square()).mean()

        # Regularization (Gradient Maximization)
        if train_cfg.get("use_softclip", True):
            loss_grad = -(-softplus(-grad_norms + 1) + 1)
        else:
            loss_grad = -torch.clip(grad_norms, max=1)

        total_loss = loss_obj + train_cfg.get("w_grad", 1e-4) * loss_grad

        return total_loss, {"obj": loss_obj.item(), "grad": -loss_grad.item()}

    return model, optimizer, loss_fn, grid_sampler, logger


def post_process(model, history, run_dir, device):
    """Handles problem-specific plotting after training."""

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

    print(f"Gray-Scott post-processing complete in {run_dir}")


def post_process_visualize(run_dir, config, device):
    from . import plot
    from . import models
    from . import utils as gs_utils
    from .visualize import analyze_latent_space

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
    analyze_latent_space(
        model,
        run_dir,
        ckpt_type=config.get('ckpt_type', 'final'),
        z_start_val=config.get('z_start_val', -10.0),
        z_end_val=config.get('z_end_val', 10.0),
        num_steps=config.get('num_steps', 50)
    )
