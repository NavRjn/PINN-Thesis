import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
from torch import nn
from tqdm import trange

from .models import init_model
from .utils import (
    get_ad_operators, laplacian_conv, gradient_conv, get_domain_grid,
    tensor_product_xz, sample_z, loss_diversity, per_z_residual_metrics,
    latent_sensitivity_metric, spectral_metrics, get_logger
)
from .plot import plot_loss_curves, plot_batch_fields_fd, plot_resolution_convergence, plot_latent_histogram


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config, run_dir):
    logger = get_logger()
    logger.info(f"Starting Gray-Scott training run in {run_dir}")
    logger.debug(f"Config: {config}")

    model_cfg = config["model"]
    physics_cfg = config["physics"]
    train_cfg = config["training"]

    device = get_device()
    logger.info(f"Using device: {device}")
    torch.manual_seed(train_cfg.get("seed", 0))

    # Architecture configuration
    arch = model_cfg["arch"]
    use_two_models = model_cfg.get("name", "DualNet") == "DualNet"
    nz, nx, ny = model_cfg["nz"], model_cfg["nx"], model_cfg["ny"]

    # Domain configuration
    bounds = physics_cfg["bounds"]
    N = physics_cfg["grid_N"]
    N0, N1 = N, N

    # Physics configuration (Correct Laplacian Scale)
    D1 = physics_cfg["D1"] / (N * N)
    D2 = physics_cfg["D2"] / (N * N)
    Fr = physics_cfg["Fr"]
    Kr = physics_cfg["Kr"]

    # Initialize Model & Operator Utils
    logger.info(f"Initializing model with architecture {arch}")
    model = init_model(arch=arch, use_two_models=use_two_models, nz=nz, nx=nx).to(device)
    params = dict(model.named_parameters())
    v_laplacian, vf_x, vf_z = get_ad_operators(model)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # Training config
    n_epochs = train_cfg["n"]
    bz = train_cfg["bz"]
    sigma = train_cfg["sigma"]
    method = train_cfg["method"]
    move_grid = train_cfg["move_grid"]
    use_softclip = train_cfg["use_softclip"]
    w_grad = train_cfg["w_grad"]
    metric_every = train_cfg["save_freq"]

    if use_softclip:
        softplus = nn.Softplus(beta=10)
        softclip = lambda x, max_: -softplus(-x + max_) + max_

    # Precompute Grid if FD
    if method == "FD":
        x, _, _, dx0, dx1 = get_domain_grid(bounds, N0, N1, device)
        logger.debug(f"Precomputed FD grid with bounds {bounds} and shape {N0}x{N1}")

    # Losses Dictionary setup
    loss_over_iters = {'obj': {}, 'grad': {}, 'div': {}, 'residual': {}, 'latent_sensitivity': {}, 'spectral': {}}
    best_residual = float("inf")

    # Base Z sampling
    z_fixed = sample_z(bz, nz, sigma, device)
    # Dynamic Z sampling function for each iteration (if needed)
    z = lambda bz_=bz: torch.randn(bz_, nz, device=device) * sigma

    z_history = []

    # -----------------------
    # Training Loop
    # -----------------------
    logger.info(f"Starting training loop for {n_epochs} epochs using method {method}")
    pbar = trange(n_epochs)
    for i in pbar:
        opt.zero_grad()
        z_ = z()
        z_history.extend(z_.detach().cpu().view(-1).tolist())

        # Grid Resampling
        if method == "AD":
            x = torch.rand(5000, nx, device=device)
            x_tp, z_tp = tensor_product_xz(x, z_, device)
        elif method == "FD":
            if move_grid:
                x_ = x + torch.rand(1, 1, device=device) * (1 / N)
                x_tp, z_tp = tensor_product_xz(x_, z_, device)
            else:
                x_tp, z_tp = tensor_product_xz(x, z_, device)

        # Forward Pass
        ys = model(x_tp, z_tp).reshape(bz, len(x), ny)
        y0, y1 = ys[..., 0], ys[..., 1]

        # Differentials
        if method == 'AD':
            ys_lap = v_laplacian(params, x_tp, z_tp).reshape(bz, len(x), ny)
            y0_lap, y1_lap = ys_lap[..., 0], ys_lap[..., 1]

            y_x = vf_x(params, x_tp, z_tp).reshape(bz, len(x), ny, nx)
            grad_norms = y_x[0].square().mean()

        elif method == 'FD':
            Y0, Y1 = y0.reshape(bz, N0, N1), y1.reshape(bz, N0, N1)

            Y0_lap = laplacian_conv(Y0, dx0, dx1, device)
            Y1_lap = laplacian_conv(Y1, dx0, dx1, device)
            y0_lap, y1_lap = Y0_lap.reshape(bz, -1), Y1_lap.reshape(bz, -1)

            Y0_x0, Y0_x1 = gradient_conv(Y0, dx0, dx1, device)
            Y1_x0, Y1_x1 = gradient_conv(Y1, dx0, dx1, device)

            y0_grad_norm = (Y0_x0.square() + Y0_x1.square()).mean()
            y1_grad_norm = (Y1_x0.square() + Y1_x1.square()).mean()
            grad_norms = (y0_grad_norm + y1_grad_norm) / 64

        # Objectives
        y011 = y0 * y1 * y1
        res1_field = -y011 + Fr * (1 - y0) + D1 * y0_lap
        res2_field = y011 - (Fr + Kr) * y1 + D2 * y1_lap
        loss_obj = (res1_field.square() + res2_field.square()).mean()

        # Magnitude Softclip
        if use_softclip:
            loss_grad = -softclip(grad_norms, 1)
        else:
            loss_grad = -torch.clip(grad_norms, max=1)

        # Diversity
        loss_div = torch.tensor([0.0],
                                device=device)  # diversity loss currently unweighted or implemented minimally as in original

        # Step
        loss = loss_obj + w_grad * loss_grad + 1e-7 * loss_div
        loss.backward()
        opt.step()

        # Logging iteration values
        pbar.set_description(f"Loss: {loss_obj.item():.2e} | Grad: {loss_grad.item():.2e}")
        loss_over_iters['obj'][i] = loss_obj.item()
        loss_over_iters['grad'][i] = -loss_grad.item()
        loss_over_iters['div'][i] = -loss_div.item()

        # Metrics & Checkpoints
        if i % metric_every == 0:
            logger.debug(f"Epoch {i}: Loss = {loss_obj.item():.4e}, Grad = {loss_grad.item():.4e}, z={z_.mean().item():.4f}")
            with torch.no_grad():
                loss_over_iters['residual'][i] = per_z_residual_metrics(res1_field, res2_field)
                loss_over_iters['latent_sensitivity'][i] = latent_sensitivity_metric(params, vf_x, vf_z, x_tp, z_tp)

                if method == "FD":
                    loss_over_iters['spectral'][i] = spectral_metrics(Y0, Y1)

                mean_residual = loss_obj.item()
                if mean_residual < best_residual:
                    best_residual = mean_residual
                    logger.debug(f"New best residual: {best_residual:.4e} at epoch {i}. Saving best.pt")
                    torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

    # -----------------------
    # Final Saves & Plots
    # -----------------------
    logger.info("Training complete. Saving final state and plots.")
    torch.save(model.state_dict(), run_dir / "checkpoints" / "final.pt")

    plot_latent_histogram(z_history, run_dir)

    with open(run_dir / "losses.json", "w") as f:
        json.dump(loss_over_iters, f)

    # Use plotting module
    from . import utils
    plot_loss_curves(loss_over_iters, run_dir)

    z_test = sample_z(1, nz, sigma, device) if bz > 1 else z_fixed
    if method == "FD":
        plot_batch_fields_fd(model, z_test, bounds, N0, N1, utils, device, run_dir)
    plot_resolution_convergence(model, z_test, bounds, utils, device, run_dir)

    logger.info(f"Done! Best residual: {best_residual:.4e}")
    print(f"\nDone! Best residual: {best_residual:.4e}")