import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import numpy as np

import plotly.graph_objects as go
import numpy as np
import torch
from . import utils


def residual_at_z_fd(model, x, z, config, N0, N1, dx0, dx1, device):
    """Computes mean squared PDE residual at a fixed latent z."""
    # Note: Use config.get("physics", {}) to match unified config structure
    phys_cfg = config.get("physics", {})
    Fr = phys_cfg.get("Fr", 0.028)
    Kr = phys_cfg.get("Kr", 0.057)
    D1 = phys_cfg.get("D1", 0.1) / (N0 * N1)
    D2 = phys_cfg.get("D2", 0.05) / (N0 * N1)

    with torch.no_grad():
        z_tp = z.unsqueeze(0).repeat(x.shape[0], 1)
        ys = model(x, z_tp).reshape(1, x.shape[0], 2)

        y0 = ys[..., 0].reshape(1, N0, N1)
        y1 = ys[..., 1].reshape(1, N0, N1)

        y0_lap = utils.laplacian_conv(y0, dx0, dx1, device).reshape(-1)
        y1_lap = utils.laplacian_conv(y1, dx0, dx1, device).reshape(-1)

        y0 = y0.reshape(-1)
        y1 = y1.reshape(-1)

        res1 = -y0 * y1 * y1 + Fr * (1 - y0) + D1 * y0_lap
        res2 = y0 * y1 * y1 - (Fr + Kr) * y1 + D2 * y1_lap

        loss = (res1.square() + res2.square()).mean()

    return loss.item()


def analyze_latent_space(model, run_dir, config):
    """Generates Plotly interpolation html and residual spectrum plot."""
    device = next(model.parameters()).device
    m_cfg = config.get("model", {})
    p_cfg = config.get("physics", {})

    # Visualization params from top-level CLI/config overrides
    num_steps = config.get("num_steps", 50)
    z_start_val = config.get("z_start_val", -10.0)
    z_end_val = config.get("z_end_val", 10.0)

    bounds = p_cfg.get("bounds", [0, 1, 0, 1])
    N = p_cfg.get("grid_N", 64)
    x, X0, X1, dx0, dx1 = utils.get_domain_grid(bounds, N, N, device)

    nz = m_cfg.get("nz", 1)
    z_start = torch.ones(nz, device=device) * z_start_val
    z_end = torch.ones(nz, device=device) * z_end_val
    alphas = torch.linspace(0, 1, num_steps, device=device)

    frames, losses = [], []
    print("Generating latent interpolation frames...")
    for i, alpha in enumerate(alphas):
        z = (1 - alpha) * z_start + alpha * z_end
        x_tp, z_tp = utils.tensor_product_xz(x, z.unsqueeze(0), device)

        with torch.no_grad():
            Y = model(x_tp, z_tp)[..., 0].reshape(X0.shape).cpu().numpy()
            loss = residual_at_z_fd(model, x, z, config, N, N, dx0, dx1, device)
            losses.append(loss)

        frames.append(
            go.Frame(
                data=[go.Heatmap(z=Y, colorscale="Viridis", zmin=0.0, zmax=1.0)],
                name=f"{i}",
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Residual: {loss:.2e}", x=0.02, y=0.98,
                        xref="paper", yref="paper", showarrow=False,
                        font=dict(size=14, color="white"), bgcolor="rgba(0,0,0,0.6)"
                    )]
                )
            )
        )
    z_start_str = np.array2string(z_start.cpu().numpy(), precision=2)
    z_end_str = np.array2string(z_end.cpu().numpy(), precision=2)

    # 1. Save Interpolation Animation
    fig_interp = go.Figure(data=frames[0].data, frames=frames)
    fig_interp.update_layout(
        title="Latent Space Interpolation", width=600, height=600,
        annotations=[
            dict(
                text=f"z_start = {z_start_str}<br>z_end = {z_end_str}",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ],
        sliders=[{
            "steps": [
                {"method": "animate", "args": [[f"{i}"], {"mode": "immediate", "frame": {"duration": 0}}],
                 "label": f"{alphas[i]:.2f}"}
                for i in range(len(frames))
            ],
            "currentvalue": {"prefix": "α = "}
        }]
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    fig_interp.write_html(figures_dir / "latent_interpolation.html", auto_open=True)

    # 2. Save Residual Spectrum Plot
    fig_spec = go.Figure()
    fig_spec.add_trace(
        go.Scatter(x=alphas.cpu().numpy(), y=losses, mode="lines+markers", line=dict(width=2), marker=dict(size=6),
                   name="Residual"))
    fig_spec.update_layout(
        title="Latent Interpolation Residual Spectrum",
        xaxis_title="α  (z = (1−α) z_start + α z_end)",
        yaxis_title="Mean squared PDE residual",
        yaxis_type="log", width=600, height=400,
        template="plotly_white",
        annotations=[
            dict(
                text=f"z_start = {z_start_str}<br>z_end = {z_end_str}",
                x=0.5, y=-0.3,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=11),
                align="center"
            )
        ]
    )

    fig_spec.write_image(figures_dir / "residual_spectrum.png", scale=2)
    print(f"✅ Saved plotting artifacts to {figures_dir}")


def save_figure(fig, name, run_dir, dpi=150):
    path = run_dir / "figures" / f"{name}.png"

    # Ensure the figures directory exists before saving!
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(loss_over_iters, run_dir):
    # Safely cast keys/values to lists to prevent Matplotlib 'dict_values' iterable errors
    iters = list(loss_over_iters['obj'].keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, list(loss_over_iters['obj'].values()), label='residual')

    if 'grad' in loss_over_iters and loss_over_iters['grad']:
        ax.plot(iters, list(loss_over_iters['grad'].values()), label='gradient')

    if 'div' in loss_over_iters and loss_over_iters['div']:
        ax.plot(iters, list(loss_over_iters['div'].values()), label='diversity')

    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    save_figure(fig, "loss_curves", run_dir)


def format_ax(ax):
    ax.axis('scaled')
    ax.axis('off')


def plot_batch_fields_fd(model, z, bounds, N0, N1, utils, device, run_dir):
    x, X0, X1, dx0, dx1 = utils.get_domain_grid(bounds, N0, N1, device)
    x_tp, z_tp = utils.tensor_product_xz(x, z, device)
    bz = len(z)

    with torch.no_grad():
        # Removed squeeze(0) which behaves unpredictably if bz > 1
        ys = model(x_tp, z_tp).reshape(bz, len(x), 2)
        y0, y1 = ys[..., 0], ys[..., 1]
        Y0, Y1 = y0.reshape(bz, N0, N1), y1.reshape(bz, N0, N1)

        Y0_lap = utils.laplacian_conv(Y0, dx0, dx1, device)
        Y1_lap = utils.laplacian_conv(Y1, dx0, dx1, device)
        Y0_x0, Y0_x1 = utils.gradient_conv(Y0, dx0, dx1, device)
        Y1_x0, Y1_x1 = utils.gradient_conv(Y1, dx0, dx1, device)

    figscale = 2
    for iy, (Y_, Y_x0_, Y_x1_, Y_lap_) in enumerate(zip([Y0, Y1], [Y0_x0, Y1_x0], [Y0_x1, Y1_x1], [Y0_lap, Y1_lap])):
        fig, axs = plt.subplots(4, bz, figsize=(figscale * bz, figscale * 4))
        if bz == 1: axs = axs[:, None]  # Handle bz=1 case
        for iz, (Y, Y_x0, Y_x1, Y_lap) in enumerate(zip(Y_, Y_x0_, Y_x1_, Y_lap_)):
            axs[0, iz].imshow(Y.cpu().numpy(), origin='lower')
            format_ax(axs[0, iz])
            axs[1, iz].imshow(Y_x0.cpu().numpy(), origin='lower', extent=bounds)
            format_ax(axs[1, iz])
            axs[2, iz].imshow(Y_x1.cpu().numpy(), origin='lower', extent=bounds)
            format_ax(axs[2, iz])
            axs[3, iz].imshow(Y_lap.cpu().numpy(), origin='lower')
            format_ax(axs[3, iz])
        save_figure(fig, f"batch_fields_y{iy}", run_dir)


def plot_resolution_convergence(model, z, bounds, utils, device, run_dir):
    fig, axs = plt.subplots(2, 4, figsize=(18, 6))

    # CRITICAL FIX: If z is a batch (e.g. bz=10), evaluating the grid shapes below will throw
    # a PyTorch size mismatch error. We slice z to only plot convergence for the FIRST latent vector.
    z_single = z[0:1]

    for i, N in enumerate([64, 128, 256, 512]):
        x, X0, X1, _, _ = utils.get_domain_grid(bounds, N, N, device)
        x_tp, z_tp = utils.tensor_product_xz(x, z_single, device)

        with torch.no_grad():
            ys = model(x_tp, z_tp).reshape(1, len(x), 2).squeeze(0)

        Y0, Y1 = ys[..., 0].reshape(X0.shape), ys[..., 1].reshape(X0.shape)

        axs[0, i].imshow(Y0.cpu().numpy(), origin='lower')
        format_ax(axs[0, i])
        axs[1, i].imshow(Y1.cpu().numpy(), origin='lower')
        format_ax(axs[1, i])

    save_figure(fig, "resolution_convergence", run_dir)


def plot_latent_histogram(z_history,  run_dir, bins=50):
    z_array = np.array(z_history)

    # Plot histogram
    plt.figure()
    plt.hist(z_array, bins=bins, density=True, alpha=0.7, label="Sampled z")

    # Overlay Gaussian for comparison
    mu, std = z_array.mean(), z_array.std()
    x = np.linspace(mu - 4 * std, mu + 4 * std, 200)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)

    plt.plot(x, gaussian, label=f"Fitted Gaussian (μ={mu:.2f}, σ={std:.2f})")
    plt.title("Distribution of sampled z (mean over batch)")
    plt.legend()
    plt.savefig(run_dir/ "figures" / "z_histogram.png")
    plt.close()