import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import numpy as np


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