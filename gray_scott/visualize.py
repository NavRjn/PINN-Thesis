import json
import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Import from the modular files we created earlier
from .models import build_model_from_config
from .utils import get_device, get_domain_grid, laplacian_conv, get_logger

logger = get_logger()

def load_config(run_dir: Path):
    """Dynamically load the config associated with this specific run."""
    config_path = run_dir / "config.yaml"
    if (run_dir / "config.yaml").exists():
        config_path = run_dir / "config.yaml"
    elif (run_dir / "params.json").exists():
        config_path = run_dir / "params.json"
    else:
        logger.error(f"No config file found in {run_dir}")
        raise FileNotFoundError(f"No config file found in {run_dir}")

    with open(config_path, "r") as f:
        if config_path.suffix == ".yaml":
            return yaml.safe_load(f)
        return json.load(f)


def residual_at_z_fd(model, x, z, config, N0, N1, dx0, dx1, device):
    """Computes mean squared PDE residual at a fixed latent z."""
    Fr = config.get("Fr", 0.028)
    Kr = config.get("Kr", 0.057)
    D1 = config.get("D1", 0.1) / (N0 * N1)
    D2 = config.get("D2", 0.05) / (N0 * N1)

    with torch.no_grad():
        z_tp = z.unsqueeze(0).repeat(x.shape[0], 1)
        ys = model(x, z_tp).reshape(1, x.shape[0], 2)

        y0 = ys[..., 0].reshape(1, N0, N1)
        y1 = ys[..., 1].reshape(1, N0, N1)

        y0_lap = laplacian_conv(y0, dx0, dx1, device).reshape(-1)
        y1_lap = laplacian_conv(y1, dx0, dx1, device).reshape(-1)

        y0 = y0.reshape(-1)
        y1 = y1.reshape(-1)

        res1 = -y0 * y1 * y1 + Fr * (1 - y0) + D1 * y0_lap
        res2 = y0 * y1 * y1 - (Fr + Kr) * y1 + D2 * y1_lap

        loss = (res1.square() + res2.square()).mean()

    return loss.item()


def analyze_latent_space(run_dir: Path, ckpt_type="final", z_start_val=-20.0, z_end_val=20.0, num_steps=50):
    device = get_device()
    config = load_config(run_dir)

    model = build_model_from_config(config).to(device)

    # Locate checkpoint (fallback logic for notebook vs CLI naming conventions)
    ckpt_path = run_dir / "checkpoints" / f"{ckpt_type}.pt"
    if not ckpt_path.exists():
        # Fallback to older notebook format if checking an old run
        exp_name = config.get("exp_name", "gray_scott")
        ckpt_path = run_dir / "checkpoints" / f"{exp_name}_{ckpt_type}.pt"

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    bounds = (0, 1, 0, 1)
    N = config.get("grid_N", 64)
    x, X0, X1, dx0, dx1 = get_domain_grid(bounds, N, N, device)

    nz = config.get("nz", 1)
    z_start = torch.ones(nz, device=device) * z_start_val
    z_end = torch.ones(nz, device=device) * z_end_val

    logger.debug(f"running plot with z_start={z_start_val}, z_end={z_end_val}, num_steps={num_steps}")
    logger.debug(f"latent space: {z_start} to {z_end}")

    alphas = torch.linspace(0, 1, num_steps, device=device)
    frames = []
    losses = []

    print("Generating latent interpolation frames...")
    for i, alpha in enumerate(alphas):
        z = (1 - alpha) * z_start + alpha * z_end
        z_tp = z.unsqueeze(0).repeat(len(x), 1)

        with torch.no_grad():
            Y = model(x, z_tp)[..., 0].reshape(X0.shape).cpu().numpy()
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


def main(config, run_dir):
    """
    Unified CLI entrypoint.
    `config` is passed as an empty dict from the CLI, so we load the
    actual config tied to the run directory inside the analysis function.
    """
    run_dir = Path(run_dir)
    print("Running Gray-Scott Latent Space Analysis...")

    num_steps = config.get("num_steps", 50)
    z_end_val = config.get("z_end_val", 10)
    z_start_val = config.get("z_start_val", -10)
    ckpt_type = config.get("ckpt_type", "final")

    # We use 50 steps here as the default from the notebook
    analyze_latent_space(run_dir, ckpt_type=ckpt_type, z_start_val=z_start_val, z_end_val=z_end_val, num_steps=num_steps)
