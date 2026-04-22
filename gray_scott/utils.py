import torch
import numpy as np
import torch.nn.functional as F
from torch.func import vmap, jacrev, jacfwd, functional_call
import logging
import sys
from pathlib import Path


# =======================
# Mathematical Operators
# =======================

def get_ad_operators(model):
    """Encapsulates AD operators so they don't rely on global variables."""

    def f(params, x, z):
        return functional_call(model, params, (x, z))

    f_x = jacrev(f, argnums=1)
    vf_x = vmap(f_x, in_dims=(None, 0, 0), out_dims=(0))

    f_z = jacrev(f, argnums=2)
    vf_z = vmap(f_z, in_dims=(None, 0, 0), out_dims=(0))

    f_xx = jacfwd(f_x, argnums=1)
    vf_xx = vmap(f_xx, in_dims=(None, 0, 0), out_dims=(0))

    laplacian = lambda params, x, z: f_xx(params, x, z).diagonal(dim1=-1, dim2=-2).sum(-1)
    v_laplacian = vmap(laplacian, in_dims=(None, 0, 0), out_dims=(0))

    return v_laplacian, vf_x, vf_z


def laplacian_conv(input_tensor, dx0, dx1, device):
    laplacian_filter = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    input_tensor_padded = F.pad(input_tensor, (1, 1, 1, 1), mode='circular').unsqueeze(1)
    laplacian_result = F.conv2d(input_tensor_padded, laplacian_filter)
    laplacian_result /= (dx0 * dx1)
    return laplacian_result.squeeze(1)


def gradient_conv(input_tensor, dx0, dx1, device):
    sobel_x_filter = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y_filter = sobel_x_filter.transpose(-1, -2)

    input_tensor_padded = F.pad(input_tensor, (1, 1, 1, 1), mode='circular').unsqueeze(1)
    gradient_x = F.conv2d(input_tensor_padded, sobel_x_filter)
    gradient_y = F.conv2d(input_tensor_padded, sobel_y_filter)

    gradient_x /= (2 * dx0 * 4)
    gradient_y /= (2 * dx1 * 4)
    return gradient_x.squeeze(1), gradient_y.squeeze(1)


# =======================
# Domain & Grids
# =======================

def get_domain_grid(bounds, N0, N1, device):
    x0s, dx0 = np.linspace(bounds[0], bounds[1], N0, endpoint=False, retstep=True)
    x1s, dx1 = np.linspace(bounds[2], bounds[3], N1, endpoint=False, retstep=True)
    X0, X1 = np.meshgrid(x0s, x1s)
    xs = np.vstack([X0.ravel(), X1.ravel()]).T
    xs = torch.tensor(xs).float().to(device)
    return xs, X0, X1, dx0, dx1


def tensor_product_xz(x, z, device):
    z_tp = z.repeat_interleave(len(x), 0).to(device)
    x_tp = x.repeat(len(z), 1)
    return x_tp, z_tp


def sample_z(bz, nz, sigma, device):
    return torch.randn(bz, nz, device=device) * sigma


# =======================
# Losses & Metrics
# =======================

def loss_diversity(y):
    pairwise_dist = (y.unsqueeze(0) - y.unsqueeze(1)).norm(dim=-1).mean(-1)
    distances = pairwise_dist.masked_fill(torch.eye(pairwise_dist.size(0), dtype=torch.bool, device=y.device),
                                          float('inf'))
    closest_dists, _ = distances.min(dim=1)
    return -(closest_dists.sqrt()).mean().square()


def per_z_residual_metrics(res1_field, res2_field):
    res_per_z = (res1_field.square() + res2_field.square()).mean(dim=1)
    return {
        "mean": res_per_z.mean().item(),
        "std": res_per_z.std().item(),
        "min": res_per_z.min().item(),
        "max": res_per_z.max().item(),
        "per_z": res_per_z.detach().cpu().tolist()
    }


def latent_sensitivity_metric(params, vf_x, vf_z, x_tp, z_tp):
    Jz = vf_z(params, x_tp, z_tp)
    Jx = vf_x(params, x_tp, z_tp)
    latent = Jz.square().sum(dim=(1, 2)).mean().sqrt()
    spatial = Jx.square().sum(dim=(1, 2)).mean().sqrt()
    return {"absolute": latent.item(), "normalized": (latent / (spatial + 1e-8)).item()}


def dominant_wavenumber(field):
    fft = torch.fft.fft2(field)
    power = torch.abs(fft) ** 2
    power[0, 0] = 0.0
    idx = torch.argmax(power)
    kx, ky = torch.unravel_index(idx, power.shape)
    return torch.sqrt((kx - field.shape[0] // 2) ** 2 + (ky - field.shape[1] // 2) ** 2).item()


def spectral_metrics(Y0, Y1):
    ks = []
    for zi in range(Y0.shape[0]):
        k0 = dominant_wavenumber(Y0[zi])
        k1 = dominant_wavenumber(Y1[zi])
        ks.append(0.5 * (k0 + k1))
    return {"mean_k": float(np.mean(ks)), "std_k": float(np.std(ks)), "k_per_z": ks}


def get_device(prefer_gpu=True):
    """Returns the available device (GPU or CPU)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =======================
# Logging
# =======================

def get_logger(name="gray_scott_cli"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if get_logger is called multiple times
    if not logger.handlers:
        # Determine the path to the CLI-tool directory
        cli_tool_dir = Path(__file__).parent.parent
        log_file_path = cli_tool_dir / "run.log"

        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler (for console output)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO) # Default to INFO for console
        stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # Custom exception hook to log uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
            sys.__excepthook__(exc_type, exc_value, exc_traceback) # Also call the default handler

        sys.excepthook = handle_exception

    return logger
