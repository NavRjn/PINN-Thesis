import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pathlib import Path
import re


# =============================================================================
# CONFIG (nested, overrideable)
# =============================================================================
CONFIG = {
    "device": "cuda:0",

    "paths": {
        "data": "./data/data.mat",
        "output_root": "./outputs/bratu",
        "run_dir": None,  # if None → automatically pick latest run
    },

    "visualization": {
        # Options: "latest", "all", "custom"
        "mode": "latest",

        # Used only if mode == "custom"
        "iterations": [0, 1000, 5000],
    },
}
# =============================================================================


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_run_dir(cfg):
    if cfg["paths"]["run_dir"] is not None:
        run_dir = Path(cfg["paths"]["run_dir"])
    else:
        base = Path(cfg["paths"]["output_root"])
        runs = sorted([p for p in base.iterdir() if p.is_dir()])
        if len(runs) == 0:
            raise RuntimeError(f"No runs found in {base}")
        run_dir = runs[-1]

    print(f"[INFO] Using run_dir: {run_dir}")
    return run_dir


def find_checkpoints(checkpoint_dir):
    files = list(Path(checkpoint_dir).glob("model_*"))

    def extract_iter(p):
        match = re.search(r"model_(\d+)", p.name)
        return int(match.group(1)) if match else -1

    files = sorted(files, key=extract_iter)
    return files


def select_checkpoints(cfg, ckpt_paths):
    mode = cfg["visualization"]["mode"]

    if len(ckpt_paths) == 0:
        print("[WARN] No checkpoints found.")
        return []

    if mode == "latest":
        return [ckpt_paths[-1]]

    elif mode == "all":
        return ckpt_paths

    elif mode == "custom":
        target_iters = set(cfg["visualization"]["iterations"])
        selected = []

        for p in ckpt_paths:
            iter_num = int(p.name.split("_")[1])
            if iter_num in target_iters:
                selected.append(p)

        return selected

    else:
        raise ValueError(f"Unknown visualization mode: {mode}")


def load_data(cfg):
    data_path = (Path("1d_bratu") / Path(cfg["paths"]["data"])).resolve()
    print(f"[INFO] Using data file: {data_path}")

    data = sio.loadmat(str(data_path))
    x_test = data["x_test"]
    u1 = data["u1"]
    u2 = data["u2"]

    return x_test, u1, u2


def plot_prediction(x, u_pred, u1, u2, save_path, title=""):
    plt.figure()

    # Ground truth
    plt.plot(x, u1, "k-", label="u1 (true)")
    plt.plot(x, u2, "b-", label="u2 (true)")

    # Predictions
    for j in range(u_pred.shape[0]):
        plt.plot(x, u_pred[j, ...], "--", alpha=0.5)

    plt.ylim([-0.5, 4.5])
    plt.title(title)
    plt.legend()

    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(config=None, run_dir=None):
    cfg = CONFIG.copy()
    if config is not None:
        cfg = deep_update(cfg, config)

    device = torch.device(cfg["device"])

    # Resolve run directory
    run_dir = run_dir if run_dir is not None else get_run_dir(cfg)
    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    # Load data
    x_test, u1, u2 = load_data(cfg)

    x_tensor = torch.tensor(
        x_test[None, ...],
        dtype=torch.float32,
        requires_grad=True,
    ).to(device)

    # Find + select checkpoints
    ckpt_paths = find_checkpoints(checkpoint_dir)
    selected_ckpts = select_checkpoints(cfg, ckpt_paths)

    if len(selected_ckpts) == 0:
        print("[WARN] No checkpoints selected.")
        return

    # Loop through checkpoints
    for ckpt_path in selected_ckpts:
        try:
            model = torch.load(ckpt_path, map_location=device)
            model.eval()

            with torch.no_grad():
                u_pred = model.forward(x_tensor).cpu().numpy()

            iter_num = ckpt_path.name.split("_")[1]

            save_path = figure_dir / f"prediction_{iter_num}.png"

            plot_prediction(
                x_test,
                u_pred,
                u1,
                u2,
                save_path,
                title=f"Iteration {iter_num}",
            )

            print(f"[INFO] Saved plot for iter {iter_num}")

        except Exception as e:
            print(f"[WARN] Could not load {ckpt_path}: {e}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()