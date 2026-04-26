import os
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import json

from . import models
from . import utils

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "reproducibility": {
        "seed_np": 77663,
        "seed_torch": 22331,
    },

    "device": "cuda:0",

    "paths": {
        "data": "./data/data.mat",
        "checkpoint_dir": "./checkpoints_unified",
        "output_dir": "./outputs/unified",
        "output_prefix": "run",
    },

    "model": {
        "name": "PNN",
        "units": 50,
        "std": 1.0,
        "R": None,
        "factor": 1,
        "ensemble_size": 1000,
    },

    "physics": {
        "lambda": 1.0,
    },

    "training": {
        "lr": 1e-3,
        "n_iters": 20000,
        "save_freq": 1000,
    },
}
# =============================================================================


def setup_env():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    np.random.seed(CONFIG["reproducibility"]["seed_np"])
    torch.manual_seed(CONFIG["reproducibility"]["seed_torch"])

    Path(CONFIG["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

def load_data():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / CONFIG["paths"]["data"]

    data = sio.loadmat(str(data_file))
    x_test = data["x_test"]
    u1 = data["u1"]
    u2 = data["u2"]
    return x_test, u1, u2

def initialize_model(device, n):
    print(CONFIG["model"])
    model_type = CONFIG["model"]["name"]

    if model_type == "PNN":
        model = models.PNN(
            units=CONFIG["model"]["units"],
            n=n,
            std=CONFIG["model"]["std"],
            factor=CONFIG["model"]["factor"]
        ).to(device)

    elif model_type == "PNN2":
        model = models.PNN2(
            units=CONFIG["model"]["units"],
            n=n,
            R=CONFIG["model"]["std"]   # (kept as-is per instruction)
        ).to(device)

    elif model_type == "MHNN":
        model = models.MHNN(
            units=CONFIG["model"]["units"],
            n=n,
            std=CONFIG["model"]["std"],
            factor=CONFIG["model"]["factor"]
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model

def plot_predictions(x_test, u_pred, title, filename, u1=None, u2=None):
    n = u_pred.shape[0]
    plt.figure()

    if u1 is not None and u2 is not None:
        plt.plot(x_test, u1, "k-", label="u1 (True)")
        plt.plot(x_test, u2, "b-", label="u2 (True)")

    for j in range(n):
        plt.plot(x_test, u_pred[j, ...], "--" if u1 is not None else "-")

    plt.ylim([-0.5, 4.5])
    plt.title(title)

    output_path = Path(CONFIG["paths"]["output_dir"]) / filename
    plt.savefig(output_path)
    plt.close()

def training_loop(n_iter, optimizer, model, x_f_train, checkpoint_dir):
    loss_history = []
    iter_history = []

    try:
        for i in range(n_iter):
            optimizer.zero_grad()

            u = model.forward(x_f_train)
            u_x = torch.autograd.grad(
                u,
                x_f_train,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
            )[0]

            u_xx = torch.autograd.grad(
                u_x,
                x_f_train,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
            )[0]

            loss = torch.mean((u_xx + CONFIG["physics"]["lambda"] * torch.exp(u)) ** 2)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            loss_history.append(loss_val)
            iter_history.append(i + 1)

            if (i + 1) % CONFIG["training"]["save_freq"] == 0:
                model.eval()
                print(f"Iter {i + 1}, Loss: {loss.item()}")

                torch.save(model, checkpoint_dir / f"model_{i + 1}")

                model.train()

        print("Training complete")
    except KeyboardInterrupt:
        print(f"Interrupted by user. Aborting at {i}th iter")
    finally:
        return loss_history, iter_history

def save_metrics(loss_history, iter_history, run_dir):
    metrics = {
        "iterations": iter_history,
        "loss": loss_history,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    print("[INFO]: Saving metrics to ", run_dir / "metrics.json")

    try:
        print("[INFO]: Saving plots to ", run_dir / "figures" / "train_loss.png")

        plt.figure()
        plt.plot(iter_history, loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")  # important for PINNs
        plt.title("Training Loss")

        plt.savefig(run_dir / "figures" / "train_loss.png")
        plt.close()
    except:
        print("[ERROR]: Did not save plot due to error.")

def main(config, run_dir):
    CONFIG.update(config)

    setup_env()

    device = torch.device(CONFIG["device"])
    n = CONFIG["model"]["ensemble_size"]

    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    x_test, u1, u2 = load_data()

    # 2. Prepare Training Data
    x_f_train = torch.tensor(
        np.tile(x_test[None, ...], [n, 1, 1]),
        dtype=torch.float32,
        requires_grad=True,
    ).to(device)

    # 3. Initialize Model and Optimizer
    model = initialize_model(device, n)

    torch.save(model, checkpoint_dir / "model_0")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=0,
    )

    # 4. Training Loop
    n_iter = CONFIG["training"]["n_iters"]
    print(f"Starting training for {n_iter} iterations...")

    loss_history, iter_history = training_loop(n_iter, optimizer, model, x_f_train, checkpoint_dir)
    save_metrics(loss_history, iter_history, run_dir)

    print("1d_bratu/train.py completed")


if __name__ == "__main__":
    raise RuntimeError("Use root train.py, not this file directly.")