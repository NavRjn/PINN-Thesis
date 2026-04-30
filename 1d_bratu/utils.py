import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def loss_function(model, x, y, noise, std):
    y_pred = model.forward(x)
    log_likeli = -torch.sum((y - y_pred) ** 2 / 2 / noise ** 2)
    log_prior = 0
    for p in model.parameters():
        log_prior += -torch.sum(p ** 2 / 2 / std ** 2)
    return -log_likeli - log_prior


def flatten(var_list):
    shapes = []
    vs = []
    n = var_list[0].shape[0]
    for v in var_list:
        shapes += [v.shape]
        vs += [v.reshape([n, -1])]
    vs = torch.concat(vs, dim=1)
    return vs, shapes


def unflatten(vs, shapes):
    var_list = []
    idx = 0
    for shape in shapes:
        length = shape[1] * shape[2]
        var_list += [vs[:, idx:idx+length].reshape(shape)]
        idx += length
    return var_list





# def plot_uq(x, y, mu, sd):
#     plt.
#     plt.plot(x, y, "k-")
#     plt.fill_between(inputs, preds_mean + preds_std,
#                      preds_mean - preds_std, color=colors[2], alpha=0.3)
#     plt.fill_between(inputs, preds_mean + 2. * preds_std,
#                      preds_mean - 2. * preds_std, color=colors[2], alpha=0.2)
#     plt.fill_between(inputs, preds_mean + 3. * preds_std,
#                      preds_mean - 3. * preds_std, color=colors[2], alpha=0.1)


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
