import torch
import numpy as np

from . import models


class ProblemDefinition:

    metric_keys = ["u_mid", "model_wise_loss"]

    model_map = {
        "PNN": models.PNN,
        "PNN2": models.PNN2,
        "MHNN": models.MHNN
    }

    def __init__(self, config, device):

        self.config = config
        self.device = device

        self.n = config["model"].get("ensemble_size", 100)

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

    # ==========================================================
    # REQUIRED METHODS
    # ==========================================================

    def build_model(self):

        cfg = self.config["model"]

        model_type = cfg.get("name", "PNN")
        units = cfg.get("units", 50)
        std = cfg.get("std", 1.0)
        factor = cfg.get("factor", 1.0)

        if model_type not in self.model_map:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type == "PNN2":
            model = models.PNN2(
                units=units,
                n=self.n,
                R=std
            )

        else:
            model = self.model_map[model_type](
                units=units,
                n=self.n,
                std=std,
                factor=factor
            )

        return model.to(self.device)

    def build_optimizer(self):

        lr = self.config["training"].get("lr", 1e-3)

        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    def grid_sampler(self):

        x = torch.linspace(0, 1, 100).reshape(-1, 1)

        x_ensemble = x.repeat(self.n, 1, 1).to(self.device)

        x_ensemble.requires_grad_(True)

        if self.config["training"].get("sigma", None) is None:

            lam = torch.tensor(
                self.config["physics"].get("lambda", 1.0),
                device=self.device
            )

        else:

            lam = torch.normal(
                mean=self.config["physics"].get("lambda", 1.0),
                std=self.config["training"].get("sigma", 1.0),
                size=(1,),
                device=self.device
            )

        return {
            "x_ensemble": x_ensemble,
            "z": abs(lam)
        }

    def loss_fn(self, model, batch):

        x = batch["x_ensemble"]
        lam = batch["z"]

        u = model(x, lam)

        u_x = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]

        residual = u_xx + lam * torch.exp(u)

        model_wise_loss = torch.mean(
            residual ** 2,
            dim=(1, 2)
        )

        u_mid = u[:, 50, :].detach().reshape(-1)

        total_loss = model_wise_loss.mean()

        metrics = {
            "obj": total_loss.item(),
            "u_mid": u_mid.cpu().numpy().tolist(),
            "model_wise_loss":
                model_wise_loss.detach().cpu().numpy().tolist()
        }

        return total_loss, metrics