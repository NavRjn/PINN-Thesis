import torch

class ProblemDefinition:

    metric_keys = ["mse"]

    def __init__(self, config, device):
        self.config, self.device = config, device
        self.train_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})

    def grid_sampler(self):
        n = self.train_cfg.get("n_points", 128)
        x = torch.linspace(0, 1, n, device=self.device).view(-1, 1)
        x.requires_grad_(False)
        return {"x": x}

    def loss_fn(self, model, batch):
        x = batch["x"]
        y = model(x)
        target = x  # identity task
        loss = ((y - target) ** 2).mean()
        return loss, {"obj": loss.item(), "mse": loss.item()}