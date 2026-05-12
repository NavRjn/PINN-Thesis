import torch
import torch.nn as nn

from . import utils


class ProblemDefinition:

    metric_keys = ["grad"]

    def __init__(self, config, device):

        self.config = config
        self.device = device

        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.physics_cfg = config.get("physics", {})

        self.model = None
        self.optimizer = None

        self._setup_operators()

    # ==========================================================
    # OPTIONAL OVERRIDES
    # ==========================================================

    # def build_model(self):
    #     return custom_model

    # def build_optimizer(self, model):
    #     return custom_optimizer

    # ==========================================================
    # INTERNAL SETUP
    # ==========================================================

    def _setup_operators(self):
        # AD operators initialized later once model exists
        self.v_laplacian = None
        self.vf_x = None
        self.vf_z = None

    def bind_model(self, model):
        params = dict(model.named_parameters())
        (self.v_laplacian, self.vf_x, self.vf_z) = utils.get_ad_operators(model)
        self.params = params


    def grid_sampler(self):

        method = self.train_cfg.get("method", "FD")
        bounds = self.physics_cfg.get( "bounds", [0, 1, 0, 1])

        N = self.physics_cfg.get("grid_N", 64)
        bz = self.train_cfg.get("bz", 10)
        sigma = self.train_cfg.get("sigma", 2.0)
        nz = self.model_cfg.get("nz", 1)
        nx = self.model_cfg.get("nx", 2)

        x_fd, _, _, dx0, dx1 = utils.get_domain_grid(bounds, N, N, self.device)

        z = utils.sample_z(bz, nz, sigma, self.device)

        if method == "AD":
            x = torch.rand(5000, nx, device=self.device)
        else:
            if self.train_cfg.get("move_grid", False):
                x = x_fd + (torch.rand(1, 1, device=self.device) * (1 / N))
            else:
                x = x_fd

        x_tp, z_tp = utils.tensor_product_xz(x, z, self.device)

        return {
            "x_tp": x_tp,
            "z_tp": z_tp,
            "z": z,
            "x_base": x,
            "dx0": dx0,
            "dx1": dx1
        }


    def loss_fn(self, model, batch):

        method = self.train_cfg.get("method", "FD")
        bz = self.train_cfg.get("bz", 10)
        N = self.physics_cfg.get("grid_N", 64)
        nx = self.model_cfg.get("nx", 2)
        ny = self.model_cfg.get("ny", 2)

        x_tp = batch["x_tp"]
        z_tp = batch["z_tp"]

        dx0 = batch["dx0"]
        dx1 = batch["dx1"]

        ys = model(x_tp, z_tp).reshape(bz, -1, ny)

        y0 = ys[..., 0]
        y1 = ys[..., 1]

        # ------------------------------------------------------
        # DIFFERENTIALS
        # ------------------------------------------------------

        if method == "AD":
            ys_lap = self.v_laplacian(self.params, x_tp, z_tp).reshape(bz, -1, ny)

            y0_lap = ys_lap[..., 0]
            y1_lap = ys_lap[..., 1]

            y_x = self.vf_x(self.params, x_tp, z_tp).reshape(bz, -1, ny, nx)

            grad_norms = y_x[0].square().mean()
        else:
            Y0 = y0.reshape(bz, N, N)
            Y1 = y1.reshape(bz, N, N)

            y0_lap = utils.laplacian_conv(Y0, dx0, dx1, self.device).reshape(bz, -1)
            y1_lap = utils.laplacian_conv(Y1, dx0, dx1, self.device).reshape(bz, -1)

            Y0_x0, Y0_x1 = utils.gradient_conv(Y0, dx0, dx1, self.device)

            grad_norms = (Y0_x0.square() + Y0_x1.square()).mean() / 32

        # ------------------------------------------------------
        # PDE
        # ------------------------------------------------------

        D1 = self.physics_cfg["D1"] / (N ** 2)
        D2 = self.physics_cfg["D2"] / (N ** 2)

        Fr = self.physics_cfg["Fr"]
        Kr = self.physics_cfg["Kr"]

        y011 = y0 * y1 * y1

        res1 = (-y011 + Fr * (1 - y0) + D1 * y0_lap)
        res2 = (y011 - (Fr + Kr) * y1 + D2 * y1_lap)

        loss_obj = (res1.square() + res2.square()).mean()

        # ------------------------------------------------------
        # REGULARIZATION
        # ------------------------------------------------------

        softplus = nn.Softplus(beta=10)

        if self.train_cfg.get("use_softclip", True):
            loss_grad = -(-softplus(-grad_norms + 1)+ 1)
        else:
            loss_grad = -torch.clip(grad_norms, max=1)

        total_loss = (loss_obj + self.train_cfg.get("w_grad", 1e-4)* loss_grad)

        metrics = {
            "obj": loss_obj.item(),
            "grad": -loss_grad.item()
        }

        return total_loss, metrics