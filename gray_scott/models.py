import torch
from torch import nn
from torch.nn.functional import relu, tanh
from .GINN.NN import encode, GeneralNet
from .GINN.siren import SIREN
from .utils import get_logger

logger = get_logger()

class GeneralNetPosEnc(GeneralNet):
    def __init__(self, ks, act=tanh, N_posenc=None):
        super(GeneralNetPosEnc, self).__init__(ks=ks, act=act)
        self.N_posenc = N_posenc

    def forward(self, x, z):
        x = encode(x, N=self.N_posenc)
        x = torch.hstack([x, z])
        x = self.fcs[0](x)
        for i in range(2, self.D + 1):
            x = self.fcs[i - 1](self.act(x))
        return torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self, ks):
        super(Net, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
                                  for in_features, out_features in zip(self.ks[:-1], self.ks[1:])])
        self.D = len(self.fcs)

    def forward(self, x, z):
        x_ = self.fcs[0](torch.hstack([x, z]))
        for i in range(2, self.D + 1):
            x_ = self.fcs[i - 1](relu(x_))
        return x_


class PosEncSIREN(SIREN):
    def __init__(self, ks, w0=1.0, w0_initial=30.0, initializer='siren', c=6, N_posenc=0):
        in_features = ks[0]
        layers = ks[1:-1]
        out_features = ks[-1]
        SIREN.__init__(self, layers, in_features, out_features, w0, w0_initial, initializer=initializer, c=c)
        self.ks = ks
        self.N_posenc = N_posenc

    def forward(self, x, z):
        if self.N_posenc:
            x = encode(x, N=self.N_posenc)
        x = torch.hstack([x, z])
        return torch.sigmoid(self.network(x))


class GeneralNetFFN(GeneralNet):
    def __init__(self, ks, act=tanh, N_ffeat=0, sigma=1, nx=2):
        super(GeneralNetFFN, self).__init__(ks=ks, act=act)
        self.N_ffeat = N_ffeat
        if self.N_ffeat:
            self.B = torch.normal(0, sigma ** 2, size=[self.N_ffeat, nx])

    def forward(self, x, z):
        if self.N_ffeat:
            x_B = 2 * torch.pi * x @ self.B.T
            x_sin = torch.sin(x_B)
            x_cos = torch.cos(x_B)
            x = torch.hstack([x_sin, x_cos])
        x = torch.hstack([x, z])
        x = self.fcs[0](x)
        for i in range(1, self.D):
            x = self.fcs[i](self.act(x))
        return torch.sigmoid(x)


class DualNet(nn.Module):
    def __init__(self, models):
        super(DualNet, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, z):
        return torch.stack([model(x, z) for model in self.models], dim=-1)


def init_model(arch="SIREN+POSENC", use_two_models=True, nz=0, nx=2):
    assert arch in ["SIREN", "SIREN+POSENC", "FFN"], "unknown architecture"
    logger.debug(f"Initializing model: arch={arch}, use_two_models={use_two_models}, nz={nz}, nx={nx}")
    if arch == "SIREN":
        if use_two_models:
            model = DualNet([PosEncSIREN(ks=[nx + nz, 256, 128, 1]) for _ in range(2)])
        else:
            model = PosEncSIREN(ks=[nx + nz, 256, 256, 2], w0_initial=30)
    elif arch == "SIREN+POSENC":
        N_posenc = 1
        enc_dim = 2 * nx * N_posenc
        if use_two_models:
            model = DualNet(
                [PosEncSIREN(ks=[enc_dim + nz, 256, 128, 1], N_posenc=N_posenc, w0_initial=3.0) for _ in range(2)])
        else:
            model = PosEncSIREN(ks=[enc_dim + nz, 256, 256, 2], N_posenc=N_posenc, w0_initial=3.0)
    elif arch == "FFN":
        N_ffeat = 128
        enc_dim = 2 * N_ffeat if N_ffeat else nx
        if use_two_models:
            model = DualNet([GeneralNetFFN(ks=[enc_dim + nz, 256, 128, 1], N_ffeat=N_ffeat, sigma=3) for _ in range(2)])
        else:
            model = GeneralNetFFN(ks=[enc_dim + nz, 256, 256, 2], N_ffeat=N_ffeat, sigma=3)

    return model


def build_model_from_config(config):
    """Builds and returns the model based on the experiment configuration."""
    arch = config.get("arch", "SIREN+POSENC")
    use_two_models = config.get("name", "DualNet") == "DualNet"
    nz = config.get("nz", 1)
    nx = config.get("nx", 2)  # Gray-Scott uses 2 spatial dimensions

    return init_model(arch=arch, use_two_models=use_two_models, nz=nz, nx=nx)