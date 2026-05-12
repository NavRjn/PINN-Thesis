import torch.nn as nn

class PNN(nn.Module):
    def __init__(self, units=32, layers=2, **_):
        super().__init__()
        net = [nn.Linear(1, units), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(units, units), nn.Tanh()]
        net += [nn.Linear(units, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)