import torch
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 5, hidden: list[int] = [256, 64], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
