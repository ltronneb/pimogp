import torch
import torch.nn as nn


class DeepSetPredictor(nn.Module):
    """
    Deep Sets predictor for drug pair inputs: rho( phi(zA) + phi(zB) ).

    phi is a shared per-drug encoder applied independently to each drug's
    latent vector; outputs are summed (symmetric aggregation) and passed to
    the pair decoder rho. Permutation-invariant by construction.
    """
    def __init__(self, drug_dim, out_dim, phi_hidden=(128, 64), rho_hidden=(32,), dropout=0.3):
        super().__init__()

        phi_layers, in_dim = [], drug_dim
        for h in phi_hidden:
            phi_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.phi = nn.Sequential(*phi_layers)

        rho_layers = []
        for h in rho_hidden:
            rho_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        rho_layers.append(nn.Linear(in_dim, out_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, zA, zB):
        return self.rho(self.phi(zA) + self.phi(zB))
