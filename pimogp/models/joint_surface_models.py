"""
Model classes for the joint (single-stage) KPL1 surface-prediction models.
See EFFICACY_EMBEDDING_EXPERIMENTS.md sections 6-7 for the full derivation.
"""
import torch
import torch.nn as nn

from pimogp.models.deep_set_predictor import DeepSetPredictor


class JointSurfaceModel(nn.Module):
    """
    Additive model: Yhat(cell; zA, zB) = mean_surface[cell] + Phi[cell] . g(zA, zB)
    """
    def __init__(self, num_cells, drug_dim, K, mean_init):
        super().__init__()
        self.mean_surface = nn.Parameter(mean_init.clone())          # (num_cells,)
        self.Phi = nn.Parameter(torch.randn(num_cells, K) * 0.01)     # (num_cells, K)
        self.g = DeepSetPredictor(drug_dim=drug_dim, out_dim=K, phi_hidden=(64, 32), rho_hidden=(32,), dropout=0.2)

    def forward(self, cell_idx, zA, zB):
        g_out = self.g(zA, zB)
        phi_out = self.Phi[cell_idx]
        deviation = (phi_out * g_out).sum(dim=-1)
        return self.mean_surface[cell_idx] + deviation


class MonotherapyCurve(nn.Module):
    """Shared function: one drug's viability at one dose, given its embedding. Sigmoid output -> [0,1]."""
    def __init__(self, drug_dim, hidden=(64, 32)):
        super().__init__()
        dims = [drug_dim + 1] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, conc, z):
        x = torch.cat([conc.unsqueeze(-1), z], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class JointSurfaceModelBliss(nn.Module):
    """
    Bliss-independence model:
      Yhat(cell; zA, zB) = curve(cA, zA) . curve(cB, zB) + Phi[cell] . g(zA, zB)
    """
    def __init__(self, num_cells, drug_dim, K):
        super().__init__()
        self.Phi = nn.Parameter(torch.randn(num_cells, K) * 0.01)
        self.g = DeepSetPredictor(drug_dim=drug_dim, out_dim=K, phi_hidden=(64, 32), rho_hidden=(32,), dropout=0.2)
        self.curve = MonotherapyCurve(drug_dim)

    def forward(self, cell_idx, concA, concB, zA, zB):
        bliss = self.curve(concA, zA) * self.curve(concB, zB)
        g_out = self.g(zA, zB)
        phi_out = self.Phi[cell_idx]
        deviation = (phi_out * g_out).sum(dim=-1)
        return bliss + deviation
