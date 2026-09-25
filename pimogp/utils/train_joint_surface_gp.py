"""
Joint surface model with a GP-based monotherapy curve component, replacing
train_bliss.py's MLP curve(c,z) with an exact Gaussian Process:

    curve_gp(c, z) ~ GP(0, k([c,z], [c,z']))     ARD RBF kernel, GaussianLikelihood
    Yhat(cell; zA,zB) = curve_gp_mean(cA,zA) . curve_gp_mean(cB,zB) + Phi[cell] . g(zA,zB)

Two changes from the MLP version:
  1. The curve is fit directly on real ONeil monotherapy data (exact GP,
     small n so no need for a variational/sparse approximation), not just
     indirectly through the combo-surface loss -- a cleaner supervision
     signal, and it gives genuine posterior uncertainty per drug/dose.
  2. That fit is frozen before Phi/g are trained, and its posterior mean is
     precomputed once per (drug, concentration level) rather than queried
     inside the training loop -- avoids GP train()/eval() mode conflicts
     with the outer model and is much cheaper (the GP never needs to see a
     combo row, only the ~10-point-per-drug monotherapy grid).

The GP is fit on SEEN drugs' real curves only (leave-drugs-out split, no
leakage). Its posterior std for held-out drugs is the key diagnostic this
script exists to produce: an ARD kernel should place a genuinely unseen
drug's one-hot vector far from every training point along those identity
dimensions, collapsing to the GP's prior (high uncertainty) -- something a
real embedding, whose unseen points can still sit near seen ones, should
not do to nearly the same degree.

Usage:
  python train_joint_surface_gp.py --input curve_embedding
  python train_joint_surface_gp.py --input onehot
(leave-drugs-out split only -- this is the decisive test)
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gpytorch
from torch.utils.data import DataLoader, TensorDataset

from pimogp.models.deep_set_predictor import DeepSetPredictor
from pimogp.utils.joint_surface_data import (
    load_kpl1_grid, load_drug_features, split_pairs, to_onehot_tensor,
    EMBEDDINGS_CSV, K, EPOCHS, BATCH_SIZE, SEED,
)

GP_TRAIN_ITERS = 300


class TanimotoKernel(gpytorch.kernels.Kernel):
    """
    Tanimoto/Jaccard similarity kernel for binary vectors -- the standard
    molecular-fingerprint similarity kernel in cheminformatics, and a
    genuinely valid PSD kernel (not RBF applied to binary data). For pure
    one-hot vectors it reduces to exactly 1 (same category) or 0 (different
    category), same as RBF's degenerate structure -- included here so that
    claim is checked empirically, not just asserted.
    """
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            dot = (x1 * x2).sum(-1)
            denom = (x1 * x1).sum(-1) + (x2 * x2).sum(-1) - dot
            return dot / (denom + 1e-8)
        dot = x1 @ x2.transpose(-2, -1)
        x1_sq = (x1 * x1).sum(-1, keepdim=True)
        x2_sq = (x2 * x2).sum(-1, keepdim=True).transpose(-2, -1)
        denom = x1_sq + x2_sq - dot
        return dot / (denom + 1e-8)


class MonotherapyCurveGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def make_covar_module(drug_dim, kernel_type):
    if kernel_type == "rbf":
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=drug_dim + 1))
    elif kernel_type == "tanimoto":
        conc_kernel = gpytorch.kernels.RBFKernel(active_dims=[0])
        drug_kernel = TanimotoKernel(active_dims=list(range(1, drug_dim + 1)))
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.ProductKernel(conc_kernel, drug_kernel))
    else:
        raise ValueError(f"unknown kernel_type: {kernel_type}")


def fit_curve_gp(train_x, train_y, drug_dim, kernel_type):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    covar_module = make_covar_module(drug_dim, kernel_type)
    model = MonotherapyCurveGP(train_x, train_y, likelihood, covar_module)
    model.train(); likelihood.train()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(GP_TRAIN_ITERS):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"  GP fit iter {i+1}/{GP_TRAIN_ITERS} | -MLL: {loss.item():.4f}")
    model.eval(); likelihood.eval()
    return model, likelihood


class JointSurfaceModelGP(nn.Module):
    """Yhat = bliss_term (precomputed from the frozen curve GP) + Phi[cell] . g(zA, zB)"""
    def __init__(self, num_cells, drug_dim, K):
        super().__init__()
        self.Phi = nn.Parameter(torch.randn(num_cells, K) * 0.01)
        self.g = DeepSetPredictor(drug_dim=drug_dim, out_dim=K, phi_hidden=(64, 32), rho_hidden=(32,), dropout=0.2)

    def forward(self, cell_idx, bliss_term, zA, zB):
        g_out = self.g(zA, zB)
        phi_out = self.Phi[cell_idx]
        deviation = (phi_out * g_out).sum(dim=-1)
        return bliss_term + deviation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", choices=["old_embedding", "curve_embedding", "onehot"], required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)

    kpl1, conc_levels, n_levels, num_cells = load_kpl1_grid()
    kpl1, meta, drug_dim = load_drug_features(args.input, kpl1)
    print(f"{len(kpl1)} rows, {kpl1[['drugA', 'drugB']].drop_duplicates().shape[0]} pairs, "
          f"input={args.input} (drug_dim={drug_dim})")

    train_df, test_df = split_pairs(kpl1, "leave_drugs_out")
    print(f"{len(train_df)} train rows, {len(test_df)} test rows")

    all_drugs = sorted(set(kpl1["drugA"]) | set(kpl1["drugB"]))
    seen_drugs = sorted(set(train_df["drugA"]) | set(train_df["drugB"]))
    held_out_drugs = sorted(set(all_drugs) - set(seen_drugs))
    print(f"{len(seen_drugs)} seen drugs, {len(held_out_drugs)} held out entirely")

    # ---- Per-drug feature vector lookup (works for both embedding and one-hot inputs) ----
    if args.input == "onehot":
        drug_to_idx = kpl1.attrs["drug_to_idx"]

        def drug_vec(name):
            v = np.zeros(drug_dim, dtype=np.float32)
            v[drug_to_idx[name]] = 1.0
            return v
    else:
        embed_df = pd.read_csv(EMBEDDINGS_CSV[args.input]).set_index("Name")
        z_cols = [c for c in embed_df.columns if c.startswith("z")]
        de = embed_df[z_cols].copy()
        de.index = de.index.str.upper()
        de = de[~de.index.duplicated(keep="first")]

        def drug_vec(name):
            return de.loc[name.upper()].values.astype(np.float32)

    # ---- Real ONeil monotherapy curve data, SEEN drugs only (no leakage) ----
    edge_A = kpl1[kpl1["drugB_conc"] == 0][["drugA", "drugA_conc", "fMean"]].rename(
        columns={"drugA": "drug", "drugA_conc": "conc"})
    edge_B = kpl1[kpl1["drugA_conc"] == 0][["drugB", "drugB_conc", "fMean"]].rename(
        columns={"drugB": "drug", "drugB_conc": "conc"})
    edges = pd.concat([edge_A, edge_B], ignore_index=True)
    edges = edges[edges["drug"].isin(seen_drugs)]
    curve_table = edges.groupby(["drug", "conc"])["fMean"].mean().unstack("conc").sort_index(axis=1).dropna()
    print(f"GP training data: {len(curve_table)} seen drugs x {curve_table.shape[1]} concentration levels "
          f"= {curve_table.size} monotherapy points")

    gp_X, gp_y = [], []
    for drug, row in curve_table.iterrows():
        v = drug_vec(drug)
        for conc, viability in row.items():
            gp_X.append(np.concatenate([[conc], v]))
            gp_y.append(viability)
    gp_X = torch.tensor(np.array(gp_X), dtype=torch.float32)
    gp_y = torch.tensor(np.array(gp_y), dtype=torch.float32)

    kernel_type = "tanimoto" if args.input == "onehot" else "rbf"
    print(f"Fitting monotherapy curve GP (kernel={kernel_type})...")
    curve_gp, curve_likelihood = fit_curve_gp(gp_X, gp_y, drug_dim=drug_dim, kernel_type=kernel_type)

    # ---- Precompute posterior mean/std for every (drug, concentration level) ----
    query_X, query_keys = [], []
    for drug in all_drugs:
        v = drug_vec(drug)
        for lvl in conc_levels:
            query_X.append(np.concatenate([[lvl], v]))
            query_keys.append((drug, round(lvl, 6)))
    query_X = torch.tensor(np.array(query_X), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = curve_likelihood(curve_gp(query_X))
        pred_mean = pred.mean.numpy()
        pred_std = pred.stddev.numpy()
    curve_mean_lookup = dict(zip(query_keys, pred_mean))
    curve_std_lookup = dict(zip(query_keys, pred_std))

    # ---- Uncertainty diagnostic: seen vs. held-out drugs ----
    seen_stds = [curve_std_lookup[(d, round(c, 6))] for d in seen_drugs for c in conc_levels]
    unseen_stds = [curve_std_lookup[(d, round(c, 6))] for d in held_out_drugs for c in conc_levels]
    print(f"Curve GP posterior std | seen drugs: mean={np.mean(seen_stds):.4f}  "
          f"| held-out drugs: mean={np.mean(unseen_stds):.4f}  "
          f"(ratio unseen/seen: {np.mean(unseen_stds)/np.mean(seen_stds):.2f}x)")

    # ---- Bliss term per row, from the frozen GP's posterior mean ----
    def bliss_terms(d):
        mA = np.array([curve_mean_lookup[(a, round(c, 6))] for a, c in zip(d["drugA"], d["drugA_conc"])])
        mB = np.array([curve_mean_lookup[(b, round(c, 6))] for b, c in zip(d["drugB"], d["drugB_conc"])])
        return torch.tensor(mA * mB, dtype=torch.float32)

    def to_tensors(d):
        cell = torch.tensor(d["cell_idx"].values, dtype=torch.long)
        bliss = bliss_terms(d)
        y = torch.tensor(d["fMean"].values, dtype=torch.float32)
        if args.input == "onehot":
            xa = to_onehot_tensor(d["drugA"].values, kpl1.attrs["drug_to_idx"], drug_dim)
            xb = to_onehot_tensor(d["drugB"].values, kpl1.attrs["drug_to_idx"], drug_dim)
        else:
            A_cols, B_cols = meta
            xa = torch.tensor(d[A_cols].values, dtype=torch.float32)
            xb = torch.tensor(d[B_cols].values, dtype=torch.float32)
        return cell, bliss, xa, xb, y

    cell_train, bliss_train, XA_train, XB_train, y_train = to_tensors(train_df)
    cell_test, bliss_test, XA_test, XB_test, y_test = to_tensors(test_df)

    model = JointSurfaceModelGP(num_cells=num_cells, drug_dim=drug_dim, K=K)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=15)
    loader = DataLoader(TensorDataset(cell_train, bliss_train, XA_train, XB_train, y_train),
                         batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for cb, bl, xa, xb, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(cb, bl, xa, xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(model(cell_train, bliss_train, XA_train, XB_train), y_train).item()
            test_mse = F.mse_loss(model(cell_test, bliss_test, XA_test, XB_test), y_test).item()
            bliss_only_test_mse = F.mse_loss(bliss_test, y_test).item()
        scheduler.step(test_mse)
        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d} | train MSE: {train_mse:.5f}  test MSE: {test_mse:.5f}  "
                  f"(Bliss-GP-only test MSE: {bliss_only_test_mse:.5f})  lr: {lr:.2e}")

    out_path = f"pimogp/data/joint_surface_gp_leave_drugs_out_{args.input}_{kernel_type}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")
