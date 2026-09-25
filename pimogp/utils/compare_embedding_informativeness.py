"""
Cross-check: is the efficacy embedding actually informative for stage-2 SVD
coefficient prediction, or would any per-drug identity signal do just as
well (the collaborator's earlier finding: swapping the VAE embedding for a
one-hot encoding didn't move surface predictions)?

Trains the identical DeepSetPredictor architecture, on the identical pairs
and train/test split, under three input conditions:
  1. "mean baseline"   -- no model, just predict the training-mean surface
                           for every pair (C = 0). If the trained models
                           don't beat this, they aren't using their input
                           at all, which alone would explain the null result.
  2. "one-hot"          -- per-drug one-hot identity vector (pure memorization
                           capacity, zero chemical/efficacy content).
  3. "efficacy embedding" -- the 32-dim SELFIES-autoencoder + KPL1-efficacy
                              embedding from train_efficacy_embedding.py.

If (3) doesn't clearly beat (2), the embedding isn't adding information
beyond drug identity on this task/split.
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pimogp.utils.processing import load_processed_data, get_unique_drug_pairs
from pimogp.models.deep_set_predictor import DeepSetPredictor

EMBEDDINGS_CSV = "pimogp/data/cancer_drugs_efficacy_embeddings_kpl1.csv"
SVD_K = 5
EPOCHS = 2000
SEED = 42


def learn_basis_torch(Y_t, K):
    mean_vec = Y_t.mean(dim=1, keepdim=True)
    Yc = Y_t - mean_vec
    U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
    Phi = U[:, :K]
    C = (S[:K].unsqueeze(1) * Vh[:K, :])
    return mean_vec, Phi, C, S


def train_deepset(XA_train, XB_train, C_train, XA_test, XB_test, C_test, drug_dim):
    model = DeepSetPredictor(drug_dim=drug_dim, out_dim=SVD_K, phi_hidden=(64, 32), rho_hidden=(32,), dropout=0.2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=150)
    loader = DataLoader(TensorDataset(XA_train, XB_train, C_train.T.clone()), batch_size=32, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for xa, xb_, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(xa, xb_), yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = F.mse_loss(model(XA_test, XB_test), C_test.T).item()
        scheduler.step(test_loss)

    with torch.no_grad():
        train_loss = F.mse_loss(model(XA_train, XB_train), C_train.T).item()
        test_loss = F.mse_loss(model(XA_test, XB_test), C_test.T).item()
    return train_loss, test_loss


if __name__ == "__main__":
    torch.set_num_threads(4)  # shared box

    embed_df = pd.read_csv(EMBEDDINGS_CSV).set_index("Name")
    z_cols = [c for c in embed_df.columns if c.startswith("z")]
    drug_embed = embed_df[z_cols]
    drug_embed_ci = drug_embed.copy()
    drug_embed_ci.index = drug_embed_ci.index.str.upper()
    drug_embed_ci = drug_embed_ci[~drug_embed_ci.index.duplicated(keep="first")]

    df = load_processed_data()
    unique_pairs = get_unique_drug_pairs(df)
    pairs_df = pd.DataFrame(unique_pairs, columns=["drugA", "drugB"])
    pairs_df["drugA_ci"] = pairs_df["drugA"].str.upper()
    pairs_df["drugB_ci"] = pairs_df["drugB"].str.upper()

    embed_A = drug_embed_ci.add_prefix("A_")
    embed_B = drug_embed_ci.add_prefix("B_")
    pairs_df = pairs_df.join(embed_A, on="drugA_ci").join(embed_B, on="drugB_ci")
    A_cols = [c for c in embed_A.columns]
    B_cols = [c for c in embed_B.columns]

    clean_pairs_df = pairs_df.dropna(subset=A_cols + B_cols, how="any").reset_index(drop=True)

    surface_files = [f"pimogp/surfaces/{a}_{b}_KPL1.txt" for a, b in clean_pairs_df[["drugA", "drugB"]].values]
    surface_data, keep = [], []
    for i, f in enumerate(surface_files):
        try:
            surface_data.append(np.loadtxt(f).flatten())
            keep.append(i)
        except OSError:
            continue
    clean_pairs_df = clean_pairs_df.iloc[keep].reset_index(drop=True)
    print(f"{len(surface_data)} pairs with an efficacy embedding for both drugs and a KPL1 GP surface")

    N = len(surface_data)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(N, size=int(0.2 * N), replace=False)
    train_idx = np.setdiff1d(np.arange(N), test_idx)

    Y_all = np.array(surface_data).T
    Y_train = torch.tensor(Y_all[:, train_idx], dtype=torch.float32)
    Y_test = torch.tensor(Y_all[:, test_idx], dtype=torch.float32)
    mean_vec, Phi, C_train, S = learn_basis_torch(Y_train, K=SVD_K)
    C_test = Phi.T @ (Y_test - mean_vec)

    # ---- Condition 1: mean baseline (predict C=0, i.e. the training-mean surface) ----
    baseline_train = F.mse_loss(torch.zeros_like(C_train), C_train).item()
    baseline_test = F.mse_loss(torch.zeros_like(C_test), C_test).item()

    # ---- Condition 2: one-hot drug identity, same pairs/split ----
    drugs_in_use = sorted(set(clean_pairs_df["drugA"]) | set(clean_pairs_df["drugB"]))
    onehot_dim = len(drugs_in_use)
    drug_to_idx = {d: i for i, d in enumerate(drugs_in_use)}

    def to_onehot(names):
        X = torch.zeros(len(names), onehot_dim)
        for i, n in enumerate(names):
            X[i, drug_to_idx[n]] = 1.0
        return X

    XA_onehot = to_onehot(clean_pairs_df["drugA"].values)
    XB_onehot = to_onehot(clean_pairs_df["drugB"].values)
    XA_oh_train, XA_oh_test = XA_onehot[train_idx], XA_onehot[test_idx]
    XB_oh_train, XB_oh_test = XB_onehot[train_idx], XB_onehot[test_idx]

    print(f"Training one-hot control (dim={onehot_dim})...")
    oh_train_mse, oh_test_mse = train_deepset(XA_oh_train, XB_oh_train, C_train, XA_oh_test, XB_oh_test, C_test, onehot_dim)

    # ---- Condition 3: efficacy embedding, same pairs/split ----
    XA_all = torch.tensor(clean_pairs_df[A_cols].values, dtype=torch.float32)
    XB_all = torch.tensor(clean_pairs_df[B_cols].values, dtype=torch.float32)
    XA_train, XA_test = XA_all[train_idx], XA_all[test_idx]
    XB_train, XB_test = XB_all[train_idx], XB_all[test_idx]

    print(f"Training efficacy-embedding model (dim={len(z_cols)})...")
    eff_train_mse, eff_test_mse = train_deepset(XA_train, XB_train, C_train, XA_test, XB_test, C_test, len(z_cols))

    print()
    print(f"{'condition':<28s} {'train MSE':>10s} {'test MSE':>10s}")
    print(f"{'mean baseline (no model)':<28s} {baseline_train:>10.4f} {baseline_test:>10.4f}")
    print(f"{'one-hot identity':<28s} {oh_train_mse:>10.4f} {oh_test_mse:>10.4f}")
    print(f"{'efficacy embedding':<28s} {eff_train_mse:>10.4f} {eff_test_mse:>10.4f}")
