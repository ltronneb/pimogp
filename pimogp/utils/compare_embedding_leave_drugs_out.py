"""
The decisive version of the informativeness cross-check: hold out entire
DRUGS (not just pairs) from training, so the test set requires predicting
surfaces for at least one drug the model never saw during training.

A one-hot identity vector is structurally incapable of saying anything about
an unseen drug -- its one-hot coordinate's weights in the first encoder
layer are never touched by a training gradient, since that input coordinate
is always 0 during training. A real embedding, by contrast, still gives the
model a well-formed point in a trained representation space for a novel
drug. If the efficacy embedding beats one-hot here, that's evidence of
actual generalization, not just memorized per-drug identity -- which the
pair-level split (compare_embedding_informativeness.py) can't distinguish,
since with only ~35 drugs almost every drug recurs across train and test
pairs there regardless.
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
HELD_OUT_DRUG_FRAC = 0.25


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
    torch.set_num_threads(4)

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
    A_cols = list(embed_A.columns)
    B_cols = list(embed_B.columns)
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
    Y_all = np.array(surface_data)

    # ---- Drug-level split: hold out entire drugs, not pairs ----
    all_drugs = sorted(set(clean_pairs_df["drugA"]) | set(clean_pairs_df["drugB"]))
    rng = np.random.default_rng(SEED)
    n_held_out = max(1, int(len(all_drugs) * HELD_OUT_DRUG_FRAC))
    held_out_drugs = set(rng.choice(all_drugs, size=n_held_out, replace=False))
    seen_drugs = set(all_drugs) - held_out_drugs
    print(f"{len(all_drugs)} unique drugs -> {len(seen_drugs)} seen, {len(held_out_drugs)} held out entirely from training")

    both_seen = clean_pairs_df["drugA"].isin(seen_drugs) & clean_pairs_df["drugB"].isin(seen_drugs)
    at_least_one_unseen = ~both_seen
    train_idx = np.where(both_seen.values)[0]
    test_idx = np.where(at_least_one_unseen.values)[0]
    both_unseen = clean_pairs_df["drugA"].isin(held_out_drugs) & clean_pairs_df["drugB"].isin(held_out_drugs)
    print(f"{len(train_idx)} train pairs (both drugs seen), {len(test_idx)} test pairs (>=1 drug unseen), "
          f"of which {both_unseen.sum()} have BOTH drugs unseen")

    Y_train = torch.tensor(Y_all[train_idx].T, dtype=torch.float32)
    Y_test = torch.tensor(Y_all[test_idx].T, dtype=torch.float32)
    mean_vec, Phi, C_train, S = learn_basis_torch(Y_train, K=SVD_K)
    C_test = Phi.T @ (Y_test - mean_vec)

    baseline_train = F.mse_loss(torch.zeros_like(C_train), C_train).item()
    baseline_test = F.mse_loss(torch.zeros_like(C_test), C_test).item()

    # one-hot vocab built over ALL drugs (so held-out drugs get a real, but never-trained, coordinate)
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    onehot_dim = len(all_drugs)

    def to_onehot(names):
        X = torch.zeros(len(names), onehot_dim)
        for i, n in enumerate(names):
            X[i, drug_to_idx[n]] = 1.0
        return X

    XA_onehot_all = to_onehot(clean_pairs_df["drugA"].values)
    XB_onehot_all = to_onehot(clean_pairs_df["drugB"].values)
    print(f"Training one-hot control (dim={onehot_dim})...")
    oh_train_mse, oh_test_mse = train_deepset(
        XA_onehot_all[train_idx], XB_onehot_all[train_idx], C_train,
        XA_onehot_all[test_idx], XB_onehot_all[test_idx], C_test, onehot_dim,
    )

    XA_all = torch.tensor(clean_pairs_df[A_cols].values, dtype=torch.float32)
    XB_all = torch.tensor(clean_pairs_df[B_cols].values, dtype=torch.float32)
    print(f"Training efficacy-embedding model (dim={len(z_cols)})...")
    eff_train_mse, eff_test_mse = train_deepset(
        XA_all[train_idx], XB_all[train_idx], C_train,
        XA_all[test_idx], XB_all[test_idx], C_test, len(z_cols),
    )

    print()
    print(f"{'condition':<28s} {'train MSE':>10s} {'test MSE (>=1 unseen drug)':>28s}")
    print(f"{'mean baseline (no model)':<28s} {baseline_train:>10.4f} {baseline_test:>28.4f}")
    print(f"{'one-hot identity':<28s} {oh_train_mse:>10.4f} {oh_test_mse:>28.4f}")
    print(f"{'efficacy embedding':<28s} {eff_train_mse:>10.4f} {eff_test_mse:>28.4f}")
