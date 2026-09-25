"""
Stage 2 of the two-stage pipeline: stage 1 (single_task_gp.py) already fit an
independent GP surface per KPL1 drug pair. Here, given the two drugs'
efficacy-shaped embeddings, a Deep Sets model (permutation-invariant in
drugA/drugB, since a pair's surface shouldn't depend on which drug is
labeled A vs B) predicts the K=5 SVD/functional-PCA coefficients that
summarise -- and, via the basis Phi, reconstruct -- that pair's surface.
This mirrors the SVD-basis half of surface_embedding_and_prediction.py and
notebooks/surface_explorer.ipynb, but swaps the VAE latents for the new
efficacy embeddings and the plain concatenation MLP for DeepSetPredictor
(pimogp/models/deep_set_predictor.py), which was already in the repo for
exactly this but wasn't wired into any pipeline yet.
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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


def reconstruct_all_torch(mean_vec, Phi, C, G):
    Yhat = mean_vec + Phi @ C
    N = C.shape[1]
    return Yhat, Yhat.T.reshape(N, G, G)


if __name__ == "__main__":
    torch.set_num_threads(4)  # shared box; keep this lightweight alongside the embedding training job
    embed_df = pd.read_csv(EMBEDDINGS_CSV).set_index("Name")
    z_cols = [c for c in embed_df.columns if c.startswith("z")]
    drug_embed = embed_df[z_cols]
    print(f"{len(drug_embed)} drugs with efficacy embeddings ({embed_df['KPL1'].notna().sum()} KPL1-labeled)")

    # Name-based join, but case-insensitive: cancer_drugs.csv/final_drugs.csv drug names
    # are ALL CAPS ("CARBOPLATIN") while the ONeil pair names are title case ("Carboplatin").
    # A case-sensitive join silently drops most of the pair universe.
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
    pairs_df = pairs_df.drop(columns=["drugA_ci", "drugB_ci"])
    A_cols = [c for c in pairs_df.columns if c.startswith("A_")]
    B_cols = [c for c in pairs_df.columns if c.startswith("B_")]

    clean_pairs_df = pairs_df.dropna(how="any")
    dropped = len(pairs_df) - len(clean_pairs_df)
    print(f"{len(clean_pairs_df)} / {len(pairs_df)} pairs have embeddings for both drugs ({dropped} dropped)")

    surface_files = [f"pimogp/surfaces/{a}_{b}_KPL1.txt" for a, b in clean_pairs_df[["drugA", "drugB"]].values]
    surface_data, keep = [], []
    for i, f in enumerate(surface_files):
        try:
            surface_data.append(np.loadtxt(f).flatten())
            keep.append(i)
        except OSError:
            continue
    clean_pairs_df = clean_pairs_df.iloc[keep].reset_index(drop=True)
    print(f"{len(surface_data)} pairs have a KPL1 GP surface on disk")

    N = len(surface_data)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(N, size=int(0.2 * N), replace=False)
    train_idx = np.setdiff1d(np.arange(N), test_idx)

    Y_all = np.array(surface_data).T
    Y_train = torch.tensor(Y_all[:, train_idx], dtype=torch.float32)
    Y_test = torch.tensor(Y_all[:, test_idx], dtype=torch.float32)

    XA_all = torch.tensor(clean_pairs_df[A_cols].values, dtype=torch.float32)
    XB_all = torch.tensor(clean_pairs_df[B_cols].values, dtype=torch.float32)
    XA_train, XA_test = XA_all[train_idx], XA_all[test_idx]
    XB_train, XB_test = XB_all[train_idx], XB_all[test_idx]

    mean_vec, Phi, C_train, S = learn_basis_torch(Y_train, K=SVD_K)
    C_test = Phi.T @ (Y_test - mean_vec)

    model = DeepSetPredictor(drug_dim=len(z_cols), out_dim=SVD_K, phi_hidden=(64, 32), rho_hidden=(32,), dropout=0.2)
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
        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                train_loss = F.mse_loss(model(XA_train, XB_train), C_train.T).item()
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:4d} | train MSE: {train_loss:.4f}  test MSE: {test_loss:.4f}  lr: {lr:.2e}")

    torch.save(model.state_dict(), "pimogp/data/deep_set_predictor_efficacy_embedding.pt")

    model.eval()
    with torch.no_grad():
        C_pred_train = model(XA_train, XB_train).T
        C_pred_test = model(XA_test, XB_test).T
    Yhat_train, _ = reconstruct_all_torch(mean_vec, Phi, C_pred_train, 30)
    Yhat_test, _ = reconstruct_all_torch(mean_vec, Phi, C_pred_test, 30)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(Y_train.numpy().flatten(), Yhat_train.detach().numpy().flatten(), alpha=0.4, c=Y_train.numpy().flatten(), cmap="viridis")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"Train [{len(train_idx)} pairs]")
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test.numpy().flatten(), Yhat_test.detach().numpy().flatten(), alpha=0.4, c=Y_test.numpy().flatten(), cmap="viridis")
    plt.xlabel("True"); plt.title(f"Test [{len(test_idx)} pairs — held out]")
    plt.suptitle("Surface reconstruction from Deep-Set-predicted SVD coefficients (efficacy-embedding features)")
    plt.tight_layout()
    plt.savefig("efficacy_embedding_svd_coeff_prediction.png")
    plt.close()
    print("Saved efficacy_embedding_svd_coeff_prediction.png")
