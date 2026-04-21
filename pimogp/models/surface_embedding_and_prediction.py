import numpy as np
import selfies as sf
import pandas as pd
import random
import os
import yaml
import torch
import importlib
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from pimogp.utils.processing import load_processed_data, get_unique_drug_pairs
from pimogp.models.mlp_predictor import MLPPredictor

def dearomatize_smiles(s):
    if pd.isna(s):
        return s  # keep NaNs as-is
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None  # or return s if you prefer to keep bad strings
    Chem.Kekulize(mol, clearAromaticFlags=True)
    # kekuleSmiles=True forces explicit non-aromatic bonds, uppercase atoms
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def stack_surfaces(mus):
    G = mus[0].shape[0]
    Y = np.stack([m.reshape(-1) for m in mus], axis=1)  # (G^2, N)
    return Y  # columns = pairs

def learn_basis(Y, K):
    # Y: (G^2, N)
    mean_vec = Y.mean(axis=1, keepdims=True)           # (G^2, 1)
    Yc = Y - mean_vec                                   # center across pairs
    # SVD
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)   # U:(G^2,N), S:(N,), Vt:(N,N)
    Phi = U[:, :K]                                      # (G^2, K)  orthonormal basis
    C = (S[:K, None] * Vt[:K, :])                       # (K, N) coefficients (scores)
    return mean_vec, Phi, C, S

def reconstruct(mean_vec, Phi, c_vec, G):
    yhat = mean_vec.squeeze() + Phi @ c_vec            # (G^2,)
    return yhat.reshape(G, G)

def project_to_coeffs(y, mean_vec, Phi):
    yc = y.reshape(-1) - mean_vec.squeeze()
    c = Phi.T @ yc                                     # (K,)
    return c

def learn_basis_torch(Y_t, K):  # Y_t: (G^2, N) float32 tensor
    mean_vec = Y_t.mean(dim=1, keepdim=True)
    Yc = Y_t - mean_vec
    U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)  # Vh: (N,N)
    Phi = U[:, :K]                                        # (G^2, K)
    C = (S[:K].unsqueeze(1) * Vh[:K, :])                  # (K, N)
    return mean_vec, Phi, C, S

def reconstruct_torch(mean_vec, Phi, c_vec, G):
    yhat = mean_vec.squeeze() + Phi @ c_vec            # (G^2,)
    return yhat.reshape(G, G)

def reconstruct_all_torch(mean_vec, Phi, C, G):
    """
    mean_vec: (G^2, 1)
    Phi:      (G^2, K)
    C:        (K, N)
    returns:  Yhat of shape (G^2, N) and optionally reshaped (N, G, G)
    """
    # (G^2, N) = (G^2,1) + (G^2,K) @ (K,N)
    Yhat = mean_vec + Phi @ C

    # reshape to (N, G, G) if you want images
    N = C.shape[1]
    Yhat_imgs = Yhat.T.reshape(N, G, G)  # each [i] is (G, G) for pair i
    return Yhat, Yhat_imgs

def project_to_coeffs_torch(y, mean_vec, Phi):
    yc = y.reshape(-1) - mean_vec.squeeze()
    c = Phi.T @ yc                                     # (K,)
    return c

if __name__ == "__main__":

        # Load the latent cancer drug representations
        z_all = torch.load("pimogp/latents/transformer_vae_cancer_drugs/z_out.pt")
        mu_all = torch.load("pimogp/latents/transformer_vae_cancer_drugs/mu_out.pt")
        logvar_all = torch.load("pimogp/latents/transformer_vae_cancer_drugs/logvar_out.pt")

        # Load the smiles of the cancer drugs
        cancer_drugs_smiles = pd.read_csv("pimogp/data/cancer_drugs.csv")
        cancer_drugs_smiles['dearomatized_smiles'] = cancer_drugs_smiles['smiles'].apply(dearomatize_smiles)
        cancer_drugs_smiles['old'] = cancer_drugs_smiles['smiles']
        cancer_drugs_smiles['smiles'] = cancer_drugs_smiles['dearomatized_smiles']

        # Tag them with the drug names
        cancer_drugs_df = pd.read_csv("pimogp/data/final_drugs.csv")
        names = cancer_drugs_df.iloc[cancer_drugs_smiles.index]['Name']

        # Tag them with the latent representations
        drug_latents = pd.DataFrame(z_all, index=names)
        drug_latents.to_csv("pimogp/data/cancer_drugs_latents.csv", index=True)

        # Build drug pair latent matrix
        df = load_processed_data()
        unique_pairs = get_unique_drug_pairs(df)
        pairs_df = pd.DataFrame(unique_pairs, columns=["drugA", "drugB"])

        latents_A = drug_latents.add_prefix("A_")
        latents_B = drug_latents.add_prefix("B_")
        pairs_df = (
            pairs_df
            .join(latents_A, on="drugA")
            .join(latents_B, on="drugB")
        )
        pairs_df.to_csv("pimogp/data/cancer_drugs_latents_pairs.csv", index=False)

        clean_pairs_df = pairs_df.dropna(how="any")
        surface_df = clean_pairs_df[['drugA', 'drugB']].copy()

        # Load GP posterior mean surfaces
        surface_files = [
            f"pimogp/surfaces/{drugA}_{drugB}_KPL1.txt"
            for drugA, drugB in clean_pairs_df[['drugA', 'drugB']].values
        ]
        surface_data = [np.loadtxt(f).flatten() for f in surface_files]
        surface_df["surface_data"] = surface_data
        surface_df.to_csv("pimogp/data/cancer_drugs_latents_pairs_surface_data.csv", index=False)

        # --- Train / test split ---
        N = len(surface_data)
        rng = np.random.default_rng(42)
        test_idx  = rng.choice(N, size=int(0.2 * N), replace=False)
        train_idx = np.setdiff1d(np.arange(N), test_idx)

        Y_all   = np.array(surface_data).T                                   # (900, N)
        Y_train = torch.tensor(Y_all[:, train_idx], dtype=torch.float32)     # (900, N_train)
        Y_test  = torch.tensor(Y_all[:, test_idx],  dtype=torch.float32)     # (900, N_test)

        latent_cols = [c for c in clean_pairs_df.columns if c.startswith("A_") or c.startswith("B_")]
        X_all   = torch.tensor(clean_pairs_df[latent_cols].values, dtype=torch.float32)
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]

        train_pairs = clean_pairs_df.iloc[train_idx][['drugA', 'drugB']].reset_index(drop=True)
        test_pairs  = clean_pairs_df.iloc[test_idx][['drugA',  'drugB']].reset_index(drop=True)

        # --- PCA on latents (fit on train only) ---
        PCA_K = 32
        X_train_np = X_train.numpy()
        pca_mean = X_train_np.mean(axis=0, keepdims=True)
        Xc = X_train_np - pca_mean
        _, _, Vt_pca = np.linalg.svd(Xc, full_matrices=False)
        W_pca = Vt_pca[:PCA_K].T                                         # (512, 32)

        X_train_pca = torch.tensor((X_train_np - pca_mean) @ W_pca, dtype=torch.float32)
        X_test_pca  = torch.tensor((X_test.numpy() - pca_mean) @ W_pca, dtype=torch.float32)

        np.save("pimogp/data/pca_mean.npy", pca_mean)
        np.save("pimogp/data/pca_components.npy", W_pca)

        # --- SVD basis (learned on train only) ---
        mean_vec, Phi, C_train, S = learn_basis_torch(Y_train, K=5)    # C_train: (5, N_train)
        C_test = Phi.T @ (Y_test - mean_vec)                            # (5, N_test)

        torch.save({"mean_vec": mean_vec, "Phi": Phi, "C": C_train, "S": S,
                    "pairs": train_pairs},
                   "pimogp/data/svd_basis.pt")

        # SVD reconstruction diagnostic (train)
        Yhat_train, _ = reconstruct_all_torch(mean_vec, Phi, C_train, 30)

        plt.figure(figsize=(15, 15))
        plt.scatter(Y_train.numpy().flatten(), Yhat_train.detach().numpy().flatten(),
                    alpha=0.5, c=Y_train.numpy().flatten(), cmap='viridis')
        plt.colorbar()
        plt.xlabel("True Surface Data", fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylabel("SVD Reconstructed", fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f"SVD Reconstruction — train [{len(train_idx)} pairs]", fontsize=15)
        plt.savefig("true_vs_predicted_surface_data_train_set.png")
        plt.close()

        # --- MLP: drug pair latents -> C coefficients ---
        Y_coeff_train = C_train.T.clone()    # (N_train, 5)
        Y_coeff_test  = C_test.T.clone()     # (N_test,  5)

        model = MLPPredictor(in_dim=PCA_K, out_dim=C_train.shape[0], hidden=[256, 64])
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=300)
        dataset = TensorDataset(X_train_pca, Y_coeff_train)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(5000):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                test_loss = F.mse_loss(model(X_test_pca), Y_coeff_test).item()
            scheduler.step(test_loss)
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    tr_loss = F.mse_loss(model(X_train_pca), Y_coeff_train).item()
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4d} | train MSE: {tr_loss:.4f}  test MSE: {test_loss:.4f}  lr: {lr:.2e}")

        torch.save(model.state_dict(), "pimogp/data/mlp_predictor.pt")
        print("Saved MLP predictor to pimogp/data/mlp_predictor.pt")

        # --- Reconstruct surfaces from MLP-predicted coefficients ---
        model.eval()
        with torch.no_grad():
            C_pred_train = model(X_train_pca).T    # (5, N_train)
            C_pred_test  = model(X_test_pca).T     # (5, N_test)

        Yhat_mlp_train, _ = reconstruct_all_torch(mean_vec, Phi, C_pred_train, 30)
        Yhat_mlp_test, Yhat_mlp_test_imgs = reconstruct_all_torch(mean_vec, Phi, C_pred_test, 30)

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.scatter(Y_train.numpy().flatten(), Yhat_mlp_train.detach().numpy().flatten(),
                    alpha=0.4, c=Y_train.numpy().flatten(), cmap='viridis')
        plt.xlabel("True", fontsize=13); plt.ylabel("MLP predicted", fontsize=13)
        plt.title(f"Train [{len(train_idx)} pairs]", fontsize=13)
        plt.subplot(1, 2, 2)
        plt.scatter(Y_test.numpy().flatten(), Yhat_mlp_test.detach().numpy().flatten(),
                    alpha=0.4, c=Y_test.numpy().flatten(), cmap='viridis')
        plt.xlabel("True", fontsize=13)
        plt.title(f"Test [{len(test_idx)} pairs — held out]", fontsize=13)
        plt.suptitle("MLP surface reconstruction", fontsize=15)
        plt.tight_layout()
        plt.savefig("true_vs_mlp_reconstructed_surface_data3.png")
        plt.close()

        # Single held-out pair comparison
        pair_id = 19
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Y_test.numpy()[:, pair_id].reshape(30, 30), cmap='viridis')
        plt.colorbar()
        title = test_pairs.iloc[pair_id]['drugA'] + " + " + test_pairs.iloc[pair_id]['drugB']
        plt.title(title + " — True")
        plt.subplot(1, 2, 2)
        plt.imshow(Yhat_mlp_test_imgs[pair_id].detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(title + " — MLP predicted")
        plt.savefig(f"true_vs_mlp_reconstructed_{title}.png")
        plt.close()

