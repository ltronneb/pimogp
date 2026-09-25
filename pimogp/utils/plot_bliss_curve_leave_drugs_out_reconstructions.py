"""
For the best model of the session (Bliss-independence joint model, curve-
trained embedding, leave-drugs-out): plots (a) true/predicted/residual
surfaces for held-out test pairs -- including pairs where BOTH drugs were
never seen in training -- and (b) the K=5 learnt Phi basis functions as
heatmaps over the concentration grid.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pimogp.models.joint_surface_models import JointSurfaceModelBliss
from pimogp.utils.joint_surface_data import PROCESSED_CSV, CELL_LINE, K, SEED

EMBEDDINGS_CSV = "pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv"
MODEL_PT = "pimogp/data/joint_surface_model_bliss_curve_embedding_kpl1_leave_drugs_out.pt"
HELD_OUT_DRUG_FRAC = 0.25
N_PAIRS_TO_PLOT = 6

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_CSV, sep=";")
    kpl1 = df[df["cell_line"] == CELL_LINE].copy()

    conc_levels = sorted(kpl1["drugA_conc"].round(6).unique())
    level_to_idx = {lvl: i for i, lvl in enumerate(conc_levels)}
    n_levels = len(conc_levels)
    kpl1["cell_idx"] = (
        kpl1["drugA_conc"].round(6).map(level_to_idx) * n_levels
        + kpl1["drugB_conc"].round(6).map(level_to_idx)
    )
    num_cells = n_levels * n_levels

    embed_df = pd.read_csv(EMBEDDINGS_CSV).set_index("Name")
    z_cols = [c for c in embed_df.columns if c.startswith("z")]
    drug_embed_ci = embed_df[z_cols].copy()
    drug_embed_ci.index = drug_embed_ci.index.str.upper()
    drug_embed_ci = drug_embed_ci[~drug_embed_ci.index.duplicated(keep="first")]

    kpl1["drugA_ci"] = kpl1["drugA"].str.upper()
    kpl1["drugB_ci"] = kpl1["drugB"].str.upper()
    embed_A = drug_embed_ci.add_prefix("A_")
    embed_B = drug_embed_ci.add_prefix("B_")
    kpl1 = kpl1.join(embed_A, on="drugA_ci").join(embed_B, on="drugB_ci")
    A_cols, B_cols = list(embed_A.columns), list(embed_B.columns)
    kpl1 = kpl1.dropna(subset=A_cols + B_cols).reset_index(drop=True)

    all_drugs = sorted(set(kpl1["drugA"]) | set(kpl1["drugB"]))
    rng = np.random.default_rng(SEED)
    n_held_out = max(1, int(len(all_drugs) * HELD_OUT_DRUG_FRAC))
    held_out_drugs = set(rng.choice(all_drugs, size=n_held_out, replace=False))

    both_unseen_mask = kpl1["drugA"].isin(held_out_drugs) & kpl1["drugB"].isin(held_out_drugs)
    both_unseen_pairs = kpl1[both_unseen_mask][["drugA", "drugB"]].drop_duplicates().reset_index(drop=True)
    print(f"{len(both_unseen_pairs)} pairs with BOTH drugs entirely unseen during training")

    model = JointSurfaceModelBliss(num_cells=num_cells, drug_dim=len(z_cols), K=K)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()

    all_cells = torch.arange(num_cells, dtype=torch.long)
    conc_grid = torch.tensor(conc_levels, dtype=torch.float32)
    concA_grid = conc_grid.repeat_interleave(n_levels)   # matches cell_idx = A*n_levels + B
    concB_grid = conc_grid.repeat(n_levels)

    plot_rng = np.random.default_rng(0)
    n_plot = min(N_PAIRS_TO_PLOT, len(both_unseen_pairs))
    chosen = plot_rng.choice(len(both_unseen_pairs), size=n_plot, replace=False)

    fig, axes = plt.subplots(3, n_plot, figsize=(3.2 * n_plot, 9))
    for col, idx in enumerate(chosen):
        drugA, drugB = both_unseen_pairs.iloc[idx]
        subset = kpl1[(kpl1["drugA"] == drugA) & (kpl1["drugB"] == drugB)].sort_values("cell_idx")

        true_grid = np.full(num_cells, np.nan)
        true_grid[subset["cell_idx"].values] = subset["fMean"].values
        true_grid = true_grid.reshape(n_levels, n_levels)

        zA = torch.tensor(subset[A_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        zB = torch.tensor(subset[B_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        with torch.no_grad():
            pred = model(all_cells, concA_grid, concB_grid, zA, zB).numpy().reshape(n_levels, n_levels)

        vmin, vmax = min(np.nanmin(true_grid), pred.min()), max(np.nanmax(true_grid), pred.max())

        ax = axes[0, col]
        im = ax.imshow(true_grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{drugA}\n+ {drugB}\n(both unseen)\nTrue", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, col]
        im = ax.imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title("Predicted", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[2, col]
        resid = pred - true_grid
        m = np.nanmax(np.abs(resid))
        im = ax.imshow(resid, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set_title("Predicted - True", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Leave-drugs-out test pairs (BOTH drugs never seen in training) — Bliss model, curve embedding", fontsize=12)
    plt.tight_layout()
    plt.savefig("bliss_curve_leave_drugs_out_reconstructions.png", dpi=150)
    print("Saved bliss_curve_leave_drugs_out_reconstructions.png")

    # --- Phi basis functions, reshaped over the concentration grid ---
    Phi = model.Phi.detach().numpy()  # (num_cells, K)
    fig2, axes2 = plt.subplots(1, K, figsize=(3.2 * K, 3.4))
    for k in range(K):
        basis_k = Phi[:, k].reshape(n_levels, n_levels)
        im = axes2[k].imshow(basis_k, origin="lower", cmap="RdBu_r",
                              vmin=-np.abs(Phi[:, k]).max(), vmax=np.abs(Phi[:, k]).max())
        axes2[k].set_title(f"Φ basis {k+1}", fontsize=10)
        axes2[k].set_xlabel("drugB conc level")
        if k == 0:
            axes2[k].set_ylabel("drugA conc level")
        plt.colorbar(im, ax=axes2[k], fraction=0.046)
    fig2.suptitle("Learnt synergy-residual basis functions Φ (K=5), Bliss model with curve embedding", fontsize=12)
    plt.tight_layout()
    plt.savefig("bliss_curve_phi_basis_functions.png", dpi=150)
    print("Saved bliss_curve_phi_basis_functions.png")
