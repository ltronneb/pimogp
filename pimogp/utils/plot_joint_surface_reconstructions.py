"""
Loads the trained JointSurfaceModel and plots true vs. predicted KPL1
surfaces for a handful of held-out (unseen-during-training) test pairs.
Rebuilds the identical data pipeline / train-test split as
train_additive.py --split pair_level so the same pairs land in the test set.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pimogp.models.joint_surface_models import JointSurfaceModel
from pimogp.utils.joint_surface_data import EMBEDDINGS_CSV, PROCESSED_CSV, CELL_LINE, K, SEED

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

    embed_df = pd.read_csv(EMBEDDINGS_CSV["old_embedding"]).set_index("Name")
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

    pairs = kpl1[["drugA", "drugB"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    n_pairs = len(pairs)
    test_pair_idx = rng.choice(n_pairs, size=int(0.2 * n_pairs), replace=False)
    test_pairs = pairs.iloc[test_pair_idx][["drugA", "drugB"]].reset_index(drop=True)

    model = JointSurfaceModel(num_cells=num_cells, drug_dim=len(z_cols), K=K, mean_init=torch.zeros(num_cells))
    model.load_state_dict(torch.load("pimogp/data/joint_surface_model_kpl1.pt", map_location="cpu"))
    model.eval()

    all_cells = torch.arange(num_cells, dtype=torch.long)

    plot_rng = np.random.default_rng(0)
    chosen = plot_rng.choice(len(test_pairs), size=min(N_PAIRS_TO_PLOT, len(test_pairs)), replace=False)

    fig, axes = plt.subplots(3, len(chosen), figsize=(3.2 * len(chosen), 9))
    for col, idx in enumerate(chosen):
        drugA, drugB = test_pairs.iloc[idx]
        subset = kpl1[(kpl1["drugA"] == drugA) & (kpl1["drugB"] == drugB)].sort_values("cell_idx")

        true_grid = np.full(num_cells, np.nan)
        true_grid[subset["cell_idx"].values] = subset["fMean"].values
        true_grid = true_grid.reshape(n_levels, n_levels)

        zA = torch.tensor(subset[A_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        zB = torch.tensor(subset[B_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        with torch.no_grad():
            pred = model(all_cells, zA, zB).numpy().reshape(n_levels, n_levels)

        vmin, vmax = min(np.nanmin(true_grid), pred.min()), max(np.nanmax(true_grid), pred.max())

        ax = axes[0, col]
        im = ax.imshow(true_grid, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{drugA}\n+ {drugB}\nTrue", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, col]
        im = ax.imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title("Predicted", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[2, col]
        resid = pred - true_grid
        im = ax.imshow(resid, origin="lower", cmap="RdBu_r", vmin=-abs(resid).max(), vmax=abs(resid).max())
        ax.set_title("Predicted - True", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Held-out test pairs: joint model surface reconstruction (KPL1)", fontsize=13)
    plt.tight_layout()
    plt.savefig("joint_model_test_reconstructions.png", dpi=150)
    print("Saved joint_model_test_reconstructions.png")
