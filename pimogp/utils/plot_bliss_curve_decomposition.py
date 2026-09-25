"""
Decomposes the predicted surface into its two additive components for a
few example (held-out-drug) pairs: the Bliss product term
curve(cA,zA)*curve(cB,zB), and the synergy-residual term Phi[cell] . g(zA,zB),
alongside the true surface and their sum (the full prediction).
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
N_PAIRS_TO_PLOT = 3

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

    model = JointSurfaceModelBliss(num_cells=num_cells, drug_dim=len(z_cols), K=K)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()

    all_cells = torch.arange(num_cells, dtype=torch.long)
    conc_grid = torch.tensor(conc_levels, dtype=torch.float32)
    concA_grid = conc_grid.repeat_interleave(n_levels)
    concB_grid = conc_grid.repeat(n_levels)

    plot_rng = np.random.default_rng(0)
    n_plot = min(N_PAIRS_TO_PLOT, len(both_unseen_pairs))
    chosen = plot_rng.choice(len(both_unseen_pairs), size=n_plot, replace=False)

    fig, axes = plt.subplots(n_plot, 4, figsize=(15, 3.6 * n_plot))
    if n_plot == 1:
        axes = axes[None, :]

    for row, idx in enumerate(chosen):
        drugA, drugB = both_unseen_pairs.iloc[idx]
        subset = kpl1[(kpl1["drugA"] == drugA) & (kpl1["drugB"] == drugB)].sort_values("cell_idx")

        true_grid = np.full(num_cells, np.nan)
        true_grid[subset["cell_idx"].values] = subset["fMean"].values
        true_grid = true_grid.reshape(n_levels, n_levels)

        zA = torch.tensor(subset[A_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        zB = torch.tensor(subset[B_cols].iloc[0].values, dtype=torch.float32).unsqueeze(0).repeat(num_cells, 1)
        with torch.no_grad():
            bliss_term = (model.curve(concA_grid, zA) * model.curve(concB_grid, zB)).numpy().reshape(n_levels, n_levels)
            g_out = model.g(zA, zB)
            phi_out = model.Phi[all_cells]
            resid_term = (phi_out * g_out).sum(dim=-1).numpy().reshape(n_levels, n_levels)
            pred = bliss_term + resid_term

        panels = [
            (true_grid, "True surface", "viridis", None),
            (bliss_term, "Bliss term\ncurve(cA,zA)·curve(cB,zB)", "viridis", None),
            (resid_term, "Residual term\nΦ·g(zA,zB)", "RdBu_r", np.nanmax(np.abs(resid_term))),
            (pred, "Predicted\n(Bliss + residual)", "viridis", None),
        ]
        for col, (grid, title, cmap, sym) in enumerate(panels):
            ax = axes[row, col]
            if sym is not None:
                im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=-sym, vmax=sym)
            else:
                im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{drugA}\n+ {drugB}\n(both unseen)", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Decomposition of the predicted surface: Bliss product term + synergy-residual term", fontsize=13)
    plt.tight_layout()
    plt.savefig("bliss_curve_decomposition.png", dpi=150)
    print("Saved bliss_curve_decomposition.png")
