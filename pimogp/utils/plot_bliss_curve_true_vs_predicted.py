"""
True vs. predicted viability scatter, side by side for train (both drugs
seen) vs test (>=1 drug unseen) -- best model of the session: Bliss
joint model, curve-trained embedding, leave-drugs-out split.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pimogp.models.joint_surface_models import JointSurfaceModelBliss
from pimogp.utils.joint_surface_data import PROCESSED_CSV, CELL_LINE, K, SEED

EMBEDDINGS_CSV = "pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv"
MODEL_PT = "pimogp/data/joint_surface_model_bliss_curve_embedding_kpl1_leave_drugs_out.pt"
HELD_OUT_DRUG_FRAC = 0.25

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
    seen_drugs = set(all_drugs) - held_out_drugs

    both_seen = kpl1["drugA"].isin(seen_drugs) & kpl1["drugB"].isin(seen_drugs)
    both_unseen = kpl1["drugA"].isin(held_out_drugs) & kpl1["drugB"].isin(held_out_drugs)
    train_df = kpl1[both_seen].reset_index(drop=True)
    test_df = kpl1[~both_seen].reset_index(drop=True)

    model = JointSurfaceModelBliss(num_cells=num_cells, drug_dim=len(z_cols), K=K)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()

    def predict(d):
        cell = torch.tensor(d["cell_idx"].values, dtype=torch.long)
        cA = torch.tensor(d["drugA_conc"].values, dtype=torch.float32)
        cB = torch.tensor(d["drugB_conc"].values, dtype=torch.float32)
        xa = torch.tensor(d[A_cols].values, dtype=torch.float32)
        xb = torch.tensor(d[B_cols].values, dtype=torch.float32)
        y = torch.tensor(d["fMean"].values, dtype=torch.float32)
        with torch.no_grad():
            pred = model(cell, cA, cB, xa, xb)
        mse = F.mse_loss(pred, y).item()
        r = np.corrcoef(y.numpy(), pred.numpy())[0, 1]
        return y.numpy(), pred.numpy(), mse, r

    y_train, pred_train, mse_train, r_train = predict(train_df)
    y_test, pred_test, mse_test, r_test = predict(test_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    axes[0].scatter(y_train, pred_train, s=4, alpha=0.15, c="tab:blue")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    axes[0].set_title(f"Train (both drugs seen)\nn={len(train_df)}, MSE={mse_train:.4f}, r={r_train:.3f}")
    axes[0].set_xlabel("True viability"); axes[0].set_ylabel("Predicted viability")

    axes[1].scatter(y_test, pred_test, s=4, alpha=0.15, c="tab:red")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    axes[1].set_title(f"Test ($\\geq$1 drug unseen)\nn={len(test_df)}, MSE={mse_test:.4f}, r={r_test:.3f}")
    axes[1].set_xlabel("True viability")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.suptitle("True vs. predicted viability — Bliss model, curve embedding, leave-drugs-out split", fontsize=13)
    plt.tight_layout()
    plt.savefig("bliss_curve_true_vs_predicted_train_test.png", dpi=150)
    print("Saved bliss_curve_true_vs_predicted_train_test.png")
    print(f"train: n={len(train_df)} MSE={mse_train:.5f} r={r_train:.4f}")
    print(f"test:  n={len(test_df)} MSE={mse_test:.5f} r={r_test:.4f}")
