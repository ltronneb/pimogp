"""
Retrains the efficacy embedding using real ONeil KPL1 monotherapy dose-
response curves as supervision, instead of the single-scalar "KPL1"
column in drugs.csv -- which was found not to correlate with ONeil's own
KPL1 screen at all (r=-0.13, some drugs actively rank oppositely; see
EFFICACY_EMBEDDING_EXPERIMENTS.md). The curve data comes from the
monotherapy "edge" of ONeil's combo screen (rows where the partner drug's
concentration is 0) in pimogp/data/ONeil/processed.csv -- ~22-37 replicate
measurements per (drug, dose) pooled across different combo partners,
~3% cross-partner noise, so a reliable per-drug curve.

Same architecture/training philosophy as train_efficacy_embedding.py:
SELFIES autoencoder (reconstruction on all 7449 cancer drugs) + auxiliary
head off the same bottleneck z, trained jointly, no held-out split for the
auxiliary loss (the point is embedding geometry). The only change: the
head now predicts the full 10-point curve (SelfiesCurveAutoencoder)
instead of one scalar, and the supervision comes from real ONeil data
(~38 drugs -- every unique drug in the KPL1 screen -- rather than the 309
drugs.csv had labels for, since only drugs actually screened by ONeil can
have a curve).
"""
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from pimogp.utils.one_hot_encoding import multiple_selfies_to_int
from pimogp.utils.transformer_vae_evals import compute_recon_quality
from pimogp.models.efficacy_embedding import SelfiesCurveAutoencoder

CANCER_DRUGS_CSV = "pimogp/data/cancer_drugs.csv"
FINAL_DRUGS_CSV = "pimogp/data/final_drugs.csv"
PROCESSED_CSV = "pimogp/data/ONeil/processed.csv"
CELL_LINE = "KPL1"

EMBED_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 128
Z_DIM = 32
EPOCHS = 60
BATCH_SIZE = 128
CURVE_LOSS_WEIGHT = 1.0
SEED = 42

if __name__ == "__main__":
    torch.set_num_threads(8)
    device = torch.device("cpu")

    # ---- Build per-drug monotherapy curves from real ONeil KPL1 data ----
    df = pd.read_csv(PROCESSED_CSV, sep=";")
    kpl1 = df[df["cell_line"] == CELL_LINE]
    edge_A = kpl1[kpl1["drugB_conc"] == 0][["drugA", "drugA_conc", "fMean"]].rename(
        columns={"drugA": "drug", "drugA_conc": "conc"})
    edge_B = kpl1[kpl1["drugA_conc"] == 0][["drugB", "drugB_conc", "fMean"]].rename(
        columns={"drugB": "drug", "drugB_conc": "conc"})
    edges = pd.concat([edge_A, edge_B], ignore_index=True)
    curve_table = edges.groupby(["drug", "conc"])["fMean"].mean().unstack("conc").sort_index(axis=1)
    curve_table = curve_table.dropna()  # keep only drugs with a full curve
    conc_levels = curve_table.columns.values
    n_conc_levels = len(conc_levels)
    curve_table.index = curve_table.index.str.upper()
    print(f"{len(curve_table)} drugs have a full {n_conc_levels}-point KPL1 monotherapy curve")

    # ---- Cancer drug corpus (same as train_efficacy_embedding.py) ----
    cancer_drugs = pd.read_csv(CANCER_DRUGS_CSV)
    final_drugs = pd.read_csv(FINAL_DRUGS_CSV)
    names = final_drugs.iloc[cancer_drugs["index"]]["Name"].reset_index(drop=True)

    drug_df = pd.DataFrame({"smiles": cancer_drugs["smiles"], "Name": names})
    drug_df["Name_ci"] = drug_df["Name"].str.upper()

    selfies_list, keep_idx = [], []
    for i, s in enumerate(drug_df["smiles"]):
        try:
            selfies_list.append(sf.encoder(s))
            keep_idx.append(i)
        except Exception:
            continue
    drug_df = drug_df.iloc[keep_idx].reset_index(drop=True)

    curve_targets = curve_table.reindex(drug_df["Name_ci"]).values  # (N, n_conc_levels), NaN where unlabeled
    labeled_mask = ~np.isnan(curve_targets).any(axis=1)
    print(f"{len(drug_df)} / {len(cancer_drugs)} cancer drugs encoded to SELFIES "
          f"({labeled_mask.sum()} labeled with a real KPL1 monotherapy curve)")

    alphabet = sorted(sf.get_alphabet_from_selfies(selfies_list) | {"[sos]", "[eos]", "[nop]"})
    largest_len = max(sf.len_selfies(s) for s in selfies_list)
    int_data = multiple_selfies_to_int(selfies_list, largest_len, alphabet)

    pad_idx = alphabet.index("[nop]")
    vocab_size = len(alphabet)

    y_curve = np.nan_to_num(curve_targets, nan=0.0).astype(np.float32)

    rng = np.random.default_rng(SEED)
    N = len(drug_df)
    test_idx = rng.choice(N, size=int(0.2 * N), replace=False)
    train_idx = np.setdiff1d(np.arange(N), test_idx)
    X_train = int_data[train_idx].to(device)
    X_test = int_data[test_idx].to(device)

    X_curve = int_data[labeled_mask].to(device)
    y_curve_labeled = torch.tensor(y_curve[labeled_mask], device=device)

    model = SelfiesCurveAutoencoder(
        vocab_size=vocab_size, embed_size=EMBED_SIZE, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, z_dim=Z_DIM, max_seq_len=largest_len + 2,
        n_conc_levels=n_conc_levels,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

    recon_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    curve_batch_size = min(32, len(X_curve))
    curve_loader = DataLoader(TensorDataset(X_curve, y_curve_labeled), batch_size=curve_batch_size, shuffle=True)

    def recon_step(xb):
        logits, z, _, x_target = model(xb, pad_idx)
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * S, V), x_target.reshape(B * S), ignore_index=pad_idx)
        acc = compute_recon_quality(x_target, logits.argmax(dim=-1), pad_idx)
        return loss, acc

    def curve_step(xb, yb):
        _, _, curve_pred, _ = model(xb, pad_idx)
        return F.mse_loss(curve_pred, yb)

    import itertools
    for epoch in range(EPOCHS):
        model.train()
        curve_iter = itertools.cycle(curve_loader)
        for xb, in recon_loader:
            xb_c, yb_c = next(curve_iter)
            optimizer.zero_grad()
            recon_loss, _ = recon_step(xb)
            curve_loss = curve_step(xb_c, yb_c)
            (recon_loss + CURVE_LOSS_WEIGHT * curve_loss).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_recon, train_acc = recon_step(X_train)
            test_recon, test_acc = recon_step(X_test)
            curve_loss_full = curve_step(X_curve, y_curve_labeled)
        scheduler.step(test_recon)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train recon: {train_recon:.4f} acc: {train_acc:.2f}%  "
              f"| test recon: {test_recon:.4f} acc: {test_acc:.2f}%  | curve (all n={labeled_mask.sum()}): {curve_loss_full:.4f}  lr: {lr:.2e}")

    torch.save(model.state_dict(), "pimogp/data/curve_embedding_kpl1.pt")
    print("Saved model to pimogp/data/curve_embedding_kpl1.pt")

    model.eval()
    with torch.no_grad():
        _, _, curve_pred, _ = model(X_curve, pad_idx)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_curve_labeled.cpu().numpy().flatten(), curve_pred.cpu().numpy().flatten(), alpha=0.3)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("True KPL1 viability (monotherapy curve point)")
    plt.ylabel("Predicted")
    plt.title("Curve embedding — regression head fit (all labeled drugs, all conc levels)")
    plt.savefig("curve_embedding_kpl1_fit.png")
    plt.close()

    model.eval()
    with torch.no_grad():
        Z_all = model.encode(int_data.to(device))
    embed_cols = [f"z{i}" for i in range(Z_DIM)]
    embed_df = pd.DataFrame(Z_all.cpu().numpy(), columns=embed_cols)
    out_df = pd.concat([drug_df[["Name", "smiles"]].reset_index(drop=True), embed_df], axis=1)
    out_df.to_csv("pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv", index=False)
    print(f"Saved embeddings for {len(out_df)} drugs to pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv")
