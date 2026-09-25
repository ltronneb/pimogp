"""
Trains a small, non-generative SELFIES autoencoder on the cancer drug corpus
(pimogp/data/cancer_drugs.csv), with an auxiliary KPL1 monotherapy efficacy
regression head off the same bottleneck z, so the embedding is shaped by
potency as well as by molecular identity.

Drug identity for cancer_drugs.csv rows is recovered via final_drugs.csv's
'index' column (see the fix in surface_embedding_and_prediction.py -- the old
code used the row's own pandas position instead of this column, which
misaligned nearly every drug name). PubchemID is then used to join against
pimogp/data/drugs_oneil_subset.csv for the KPL1 efficacy label, since only
309 of the ~7450 cancer drugs have monotherapy data.

The point of this model is the embedding itself, not held-out efficacy
prediction, so all 309 labeled drugs drive the efficacy loss every epoch.
Only the reconstruction objective keeps a train/test split, to monitor
whether the autoencoder generalizes at all.
"""
import itertools

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
from pimogp.models.efficacy_embedding import SelfiesEfficacyAutoencoder

CANCER_DRUGS_CSV = "pimogp/data/cancer_drugs.csv"
FINAL_DRUGS_CSV = "pimogp/data/final_drugs.csv"
DRUGS_ONEIL_CSV = "pimogp/data/drugs_oneil_subset.csv"
CELL_LINE = "KPL1"

EMBED_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 128
Z_DIM = 32
EPOCHS = 60
BATCH_SIZE = 128
EFFICACY_BATCH_SIZE = 64
EFFICACY_LOSS_WEIGHT = 1.0
SEED = 42

if __name__ == "__main__":
    torch.set_num_threads(8)  # this is a shared 256-core box; avoid thrashing other users' jobs
    device = torch.device("cpu")  # GPU on this shared box is contended by other users' jobs

    cancer_drugs = pd.read_csv(CANCER_DRUGS_CSV)
    final_drugs = pd.read_csv(FINAL_DRUGS_CSV)
    names = final_drugs.iloc[cancer_drugs["index"]]["Name"].reset_index(drop=True)
    pubchem_ids = final_drugs.iloc[cancer_drugs["index"]]["PubchemID"].reset_index(drop=True)

    drugs_oneil = pd.read_csv(DRUGS_ONEIL_CSV)
    kpl1_by_pubchem = dict(zip(drugs_oneil["PubchemID"], drugs_oneil[CELL_LINE]))

    df = pd.DataFrame({
        "smiles": cancer_drugs["smiles"],
        "Name": names,
        "PubchemID": pubchem_ids,
    })
    df[CELL_LINE] = df["PubchemID"].map(kpl1_by_pubchem)

    # Encode to SELFIES, dropping any SMILES selfies can't handle
    selfies_list, keep_idx = [], []
    for i, s in enumerate(df["smiles"]):
        try:
            selfies_list.append(sf.encoder(s))
            keep_idx.append(i)
        except Exception:
            continue
    df = df.iloc[keep_idx].reset_index(drop=True)
    print(f"{len(df)} / {len(cancer_drugs)} cancer drugs encoded to SELFIES "
          f"({df[CELL_LINE].notna().sum()} labeled with {CELL_LINE} efficacy)")

    alphabet = sorted(sf.get_alphabet_from_selfies(selfies_list) | {"[sos]", "[eos]", "[nop]"})
    largest_len = max(sf.len_selfies(s) for s in selfies_list)
    int_data = multiple_selfies_to_int(selfies_list, largest_len, alphabet)  # [N, largest_len+2]

    pad_idx = alphabet.index("[nop]")
    vocab_size = len(alphabet)

    labeled_mask = df[CELL_LINE].notna().values
    y_mean, y_std = df.loc[labeled_mask, CELL_LINE].mean(), df.loc[labeled_mask, CELL_LINE].std()
    y_labeled = ((df.loc[labeled_mask, CELL_LINE].values - y_mean) / y_std).astype(np.float32)

    # Reconstruction: train/test split, purely to monitor whether the autoencoder generalizes
    rng = np.random.default_rng(SEED)
    N = len(df)
    test_idx = rng.choice(N, size=int(0.2 * N), replace=False)
    train_idx = np.setdiff1d(np.arange(N), test_idx)
    X_train = int_data[train_idx].to(device)
    X_test = int_data[test_idx].to(device)

    # Efficacy: every labeled drug is used for training, every epoch -- no held-out set,
    # since the point is the embedding geometry, not held-out efficacy prediction.
    X_eff = int_data[labeled_mask].to(device)
    y_eff = torch.tensor(y_labeled, device=device)

    model = SelfiesEfficacyAutoencoder(
        vocab_size=vocab_size, embed_size=EMBED_SIZE, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, z_dim=Z_DIM, max_seq_len=largest_len + 2,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

    recon_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    eff_loader = DataLoader(TensorDataset(X_eff, y_eff), batch_size=EFFICACY_BATCH_SIZE, shuffle=True)

    def recon_step(xb):
        logits, z, _, x_target = model(xb, pad_idx)
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * S, V), x_target.reshape(B * S), ignore_index=pad_idx)
        acc = compute_recon_quality(x_target, logits.argmax(dim=-1), pad_idx)
        return loss, acc

    def efficacy_step(xb, yb):
        _, _, efficacy_pred, _ = model(xb, pad_idx)
        return F.mse_loss(efficacy_pred, yb)

    for epoch in range(EPOCHS):
        model.train()
        eff_iter = itertools.cycle(eff_loader)
        for xb, in recon_loader:
            xb_eff, yb_eff = next(eff_iter)
            optimizer.zero_grad()
            recon_loss, _ = recon_step(xb)
            efficacy_loss = efficacy_step(xb_eff, yb_eff)
            (recon_loss + EFFICACY_LOSS_WEIGHT * efficacy_loss).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_recon, train_acc = recon_step(X_train)
            test_recon, test_acc = recon_step(X_test)
            eff_loss = efficacy_step(X_eff, y_eff)
        scheduler.step(test_recon)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | train recon: {train_recon:.4f} acc: {train_acc:.2f}%  "
              f"| test recon: {test_recon:.4f} acc: {test_acc:.2f}%  | eff (all n={labeled_mask.sum()}): {eff_loss:.4f}  lr: {lr:.2e}")

    torch.save(model.state_dict(), "pimogp/data/efficacy_embedding_kpl1.pt")
    print("Saved model to pimogp/data/efficacy_embedding_kpl1.pt")

    model.eval()
    with torch.no_grad():
        _, _, eff_pred, _ = model(X_eff, pad_idx)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_eff.cpu() * y_std + y_mean, eff_pred.cpu() * y_std + y_mean,
                alpha=0.5, label=f"all labeled (n={labeled_mask.sum()})")
    lims = [df.loc[labeled_mask, CELL_LINE].min(), df.loc[labeled_mask, CELL_LINE].max()]
    plt.plot(lims, lims, "k--", alpha=0.3)
    plt.xlabel("True KPL1 efficacy"); plt.ylabel("Predicted KPL1 efficacy")
    plt.title("SELFIES efficacy autoencoder — regression head fit (trained on all labeled)")
    plt.legend()
    plt.savefig("efficacy_embedding_kpl1_fit.png")
    plt.close()

    # Embed the full cancer drug panel (labeled and unlabeled) with the trained encoder
    model.eval()
    with torch.no_grad():
        Z_all = model.encode(int_data.to(device))
    embed_cols = [f"z{i}" for i in range(Z_DIM)]
    embed_df = pd.DataFrame(Z_all.cpu().numpy(), columns=embed_cols)
    out_df = pd.concat([df[["Name", "PubchemID", "smiles", CELL_LINE]].reset_index(drop=True), embed_df], axis=1)
    out_df.to_csv("pimogp/data/cancer_drugs_efficacy_embeddings_kpl1.csv", index=False)
    print(f"Saved embeddings for {len(out_df)} drugs to pimogp/data/cancer_drugs_efficacy_embeddings_kpl1.csv")
