"""
Bliss-independence joint surface model:
  Yhat(cell; zA, zB) = curve(cA, zA) . curve(cB, zB) + Phi[cell] . g(zA, zB)
See EFFICACY_EMBEDDING_EXPERIMENTS.md section 6 for the full derivation.
Best-performing model of the study (see section 13: 0.00591 test MSE on
leave-drugs-out with the curve embedding).

Usage:
  python train_bliss.py --split pair_level --input curve_embedding
  python train_bliss.py --split leave_drugs_out --input onehot
  python train_bliss.py --split leave_drugs_out --input old_embedding
"""
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pimogp.models.joint_surface_models import JointSurfaceModelBliss
from pimogp.utils.joint_surface_data import (
    load_kpl1_grid, load_drug_features, split_pairs, to_onehot_tensor,
    K, EPOCHS, BATCH_SIZE,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["pair_level", "leave_drugs_out"], required=True)
    parser.add_argument("--input", choices=["old_embedding", "curve_embedding", "onehot"], required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)

    kpl1, conc_levels, n_levels, num_cells = load_kpl1_grid()
    kpl1, meta, drug_dim = load_drug_features(args.input, kpl1)
    print(f"{len(kpl1)} rows, {kpl1[['drugA', 'drugB']].drop_duplicates().shape[0]} pairs, "
          f"input={args.input} (drug_dim={drug_dim})")

    train_df, test_df = split_pairs(kpl1, args.split)
    print(f"{len(train_df)} train rows, {len(test_df)} test rows")

    def to_tensors(d):
        cell = torch.tensor(d["cell_idx"].values, dtype=torch.long)
        cA = torch.tensor(d["drugA_conc"].values, dtype=torch.float32)
        cB = torch.tensor(d["drugB_conc"].values, dtype=torch.float32)
        y = torch.tensor(d["fMean"].values, dtype=torch.float32)
        if args.input == "onehot":
            drug_to_idx = kpl1.attrs["drug_to_idx"]
            xa = to_onehot_tensor(d["drugA"].values, drug_to_idx, drug_dim)
            xb = to_onehot_tensor(d["drugB"].values, drug_to_idx, drug_dim)
        else:
            A_cols, B_cols = meta
            xa = torch.tensor(d[A_cols].values, dtype=torch.float32)
            xb = torch.tensor(d[B_cols].values, dtype=torch.float32)
        return cell, cA, cB, xa, xb, y

    cell_train, cA_train, cB_train, XA_train, XB_train, y_train = to_tensors(train_df)
    cell_test, cA_test, cB_test, XA_test, XB_test, y_test = to_tensors(test_df)

    model = JointSurfaceModelBliss(num_cells=num_cells, drug_dim=drug_dim, K=K)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=15)
    loader = DataLoader(TensorDataset(cell_train, cA_train, cB_train, XA_train, XB_train, y_train),
                         batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for cb, ca_, cb_, xa, xb, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(cb, ca_, cb_, xa, xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(model(cell_train, cA_train, cB_train, XA_train, XB_train), y_train).item()
            test_mse = F.mse_loss(model(cell_test, cA_test, cB_test, XA_test, XB_test), y_test).item()
            bliss_only_test_mse = F.mse_loss(
                model.curve(cA_test, XA_test) * model.curve(cB_test, XB_test), y_test).item()
        scheduler.step(test_mse)
        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d} | train MSE: {train_mse:.5f}  test MSE: {test_mse:.5f}  "
                  f"(Bliss-only test MSE: {bliss_only_test_mse:.5f})  lr: {lr:.2e}")

    out_path = f"pimogp/data/joint_surface_bliss_{args.split}_{args.input}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")
