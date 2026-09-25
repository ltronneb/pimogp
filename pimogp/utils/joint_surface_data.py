"""
Shared data loading, embedding/one-hot feature construction, and
pair-level / leave-drugs-out splitting for the joint surface models
(train_additive.py, train_bliss.py).
"""
import numpy as np
import pandas as pd
import torch

PROCESSED_CSV = "pimogp/data/ONeil/processed.csv"
CELL_LINE = "KPL1"
K = 5
EPOCHS = 300
BATCH_SIZE = 512
SEED = 42
HELD_OUT_DRUG_FRAC = 0.25

EMBEDDINGS_CSV = {
    "old_embedding": "pimogp/data/cancer_drugs_efficacy_embeddings_kpl1.csv",
    "curve_embedding": "pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv",
}


def load_kpl1_grid():
    """Load processed.csv KPL1 rows and attach a cell_idx column (0..99)."""
    df = pd.read_csv(PROCESSED_CSV, sep=";")
    kpl1 = df[df["cell_line"] == CELL_LINE].copy()

    conc_levels = sorted(kpl1["drugA_conc"].round(6).unique())
    level_to_idx = {lvl: i for i, lvl in enumerate(conc_levels)}
    n_levels = len(conc_levels)
    kpl1["cell_idx"] = (
        kpl1["drugA_conc"].round(6).map(level_to_idx) * n_levels
        + kpl1["drugB_conc"].round(6).map(level_to_idx)
    )
    return kpl1, conc_levels, n_levels, n_levels * n_levels


def load_drug_features(input_type, kpl1):
    """
    input_type: "old_embedding" | "curve_embedding" | "onehot"

    Returns (kpl1_filtered, meta, drug_dim):
      - kpl1_filtered: rows restricted to the drug pool covered by this input type
      - meta: (A_cols, B_cols) column-name tuple for embedding inputs (pull features
        directly via kpl1[A_cols]/kpl1[B_cols]), or None for one-hot (use
        kpl1.attrs["drug_to_idx"] + to_onehot_tensor() instead)
      - drug_dim: dimensionality of the per-drug feature vector
    """
    if input_type in EMBEDDINGS_CSV:
        embed_df = pd.read_csv(EMBEDDINGS_CSV[input_type]).set_index("Name")
        z_cols = [c for c in embed_df.columns if c.startswith("z")]
        drug_embed_ci = embed_df[z_cols].copy()
        drug_embed_ci.index = drug_embed_ci.index.str.upper()
        drug_embed_ci = drug_embed_ci[~drug_embed_ci.index.duplicated(keep="first")]

        kpl1 = kpl1.copy()
        kpl1["drugA_ci"] = kpl1["drugA"].str.upper()
        kpl1["drugB_ci"] = kpl1["drugB"].str.upper()
        embed_A = drug_embed_ci.add_prefix("A_")
        embed_B = drug_embed_ci.add_prefix("B_")
        kpl1 = kpl1.join(embed_A, on="drugA_ci").join(embed_B, on="drugB_ci")
        A_cols, B_cols = list(embed_A.columns), list(embed_B.columns)
        kpl1 = kpl1.dropna(subset=A_cols + B_cols).reset_index(drop=True)

        return kpl1, (A_cols, B_cols), len(z_cols)

    elif input_type == "onehot":
        kpl1 = kpl1.copy()
        embed_df = pd.read_csv(EMBEDDINGS_CSV["curve_embedding"]).set_index("Name")
        drug_names_ci = embed_df.index.str.upper()
        drug_names_ci = drug_names_ci[~drug_names_ci.duplicated()]
        valid_ci = set(drug_names_ci)

        kpl1["drugA_ci"] = kpl1["drugA"].str.upper()
        kpl1["drugB_ci"] = kpl1["drugB"].str.upper()
        kpl1 = kpl1[kpl1["drugA_ci"].isin(valid_ci) & kpl1["drugB_ci"].isin(valid_ci)].reset_index(drop=True)

        all_drugs = sorted(set(kpl1["drugA"]) | set(kpl1["drugB"]))
        drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
        onehot_dim = len(all_drugs)
        kpl1.attrs["drug_to_idx"] = drug_to_idx
        return kpl1, None, onehot_dim

    else:
        raise ValueError(f"unknown input_type: {input_type}")


def to_onehot_tensor(names, drug_to_idx, onehot_dim):
    X = torch.zeros(len(names), onehot_dim)
    for i, n in enumerate(names):
        X[i, drug_to_idx[n]] = 1.0
    return X


def split_pairs(kpl1, split_type):
    """
    split_type: "pair_level" | "leave_drugs_out"
    Returns (train_df, test_df).
    """
    rng = np.random.default_rng(SEED)

    if split_type == "pair_level":
        pairs = kpl1[["drugA", "drugB"]].drop_duplicates().reset_index(drop=True)
        n_pairs = len(pairs)
        test_pair_idx = rng.choice(n_pairs, size=int(0.2 * n_pairs), replace=False)
        test_pairs = set(map(tuple, pairs.iloc[test_pair_idx][["drugA", "drugB"]].values))
        is_test = kpl1.apply(lambda r: (r["drugA"], r["drugB"]) in test_pairs, axis=1)
        return kpl1[~is_test].reset_index(drop=True), kpl1[is_test].reset_index(drop=True)

    elif split_type == "leave_drugs_out":
        all_drugs = sorted(set(kpl1["drugA"]) | set(kpl1["drugB"]))
        n_held_out = max(1, int(len(all_drugs) * HELD_OUT_DRUG_FRAC))
        held_out_drugs = set(rng.choice(all_drugs, size=n_held_out, replace=False))
        seen_drugs = set(all_drugs) - held_out_drugs
        both_seen = kpl1["drugA"].isin(seen_drugs) & kpl1["drugB"].isin(seen_drugs)
        return kpl1[both_seen].reset_index(drop=True), kpl1[~both_seen].reset_index(drop=True)

    else:
        raise ValueError(f"unknown split_type: {split_type}")
