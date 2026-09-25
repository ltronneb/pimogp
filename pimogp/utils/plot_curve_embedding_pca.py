"""
2D PCA projection of the curve-trained drug embedding, for all 7449 cancer
drugs, shaded by the real ONeil monotherapy curve supervision for the 35
drugs that have one -- two panels: depth (curve PC1, viability at max dose)
and shape (curve PC2, the dominant shape-beyond-depth direction).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EMBEDDINGS_CSV = "pimogp/data/cancer_drugs_curve_embeddings_kpl1.csv"
PROCESSED_CSV = "pimogp/data/ONeil/processed.csv"

embed_df = pd.read_csv(EMBEDDINGS_CSV)
z_cols = [c for c in embed_df.columns if c.startswith("z")]
Z = embed_df[z_cols].values

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)
embed_df["pc1"], embed_df["pc2"] = Z_pca[:, 0], Z_pca[:, 1]

df = pd.read_csv(PROCESSED_CSV, sep=";")
kpl1 = df[df["cell_line"] == "KPL1"]
edge_A = kpl1[kpl1["drugB_conc"] == 0][["drugA", "drugA_conc", "fMean"]].rename(columns={"drugA": "drug", "drugA_conc": "conc"})
edge_B = kpl1[kpl1["drugA_conc"] == 0][["drugB", "drugB_conc", "fMean"]].rename(columns={"drugB": "drug", "drugB_conc": "conc"})
edges = pd.concat([edge_A, edge_B], ignore_index=True)
curve_table = edges.groupby(["drug", "conc"])["fMean"].mean().unstack("conc").sort_index(axis=1).dropna()
curve_table.index = curve_table.index.str.upper()

curve_pca = PCA(n_components=2)
curve_pcs = curve_pca.fit_transform(curve_table.values)
depth = pd.Series(curve_pcs[:, 0], index=curve_table.index, name="depth")
shape = pd.Series(curve_pcs[:, 1], index=curve_table.index, name="shape")

embed_df["Name_ci"] = embed_df["Name"].str.upper()
embed_df = embed_df.set_index("Name_ci")
embed_df["depth"] = depth
embed_df["shape"] = shape
labeled = embed_df["depth"].notna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, title in zip(axes, ["depth", "shape"],
                           ["curve PC1 (depth, 79.4% var)", "curve PC2 (shape, 16.5% var)"]):
    ax.scatter(embed_df.loc[~labeled, "pc1"], embed_df.loc[~labeled, "pc2"],
               c="lightgray", s=6, alpha=0.35, label=f"unlabeled (n={(~labeled).sum()})")
    sc = ax.scatter(embed_df.loc[labeled, "pc1"], embed_df.loc[labeled, "pc2"],
                     c=embed_df.loc[labeled, col], cmap="viridis", s=45, edgecolor="k", linewidth=0.4,
                     label=f"ONeil-labeled (n={labeled.sum()})")
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_xlabel(f"Embedding PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"Embedding PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"Shaded by {title}")
    ax.legend(loc="best", fontsize=8)

fig.suptitle("Curve-trained embedding: PCA projection shaded by real ONeil monotherapy curve summaries", fontsize=13)
plt.tight_layout()
plt.savefig("curve_embedding_pca_depth_shape.png", dpi=150)
print("Saved curve_embedding_pca_depth_shape.png")
