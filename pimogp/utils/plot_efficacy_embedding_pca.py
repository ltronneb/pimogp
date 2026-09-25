import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("pimogp/data/cancer_drugs_efficacy_embeddings_kpl1.csv")
z_cols = [c for c in df.columns if c.startswith("z")]
Z = df[z_cols].values

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)
df["pc1"], df["pc2"] = Z_pca[:, 0], Z_pca[:, 1]

labeled = df["KPL1"].notna()

plt.figure(figsize=(8, 7))
plt.scatter(df.loc[~labeled, "pc1"], df.loc[~labeled, "pc2"],
            c="lightgray", s=8, alpha=0.4, label=f"unlabeled (n={(~labeled).sum()})")
sc = plt.scatter(df.loc[labeled, "pc1"], df.loc[labeled, "pc2"],
                  c=df.loc[labeled, "KPL1"], cmap="viridis", s=30, edgecolor="k", linewidth=0.3,
                  label=f"KPL1-labeled (n={labeled.sum()})")
plt.colorbar(sc, label="KPL1 efficacy")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of efficacy-embedding latents, shaded by KPL1 efficacy")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("efficacy_embedding_pca.png", dpi=150)
print("Saved efficacy_embedding_pca.png")
print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_[:2])
