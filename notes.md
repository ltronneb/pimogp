# PIMoGP Project Notes

## What this project is

**PIMoGP** — Permutation Invariant Multi-Output Gaussian Processes for drug synergy prediction in cancer.

Core idea: predict cell viability surfaces for drug combinations across cancer cell lines. Drug A + Drug B = Drug B + Drug A (commutative), so a custom permutation-invariant kernel exploits this symmetry. Drug molecules are encoded via a pre-trained Transformer VAE (SMILES → latent vectors), and a variational multi-output GP predicts response surfaces.

## Key files

- `pimogp/models/single_task_gp.py` — fits an independent exact GP per drug pair + cell line. ARD RBF kernel over 2 concentration inputs. Generated the 293 plots in `pimogp/plots/` and 583 surface files in `pimogp/surfaces/`, all for KPL1 cell line, using processed fMean data.
- `pimogp/models/surface_embedding_and_prediction.py` — two things: (1) assembles drug pair latent representations from VAE outputs; (2) stacks single-task GP surfaces into a (900, N) matrix and runs truncated SVD (K=5) to learn a low-rank basis. Essentially PCA on the surfaces.
- `pimogp/models/models.py` — multi-output GP model classes (ICM, LMC variants) with permutation-invariant kernel. The more principled joint model.
- `pimogp/kernels/permutation_invariant_rbf.py` — core custom kernel.
- `pimogp/variational/` — custom variational strategies for the multi-output GP.
- `cv.py` — 5-fold cross-validation across LTO/LPO/LDO/LCO settings.
- `notebooks/Optimising_Optimisation.ipynb` — does NOT use the surfaces folder (contrary to what it sounds like). Separate analysis.
- `notebooks/look_at_latents.ipynb` — latent space analysis (PCA, UMAP, t-SNE) of VAE drug embeddings.

## Data

- `pimogp/data/ONeil/raw.csv` — raw drug concentrations + viability
- `pimogp/data/ONeil/processed.csv` — processed fMean values (used for surface fitting)
- `pimogp/latents/transformer_vae_cancer_drugs/` — pre-trained VAE outputs (z_out.pt, mu_out.pt, logvar_out.pt)
- `pimogp/surfaces/` — 583 posterior mean surfaces + 583 stddev surfaces (30x30 grid, flat .txt files)
- `pimogp/plots/` — 293 PNG plots of GP mean + uncertainty, all KPL1

## Pipeline

1. Encode drugs via pre-trained Transformer VAE → latent vectors
2. Fit single-task GPs per drug pair + cell line → posterior mean surfaces saved to `surfaces/`
3. Stack surfaces → truncated SVD → K=5 coefficients per pair (exploratory, not yet wired up downstream)
4. Multi-output GP (models.py) is the more principled joint approach — takes drug latents + concentrations as input

## Current status / what was discussed

- Single-task GP pipeline is complete and has run for KPL1 (293 pairs)
- Surface embedding script (`surface_embedding_and_prediction.py`) is exploratory — the SVD compression machinery works but C (the coefficients) is not saved or used downstream. Pipeline is half-finished.
- The surface PCA is closely related to functional PCA. Since surfaces come from GP posteriors they are already smooth, so plain SVD is a reasonable approximation to fPCA.
- There is some tension between the two-stage approach (single-task → compress → predict coefficients) and the direct joint multi-output GP approach in models.py.
- Suggested things to show collaborator: singular value decay plot, basis function heatmaps (top 5 Phi columns as 30x30), reconstruction quality scatter.

## Pipeline completion: surface_embedding_and_prediction.py

The script is now a complete two-stage pipeline: SVD basis → MLP coefficient predictor.

### What was added / fixed

- **Missing imports**: `load_processed_data` and `get_unique_drug_pairs` were called but never imported — now imported from `pimogp/utils/processing.py` at the top of the file.
- **C overwrite bug**: line 203 (old) overwrote the `(K, N)` coefficient matrix `C` with a single projected vector `(K,)` — removed.
- **`reconstruct_torch` shape bug**: was being called with the full `C` matrix instead of a `c_vec` — fixed.
- **Train/test split**: 80/20 random split (seed 42) on drug pairs *before* SVD, so the basis is learned on train surfaces only. Test surfaces are projected onto the train basis via `Phi.T @ (Y_test - mean_vec)`.
- **SVD basis saved**: `pimogp/data/svd_basis.pt` — contains `mean_vec`, `Phi`, `C` (train coefficients), `S` (singular values), and `pairs` DataFrame so columns of C are traceable to drug pair names.
- **MLP predictor**: `pimogp/models/mlp_predictor.py` — `MLPPredictor` class, configurable hidden layers and dropout. Kept separate from the main script.
- **PCA on latents**: 512-dim pair latents compressed to 32 PCs (fit on train only) before MLP input to combat overfitting. Components saved to `pimogp/data/pca_mean.npy` and `pimogp/data/pca_components.npy`.
- **MLP saved**: `pimogp/data/mlp_predictor.pt`
- **Reconstruction block**: after training, MLP predicts `C_pred` from latents and reconstructs surfaces via `reconstruct_all_torch` — train vs test scatter plot saved.

### MLP architecture / training (current)

- Input: 32-dim PCA projection of 512-dim pair latents
- Hidden: 128 → 64 → 32 (deeper + narrower than original 256 → 64)
- Dropout: 0.3 after each hidden ReLU
- Optimizer: Adam, lr=1e-3, weight_decay=1e-3
- Scheduler: `ReduceLROnPlateau`, factor=0.7, patience=300, monitoring test MSE
- Epochs: 5000

### Overfitting history

- No PCA, no regularisation: train MSE → ~0, test MSE ~0.73 — severe overfitting
- Weight decay alone (`1e-3`): marginal improvement
- PCA (32 dims) + weight decay + dropout (0.3) + deeper/narrower arch + LR scheduler: visually promising — surface shapes captured on held-out pairs, main failure is undershooting low-viability at high doses

### Files saved by the script

| File | Contents |
|---|---|
| `pimogp/data/cancer_drugs_latents.csv` | Per-drug VAE latents with drug name index |
| `pimogp/data/cancer_drugs_latents_pairs.csv` | Per-pair latents (A_ and B_ prefixed columns) |
| `pimogp/data/cancer_drugs_latents_pairs_surface_data.csv` | Drug pair names + flattened GP surface |
| `pimogp/data/svd_basis.pt` | SVD basis: mean_vec, Phi, C, S, pairs |
| `pimogp/data/pca_mean.npy` | PCA mean vector (512,) |
| `pimogp/data/pca_components.npy` | PCA projection matrix (512, 32) |
| `pimogp/data/mlp_predictor.pt` | Trained MLP state dict |

## Collaborator notebook: notebooks/surface_explorer.ipynb

Created `notebooks/surface_explorer.ipynb` — a self-contained notebook for collaborators to explore the two-stage pipeline interactively.

### Structure (23 cells)

1. **Load surfaces** — reads `cancer_drugs_latents_pairs_surface_data.csv` directly (parses stored array strings via `np.array([float(v) for v in x.strip('[]').split()])`), then merges latent columns from `cancer_drugs_latents_pairs.csv` on drugA/drugB. Stddev surfaces are loaded on-demand only for the 6 visualised pairs.
2. **Functional PCA (SVD)** — train/test split (seed 42, same as script), SVD on train surfaces, singular value decay plot, basis function heatmaps, reconstruction scatter.
3. **MLP training** — runs the full training loop in-notebook. Hyperparameters (`HIDDEN`, `DROPOUT`, `LR`, `WEIGHT_DECAY`, `EPOCHS`) exposed at the top of a cell. Saves model + PCA artifacts to `pimogp/data/`. Includes learning curve plot.
4. **Evaluation** — true vs MLP-predicted scatter (train + test), side-by-side surface comparison for any held-out pair (change `pair_id`).
5. **Interactive prediction** — `predict_pair(drugA, drugB)` function: if a GP surface exists, shows 3-panel true/predicted/residual; otherwise just plots the prediction.

### Key design decision
Notebook re-derives the SVD basis and PCA from scratch each run (using the same seed 42 split), so `svd_basis.pt` is not needed. Only the two tracked CSVs are required as inputs.

## .gitignore fixes (2026-04-21)

- **Critical**: changed `/pimogp/data/` → `/pimogp/data/*` — excluding a directory entirely prevents `!` negation rules inside it from working; excluding contents (`/*`) allows negation.
- Fixed wrong paths `/pimogp/pimogp/plots/` and `/pimogp/pimogp/surfaces/` → `/pimogp/plots/` and `/pimogp/surfaces/`
- Added `/pimogp/latents/` and `/pimogp/trained_transformers/` (were untracked but unexcluded)
- Moved `/pimogp/plots/` and `/pimogp/results_plots/` to `/plots/` and `/results_plots/` (correct repo-root-relative paths)
- Only two data files are tracked: `cancer_drugs_latents_pairs.csv` and `cancer_drugs_latents_pairs_surface_data.csv`

## Bug fixes made

- `surface_embedding_and_prediction.py` line 11: changed `import matplotlib.pylab as plt` → `import matplotlib.pyplot as plt` (pylab deprecated in matplotlib 3.9+, caused AttributeError on rcParams)
- `single_task_gp.py` line 90: `device` is referenced inside `plot_gp_surface` but not passed as a parameter — relies on it being in global scope. Works in __main__ context but would fail if called standalone.

## Environment

- Conda env: `pimogp`
- Python: 3.10, at `/local_home/vrameshl/miniconda3/envs/pimogp/bin/python`
- matplotlib 3.10.6, PyTorch, GPyTorch, BoTorch
- `selfies` must be installed via pip (not on conda)
