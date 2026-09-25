# Efficacy Embedding → Surface Prediction: Experiment Log

**Goal:** build a drug embedding shaped by KPL1 monotherapy efficacy, and test whether it's actually informative for predicting drug-pair combination surfaces — motivated by a collaborator's finding that swapping the Transformer-VAE embedding for a one-hot encoding didn't change surface predictions at all.

**Bottom line:** it wasn't the embedding — it was the two-stage architecture. Once the pipeline is combined into one jointly-trained model, the efficacy embedding measurably beats one-hot identity on the decisive test (unseen drugs), which it could not do in the original two-stage pipeline.

---

## 1. Bug fix: drug names were misaligned to VAE latents

`pimogp/models/surface_embedding_and_prediction.py:101` used:

```python
names = cancer_drugs_df.iloc[cancer_drugs_smiles.index]['Name']   # WRONG
```

`cancer_drugs_smiles.index` is pandas' default `0..7449` row position, not the `index` column `cancer_drugs.csv` actually stores (the true row number in `final_drugs.csv`, since 188 long-SMILES rows were dropped scattered throughout during corpus construction). The two diverge for **7446 of 7449 rows**. Example: row 3 has `index=4` (maps to **Amifostine**) but the buggy code fetched `final_drugs.csv.iloc[3]` (**Cosmegen** — a different drug entirely).

**Fix:** `cancer_drugs_df.iloc[cancer_drugs_smiles['index']]['Name']`. Regenerated `cancer_drugs_latents.csv`, `cancer_drugs_latents_pairs.csv`, `cancer_drugs_latents_pairs_surface_data.csv`, the SVD basis, and the MLP predictor with the fix applied.

## 2. Cell-line and drug-panel mapping

- Built `pimogp/data/oneil_cell_line_to_cvcl.csv`: maps ONeil's 39 cell-line names to Cellosaurus CVCL IDs in the new `drugs.csv` (1375-cell-line pan-cancer panel), via the Cellosaurus API. 37/39 matched; `COLO320DM` and `UWB1289BRCA1` have no data in `drugs.csv`.
- Caught 3 name collisions where the naive match would have picked the *wrong* biological entity: **PA1** (correct: `CVCL_0479`, not `CVCL_T887` which is an unrelated skin-fibroblast line), **ES2** (correct: `CVCL_3509`, not `CVCL_AX39` which carries an EWSR1-FLI1 fusion — a different, Ewing-sarcoma-associated line), **HT29** (correct: `CVCL_0320`, ATCC HTB-38/DepMap ACH-000552, not the near-empty duplicate stub `CVCL_A8EZ`).
- Subsetted to `pimogp/data/drugs_oneil_subset.csv`: 8481 drugs × 37 ONeil cell lines. 319 drugs have a KPL1 monotherapy value.

## 3. Efficacy embedding model

**Design** (`pimogp/models/efficacy_embedding.py`): `SelfiesEfficacyAutoencoder` — SELFIES token sequence → Transformer encoder → 32-dim bottleneck `z` → Transformer decoder (reconstruction) + linear head off `z` (KPL1 efficacy regression). Deterministic (no reparameterization/KLD — not a generative VAE), trained on the 7449-molecule cancer-drug corpus (`cancer_drugs.csv`), with the KPL1 regression loss driven by all 309 labeled drugs every epoch (no held-out split for that loss — the point is embedding geometry, not held-out efficacy prediction; reconstruction keeps its own train/test split to monitor generalization).

**Training script:** `pimogp/utils/train_efficacy_embedding.py`. To run:
```
cd pimogp && PYTHONPATH=. /local_home/vrameshl/miniconda3/envs/pimogp_clean/bin/python pimogp/utils/train_efficacy_embedding.py
```

**Result (60 epochs):** 70.0% / 67.9% train/test SELFIES token reconstruction accuracy (still improving when stopped). Efficacy loss on the full labeled set: 0.008 (normalized units).

**Embedding organization check** (`pimogp/utils/plot_efficacy_embedding_pca.py`): PCA of the 32-dim embedding across all 7449 drugs, colored by KPL1 efficacy for the 309 labeled ones.

| PC | % variance | Pearson r vs. KPL1 efficacy |
|---|---|---|
| PC1 | 42.2% | **−0.994** |
| PC2 | 19.3% | −0.27 |

The embedding is strongly organized by potency along its dominant axis.

## 4. Two-stage pipeline: is the embedding actually informative?

Old pipeline: independent per-pair GP (`single_task_gp.py`) → SVD/functional-PCA basis (K=5) on the resulting surfaces → separate `DeepSetPredictor(zA,zB)` regression against the fixed SVD coefficients.

**Cross-check methodology** — reproduce the collaborator's "one-hot vs. embedding" comparison directly, plus a mean-only baseline (predict the training-mean surface, no model at all):

### Pair-level split (`pimogp/utils/compare_embedding_informativeness.py`)
504/583 pairs covered (after fixing a case-sensitivity bug in the drug-name join — ONeil names are title case, `final_drugs.csv` names are ALL CAPS).

| condition | train MSE | test MSE |
|---|---|---|
| mean baseline (no model) | 4.78 | 4.69 |
| one-hot identity | 0.87 | 1.22 |
| efficacy embedding | 0.88 | **1.34** |

Embedding does *not* beat one-hot. But this split can't distinguish memorization from generalization — with only 35 drugs recurring across train and test pairs, one-hot's perfect identity signal is sufficient.

### Leave-drugs-out split (`pimogp/utils/compare_embedding_leave_drugs_out.py`) — the decisive test
35 drugs → 27 seen / 8 held out entirely from training. Test = any pair touching a held-out drug (219 pairs; 27 have *both* drugs unseen).

| condition | train MSE | test MSE (≥1 drug unseen) |
|---|---|---|
| mean baseline | — | 4.34 |
| one-hot identity | 0.99 | **2.81** |
| efficacy embedding | 1.02 | 4.08 (≈ baseline) |

On genuinely novel drugs, the embedding is statistically indistinguishable from predicting the mean, and loses decisively to one-hot. **This reproduces and sharpens the collaborator's original finding.**

## 5. Root-cause analysis: why is there so little to predict?

- The mean surface alone explains **95.7%** of raw KPL1 surface variance — almost every pair looks like a generic dose-response shape.
- Of the residual (pair-specific) variance, the 5 SVD components split **53.1% / 26.2% / 9.6% / 5.5% / 1.6%** — components 1+2 alone carry 79%. The "5-dimensional" prediction task is effectively 2-dimensional, with thin, easily-noise-dominated tail components.
- Checked whether `single_task_gp.py` shares/ties hyperparameters across pairs (which would artificially oversmooth toward a common shape) — it doesn't; each pair gets an independently-initialized `ConstantMean()` + ARD `RBFKernel`, trained from scratch. The low pair-to-pair variance appears to be genuine (shared qualitative dose-response form across arbitrary drug pairs), not a fitting artifact.
- Checked whether `fMean` (the GP fitting target) is already a Bliss/Loewe synergy deviation rather than raw viability — it is not. `fMean` ranges 0–1 (mean 0.66), consistent with raw viability fraction; no baseline-subtraction step exists anywhere in the pipeline.

**Diagnosis (confirmed by the user):** not enough target variance for any input — however informative — to visibly move the needle in the two-stage architecture, because most of a pair's variance is thrown away at the fixed, embedding-blind SVD compression step before the embedding-based model ever sees the data.

## 6. Redesign: joint single-stage model

Replaces independent-per-pair-GP → SVD → separate-regression with one model trained end-to-end on raw dose-viability data:

```
Yhat(cell; zA, zB) = mean_surface[cell] + Φ[cell] · g(zA, zB)
```

- **`mean_surface`**: one free number per concentration grid cell, shared by every pair. KPL1's grid is discrete and *exactly* shared across all 583 pairs (10×10=100 cells; verified 583 pairs × 100 cells = 58,300 raw rows, one value per cell, no exceptions) — so it's initialized to the closed-form per-cell average of `fMean` across all pairs (exact, one pandas `groupby`, no fitting), but left trainable, since that closed-form average is only *guaranteed* jointly optimal when the other factor (`g`) is unconstrained — here `g` is restricted to whatever the embedding network can express, so the true joint optimum can drift from the raw average.
- **`Φ`**: K=5 basis values per grid cell, one shared table for every pair. Same discrete-grid argument as `mean_surface`, but `Φ`'s fit is *coupled to `g`* through the loss (a bilinear product, like matrix factorization) — no closed form, randomly initialized, learned by gradient descent. Loses SVD's orthogonality/ranking guarantees (any invertible rotation of `Φ` and `g` gives an identical product), but is still extractable/plottable as basis-shape heatmaps exactly like the old SVD `Φ`.
- **`g`**: `DeepSetPredictor(zA, zB)` → K coefficients, evaluated **once per pair** (not once per grid cell — concentration varies within a pair, the embedding doesn't) and broadcast across that pair's 100 cells via the dot product. The only place drug identity enters the model.

Trained jointly against every raw `(pair, cell, fMean)` row at once — no independent per-pair GP fitting, no SVD, no train/test handoff between stages.

**Code:** `pimogp/models/deep_set_predictor.py` (reused unchanged) + `JointSurfaceModel` class and training loop in `pimogp/utils/train_joint_surface_model.py`.

### Pair-level split result
504 pairs, 404 train / 100 test (test pairs held out entirely, drugs still seen elsewhere in training).

| | train MSE | test MSE |
|---|---|---|
| mean-only baseline | — | 0.366 |
| joint model (embedding) | 0.0063 | **0.0068** |

**~54x lower error than the mean baseline**, train ≈ test (minimal overfitting). Dramatically different from anything the two-stage pipeline produced. Visual check: `pimogp/utils/plot_joint_surface_reconstructions.py` → `joint_model_test_reconstructions.png` (6 held-out pairs, true/predicted/residual).

### Leave-drugs-out split result — the decisive test, revisited
Same 35-drug pool, same 27-seen/8-held-out split as the two-stage experiment (`pimogp/utils/train_joint_surface_model_leave_drugs_out.py`).

| condition | train MSE | test MSE (≥1 drug unseen) | vs. baseline |
|---|---|---|---|
| mean-only baseline | — | 0.0744 | — |
| **one-hot** (`..._onehot.py`) | 0.0301 | 0.0267 | −61.5% |
| **efficacy embedding** | 0.0073 | **0.0230** | −69.1% |

The embedding now beats one-hot (0.0230 vs 0.0267, ~14% lower test error) — **reversing the two-stage result** on the exact same decisive test.

### Full comparison, both architectures

| Test | Architecture | Input | test MSE | vs. baseline |
|---|---|---|---|---|
| Pair-level | two-stage (SVD) | one-hot | 1.22 | −74% |
| Pair-level | two-stage (SVD) | embedding | 1.34 | −71% |
| Leave-drugs-out | two-stage (SVD) | one-hot | **2.81** | −35% |
| Leave-drugs-out | two-stage (SVD) | embedding | 4.08 | −6% (≈ baseline) |
| Leave-drugs-out | joint model | one-hot | 0.0267 | −61.5% |
| Leave-drugs-out | joint model | embedding | **0.0230** | −69.1% |

*(MSE values aren't on a comparable absolute scale between architectures — two-stage predicts 5 SVD coefficients directly, joint model predicts raw per-cell viability — compare the "vs. baseline" percentages across architectures, not raw MSE.)*

## 7. Why does one-hot do as well as it does, even on unseen drugs?

Broke the leave-drugs-out test set down by how many drugs in the pair are unseen (19,200 rows with exactly one unseen drug, 2,700 rows with both unseen), evaluating both saved joint-model checkpoints on each subset:

| | one drug unseen (n=19,200) | both drugs unseen (n=2,700) |
|---|---|---|
| embedding | 0.0217 | 0.0321 |
| one-hot | 0.0270 | 0.0246 |

**Hypothesis going in:** one-hot should specifically collapse on the both-unseen subset (its one-hot coordinate for a never-seen drug is fed into a weight column that gradient descent never touches, since that input is always 0 during training). **Not confirmed** — one-hot doesn't degrade on both-unseen; if anything it's slightly better there than on one-unseen (though n=2,700/27 pairs is a small enough sample that this specific breakdown shouldn't be over-trusted without repeating across several random drug-holdout splits).

**Revised explanation:** `mean_surface` and `Φ` are pure functions of concentration — zero dependence on drug identity — and concentration is always in-distribution (same 100 cells for every pair, seen or unseen drugs alike). Given the mean surface alone explains 95.7% of raw variance, `g(zA,zB)` is only ever a *small correction* on top of an already-dominant, always-reliable, drug-agnostic baseline. A "garbage" (untrained, random-init) `g` output for an unseen one-hot drug therefore adds noise to a small correction term, not to the bulk of the prediction — which is consistent with one-hot not collapsing, and with the embedding's real-but-modest (14%) edge over it.

## 8. Files created/modified this session

| File | Purpose |
|---|---|
| `pimogp/models/surface_embedding_and_prediction.py` | Bug fix (line 101) |
| `pimogp/data/oneil_cell_line_to_cvcl.csv` | ONeil cell line → CVCL ID mapping |
| `pimogp/utils/subset_oneil_drugs.py`, `pimogp/data/drugs_oneil_subset.csv` | Cell-line-subsetted drug monotherapy panel |
| `pimogp/models/efficacy_embedding.py` | `SelfiesEfficacyAutoencoder` |
| `pimogp/utils/train_efficacy_embedding.py` | Efficacy embedding training script |
| `pimogp/utils/plot_efficacy_embedding_pca.py` | Embedding organization check |
| `pimogp/utils/predict_svd_coeffs_from_efficacy_embedding.py` | Two-stage stage-2, DeepSet version |
| `pimogp/utils/compare_embedding_informativeness.py` | Pair-level one-hot/embedding/baseline comparison |
| `pimogp/utils/compare_embedding_leave_drugs_out.py` | Leave-drugs-out comparison (two-stage) |
| `pimogp/utils/train_joint_surface_model.py` | `JointSurfaceModel`, pair-level split |
| `pimogp/utils/train_joint_surface_model_leave_drugs_out.py` | Joint model, leave-drugs-out, embedding |
| `pimogp/utils/train_joint_surface_model_leave_drugs_out_onehot.py` | Joint model, leave-drugs-out, one-hot control |
| `pimogp/utils/plot_joint_surface_reconstructions.py` | True/predicted/residual surface plots |

## 9. Open question for next steps

The embedding wins the decisive test, but only by a modest margin (14%), because `g(zA,zB)` only ever contributes a *small correction* on top of a dominant, drug-agnostic `mean_surface + Φ` baseline — the architecture structurally limits how much the embedding's quality can matter. Next: brainstorm ways to give the embedding a deeper / more load-bearing role in the prediction, rather than leaving it confined to a thin residual term.

## 10. Attempt 1: multiplicative potency scaling — did not help

Tried the first brainstormed idea for giving the embedding a more load-bearing role:

```
Yhat(cell; zA,zB) = mean_surface[cell] * scale(zA,zB) + Phi[cell] . g(zA,zB)
```

`scale(zA,zB)` (a `DeepSetPredictor` with `out_dim=1`, parameterised as `1 + tanh(...)`) lets the embedding multiply the entire dominant `mean_surface` term, not just add a thin correction. Code: `pimogp/utils/train_joint_surface_model_scaled.py` (pair-level), `..._scaled_leave_drugs_out.py` (embedding), `..._scaled_leave_drugs_out_onehot.py` (one-hot control).

| Test | Architecture | Input | train MSE | test MSE | vs. own baseline |
|---|---|---|---|---|---|
| Pair-level | additive | embedding | 0.0063 | 0.0068 | −98.1% |
| Pair-level | scaled | embedding | 0.0058 | 0.0068 | −97.9% (no change) |
| Leave-drugs-out | additive | one-hot | 0.0301 | 0.0267 | −61.5% |
| Leave-drugs-out | additive | embedding | 0.0073 | 0.0230 | −69.1% |
| Leave-drugs-out | scaled | one-hot | 0.0068 | **0.0193** | −86.9% |
| Leave-drugs-out | scaled | embedding | 0.0082 | 0.0228 | −67.9% |

**Result: the reverse of what we wanted.** One-hot improved substantially (0.0267 → 0.0193) while the embedding barely moved (0.0230 → 0.0228) — one-hot now *beats* the embedding again, undoing the additive model's win.

**Why:** `scale(zA,zB)` is a generic capacity increase, not an embedding-specific lever — it benefits whichever per-pair signal is cleanest. For ~88% of leave-drugs-out test rows, one drug is still seen, and one-hot's identity signal for that seen partner is sharp/unambiguous (perfect memorization); giving the model more room to exploit that let one-hot extract more value from it. The embedding's softer, less-exploitable-in-this-way signal didn't benefit as much.

**Takeaway for next steps:** generic "more capacity" changes reward whichever input is cleanest for memorization, which is still one-hot on this test. What's needed instead is an architectural constraint that *specifically* requires generalizable, structure-aware behavior that one-hot cannot provide at all (no "similar drugs behave similarly" notion exists in a one-hot space) — motivating idea 2, the monotherapy-anchored Bliss-independence combination.

## 11. Root cause found: the efficacy label itself was wrong

Investigating whether the embedding encodes dose-response *shape* (not just depth/potency) surfaced something more fundamental: the "KPL1 efficacy" scalar in `drugs.csv` (used for all embedding training up to this point) **does not correlate with ONeil's own KPL1 monotherapy screen data** (r = -0.13; some drugs actively rank oppositely — e.g. Sorafenib is ONeil's most potent killer at max dose but is labeled one of the *weakest* drugs in `drugs.csv`). `drugs.csv` is almost certainly aggregated from a different, unrelated assay/database (likely GDSC/CTRP/PRISM-style, not ONeil's own protocol).

Consequence: every embedding trained against that column (sections 3, 4, 6-10) was shaped by a signal essentially unrelated to the actual ONeil potency/surface-prediction task — no architecture change downstream could have fixed that. This likely explains a good chunk of why one-hot kept winning regardless of what we tried to the model.

**The real ONeil monotherapy data was available all along**, just unused: every combo screen's concentration grid includes the "edge" where one drug's concentration is 0 (see section 6's grid structure) -- so a full 10-point dose-response curve per drug falls out of `processed.csv` directly. Checked robustness: ~22-37 replicate measurements per (drug, dose) pooled across different combo partners, ~3% cross-partner noise (mean std 0.029 on a scale spanning ~0.93) -- reliable, not noise-dominated.

## 12. Retrained embedding on real ONeil monotherapy curves

New model: `SelfiesCurveAutoencoder` (`pimogp/models/efficacy_embedding.py`) -- identical to `SelfiesEfficacyAutoencoder` except the auxiliary head predicts the full 10-point curve (sigmoid-bounded) instead of one scalar. Training script: `pimogp/utils/train_efficacy_embedding_from_curves.py`. Supervision: real ONeil KPL1 curves for all 38 drugs that appear in the KPL1 screen (down from 309 drugs.csv-labeled, but now the *right* 38 -- and richer, 10 points each instead of 1).

**Result (60 epochs):** 72.2%/70.0% train/test SELFIES reconstruction accuracy (slightly better than the drugs.csv-trained version, 70.0%/67.9%). Curve MSE converged to 0.0007.

**Shape/depth diagnostic, repeated with the new embedding** (same method as section 9's initial check, curve PCA + leave-one-out cross-validated Ridge regression from the full 32-dim embedding):

| | old embedding (drugs.csv scalar) | new embedding (ONeil curves) |
|---|---|---|
| curve PC1 (depth, 79.4% var) vs embedding, LOOCV r | 0.001 | **0.852** |
| curve PC2 (shape, 16.5% var) vs embedding, LOOCV r | -0.677 (worse than useless) | **0.576** |
| curve PC3 (shape, 2.4% var) vs embedding, LOOCV r | -0.088 | 0.159 |

The new embedding honestly captures both real ONeil depth (r=0.85) and a meaningful share of genuine shape variation beyond depth (r=0.58) -- the old embedding had ~zero grip on either, now explained by the label mismatch in section 11 rather than an architecture limitation.

**Not yet done:** rerunning the joint-model informativeness comparisons (pair-level, leave-drugs-out, one-hot control) with this new embedding, to see whether a genuinely-informative embedding changes the one-hot-keeps-winning pattern from sections 10 and the Bliss-model attempt. Natural next session's starting point.

## 13. Rerunning the informativeness battery with the curve embedding — decisive win

Reran the leave-drugs-out comparison (identical seed, identical 35-drug pool -- verified the new embedding's Name coverage is exactly identical to the old one, so the split reproduces exactly) with the curve-trained embedding in place of the old drugs.csv-scalar one, for both architectures. One-hot results reused unchanged (same split, no need to rerun).

Code: `pimogp/utils/train_joint_surface_model_leave_drugs_out_curve_embedding.py`, `pimogp/utils/train_joint_surface_model_bliss_leave_drugs_out_curve_embedding.py`.

| Architecture | Input | Leave-drugs-out test MSE | vs. one-hot |
|---|---|---|---|
| Additive | one-hot | 0.0267 | — |
| Additive | old embedding (drugs.csv label) | 0.0230 | −14% |
| Additive | **new embedding (real ONeil curves)** | **0.0174** | **−35%** |
| Bliss | one-hot | 0.0130 | — |
| Bliss | old embedding (drugs.csv label) | 0.0224 | +72% (worse than one-hot) |
| Bliss | **new embedding (real ONeil curves)** | **0.00591** | **−55%** |

**The Bliss architecture reverses completely**: it lost badly to one-hot with the mislabeled embedding (section 10), and now wins decisively with the correctly-labeled one -- consistent with the diagnosis that Bliss specifically needs real monotherapy-shape information in `z`, which only the curve-trained embedding actually provides (section 12's shape correlation check). It also produces the best absolute result of the entire session: 0.00591 test MSE on pairs where at least one drug was never seen during training, ~4x better than the additive model's best result and ~74% better than the same architecture with the old embedding.

**Overall conclusion for the session**: the original "embedding doesn't matter" finding (collaborator's observation, reproduced in section 4) was ultimately a *data* problem, not an architecture or embedding-capacity problem -- `drugs.csv`'s KPL1 label didn't correlate with the real task at all (section 11). Once the embedding is supervised with the label that's actually relevant (real ONeil monotherapy curves) and paired with an architecture designed to use curve-shape information (Bliss independence), the embedding clearly, substantially outperforms identity memorization on the decisive unseen-drug test.

## 14. Pair-level split, all benchmarks, with the curve embedding

Completing the pair-level comparison (only the old-embedding numbers existed before) for both architectures, one-hot included this time.

Code: `pimogp/utils/train_joint_surface_model_curve_embedding.py`, `..._onehot.py`, `..._bliss_curve_embedding.py`, `..._bliss_onehot.py`.

| Architecture | Split | Input | test MSE |
|---|---|---|---|
| Additive | pair-level | one-hot | 0.0268 ⚠️ see caveat below |
| Additive | pair-level | old embedding | 0.0068 |
| Additive | pair-level | **new curve embedding** | **0.0054** |
| Bliss | pair-level | one-hot | 0.00147 |
| Bliss | pair-level | old embedding | 0.0016 |
| Bliss | pair-level | new curve embedding | 0.00147 (tied with one-hot) |

**Bliss pair-level**: one-hot and the new embedding land at an effective tie (0.00147 both) -- consistent with pair-level being the "easy" split where one-hot's perfect memorization is fully available regardless of input quality. The embedding's real edge over one-hot is specific to leave-drugs-out (section 13: 0.00591 vs 0.0130), not pair-level.

**Additive + one-hot pair-level -- reproducibility caveat**: this run is suspicious and was rerun to check. Both runs converged to the *exact same* test MSE (0.02678, matching to the last digit) with train MSE flat from epoch 5 through epoch 300 (0.02866-0.02868, no meaningful movement) while the mean-only baseline drifted from ~0.027 up to ~0.10 underneath it. Reproducing identically across two independent random initializations rules out ordinary bad-init noise -- this looks like a genuine, deterministic optimization pathology specific to the additive-model + one-hot-input combination (weight decay interacting badly with a sparse 35-dim one-hot input, or the LR schedule cutting before the model escapes an early plateau, are the two most likely culprits). **Not resolved this session** -- treat the 0.0268 number as unreliable/needing a different training setup (e.g. no weight decay on the one-hot pathway, longer patience, or a warm restart) before trusting a comparison against it. Does not affect the Bliss-model or leave-drugs-out conclusions, which used independently-behaving training runs.

## 15. Collaborator-facing materials

Produced for a presentation to the collaborator: `MODEL.tex` (full mathematical writeup: notation, embedding model + loss, both surface architectures, training/splits, all benchmark tables) plus four figures:

- `curve_embedding_pca_depth_shape.png` -- 2D PCA of the curve embedding shaded by real ONeil depth/shape
- `bliss_curve_leave_drugs_out_reconstructions.png` -- true/predicted/residual surfaces, 6 pairs with both drugs entirely unseen
- `bliss_curve_phi_basis_functions.png` -- the 5 learnt Φ synergy-residual basis functions
- `bliss_curve_true_vs_predicted_train_test.png` -- true-vs-predicted scatter across every cell, train (r=0.989, n=28500) vs. test (r=0.946, n=21900), best model (Bliss + curve embedding, leave-drugs-out)

No LaTeX compiler available in this environment (`pdflatex`/`latexmk` not installed) -- `MODEL.tex` and the 4 PNGs need to sit in the same directory and be compiled elsewhere (Overleaf or a local TeX install).

## 16. Repo cleanup: consolidated the 15 joint-model scripts down to 2

The additive/scaled/Bliss x pair-level/leave-drugs-out x old-embedding/one-hot/curve-embedding grid had accumulated 15 near-duplicate training scripts (12 after deleting the abandoned scaled-model variant, see the "delete this" note below). Consolidated to:

- `pimogp/models/joint_surface_models.py` -- `JointSurfaceModel` (additive), `JointSurfaceModelBliss` + `MonotherapyCurve` (Bliss), no longer defined inline inside training scripts or imported sideways between them.
- `pimogp/utils/joint_surface_data.py` -- shared grid loading, embedding/one-hot feature construction, and pair-level/leave-drugs-out splitting.
- `pimogp/utils/train_additive.py`, `pimogp/utils/train_bliss.py` -- one script per architecture, parameterized by `--split {pair_level,leave_drugs_out}` and `--input {old_embedding,curve_embedding,onehot}`.

Both new scripts were verified against the original 15-script results before deleting them: additive leave-drugs-out/curve-embedding reproduced 0.0180 (vs. 0.0174 original), Bliss reproduced 0.0060 (vs. 0.0059) -- differences are normal run-to-run variance, not a regression. The 4 plotting scripts (`plot_joint_surface_reconstructions.py`, `plot_bliss_curve_*.py`) had their imports updated to the new module locations and were confirmed to still import cleanly; they still point at the specific already-trained checkpoints used for `MODEL.pdf`'s figures, which were left in place.

**Deleted separately** (per explicit request, superseded by the Bliss result): the "scaled" model (`train_joint_surface_model_scaled*.py`, 3 scripts + 3 checkpoints) -- the multiplicative-potency-scaling attempt from section 10 that made one-hot improve more than the embedding. Its numeric result stays documented in section 10; only the dead code was removed.

