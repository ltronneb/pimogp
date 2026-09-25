"""
Subset drugs.csv (1375 CVCL columns) down to the 37 ONeil cell lines that have
a confirmed mapping in pimogp/data/oneil_cell_line_to_cvcl.csv, and rename the
CVCL columns back to the ONeil cell_line names for easy merging with the
GP surfaces / latents pipeline.
"""
import pandas as pd

DRUGS_CSV = "drugs.csv"
MAPPING_CSV = "pimogp/data/oneil_cell_line_to_cvcl.csv"
OUT_CSV = "pimogp/data/drugs_oneil_subset.csv"

META_COLS = ["Name", "PubchemID", "MolecularFormula", "MolecularWeight", "CanonicalSMILES"]

mapping = pd.read_csv(MAPPING_CSV)
mapping = mapping[mapping["in_drugs_csv"]]

drugs = pd.read_csv(DRUGS_CSV, sep=";")

cvcl_to_cellline = dict(zip(mapping["cvcl_id"], mapping["oneil_cell_line"]))
keep_cvcl = [c for c in cvcl_to_cellline if c in drugs.columns]
missing = set(cvcl_to_cellline) - set(keep_cvcl)
if missing:
    print(f"WARNING: {len(missing)} mapped CVCL ids not found as columns in drugs.csv: {missing}")

subset = drugs[META_COLS + keep_cvcl].rename(columns=cvcl_to_cellline)
subset.to_csv(OUT_CSV, index=False)

print(f"Wrote {OUT_CSV}: {subset.shape[0]} drugs x {len(keep_cvcl)} cell lines")
n_with_data = (subset[list(cvcl_to_cellline[c] for c in keep_cvcl)].notna().sum(axis=1) > 0).sum()
print(f"Drugs with at least one non-null viability value across these cell lines: {n_with_data}")
