'''
This script is used to convert the kekulé SMILES in the final_drugs.csv (cancer drugs) to aromatic SMILES and merge with the pubchem 500k training data

'''

import pandas as pd
import numpy as np
import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as rdms

# Reading in the data to make sure it's all good

# Final drugs
cancer_df = pd.read_csv("pimogp/data/final_drugs.csv")

# Training data
pubchem_df = pd.read_csv("pimogp/data/pubchem_500k.csv", delimiter=",")

mols_aromatic = [Chem.MolFromSmiles(x) for x in cancer_df['smiles']]
arom_smiles = [Chem.MolToSmiles(m, kekuleSmiles=False) for m in mols_aromatic]
cancer_df['aromatic_smiles'] = arom_smiles

# Filter out smiles that are too long
smiles_lengths = [len(x) for x in arom_smiles]
cancer_df['smiles_lengths'] = smiles_lengths
cancer_df_short = cancer_df[cancer_df['smiles_lengths'] < 120]
cancer_df_long = cancer_df[cancer_df['smiles_lengths'] >= 120]

# Extract parent compound from smiles
parent_mols = [rdms.FragmentParent(Chem.MolFromSmiles(x)) for x in cancer_df_short['aromatic_smiles']]
parent_smiles = [Chem.MolToSmiles(m) for m in parent_mols]
cancer_df_short['smiles'] = parent_smiles

# Create a new dataframe with the cancer and pubchem data using the smiles columns
merged_df = pd.DataFrame()
merged_df['smiles'] = pd.concat([pubchem_df['smiles'], cancer_df_short['smiles']], keys=["pubchem", "cancer"])

# Save the merged data
merged_df.to_csv("pimogp/data/merged_drugs.csv", index=True)


