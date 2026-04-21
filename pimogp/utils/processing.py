import pandas as pd
import numpy as np
import torch 

# Load and process raw and processed data

def load_processed_data():
    df = pd.read_csv("pimogp/data/ONeil/processed.csv", sep=";")
    return df

def load_raw_data():
    df = pd.read_csv("pimogp/data/ONeil/raw.csv")
    return df

def filter_by_cell_line(df, cell_line):
    return df[df['cell_line'] == cell_line]

def filter_by_drug_pair(df, drug1, drug2):
    cond1 = (df['drugA'] == drug1) & (df['drugB'] == drug2)
    cond2 = (df['drugA'] == drug2) & (df['drugB'] == drug1)
    return df[cond1 | cond2]

def filter_by_drug_pair_and_cell_line(df, drug1, drug2, cell_line):
    cond1 = (df['drugA'] == drug1) & (df['drugB'] == drug2) & (df['cell_line'] == cell_line)
    cond2 = (df['drugA'] == drug2) & (df['drugB'] == drug1) & (df['cell_line'] == cell_line)
    return df[cond1 | cond2]

def get_unique_drug_pairs(df):
    """
    Returns a sorted list of unique drug pairs (as tuples) from the DataFrame.
    Each pair is sorted alphabetically to avoid duplicates like (A, B) and (B, A).
    """
    pairs = df[['drugA', 'drugB']].apply(lambda row: tuple(sorted([row['drugA'], row['drugB']])), axis=1)
    unique_pairs = set(pairs)
    return sorted(unique_pairs)

def get_unique_drugs(df):
    """
    Returns a sorted list of unique drugs from the DataFrame.
    """
    drugs = pd.unique(df[['drugA', 'drugB']].values.ravel('K'))
    return sorted(drugs)

def get_unique_cell_lines(df):
    """
    Returns a sorted list of unique cell lines from the DataFrame.
    """
    return sorted(df['cell_line'].unique())