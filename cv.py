import argparse
import csv
import os
from pathlib import Path
from typing import Literal, List

import numpy as np
import torch
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GroupKFold

import pimogp
import pandas as pd
from importlib.resources import files, as_file

from runModel import runmodel

def prepdata(dataset,targets,predtarget):
    y = torch.tensor(dataset[targets].values).float()
    conc = dataset[['drugA_conc','drugB_conc']]
    task_indices = torch.tensor(dataset["task_index"].values).long()
    dataset = dataset.drop(columns=["task_index"])
    drugcovars = dataset.iloc[:,-(50*2):]
    X = torch.tensor(pd.concat([conc,drugcovars],axis=1).values).float()
    # Noise and weights (only really do this when working with the latent GP prediction)
    if predtarget == "latent":
        noise = torch.tensor(dataset["GPVar"].values).float() # These are for implicitly weighting observations through likelihood
        weights = 1.0/noise # These are for sampling (during minibatching, and sampling inducing points)
    else:
        noise = torch.zeros(y.shape) # Equal weights
        weights = 1.0/(noise + 1)
    return y, X, task_indices, noise, weights


# Need a custom splitter for the LPO setting
def LDO_CV_split(train: pd.DataFrame, gkf: GroupKFold):
    drugs = np.unique(np.concatenate([train["drugA"].unique(),train["drugB"].unique()]))
    # Set up a list here
    idx_list = []
    for fold, (train_drug_idx, test_drug_idx) in enumerate(gkf.split(drugs, groups=drugs), 1):
        # Get names of drugs
        train_drugs = drugs[train_drug_idx]
        # Pull out training dataset and test dataset
        # Filter the data: rows where both DrugA and DrugB are in the train drugs
        idx_cv_train = train.index[(train['drugA'].isin(train_drugs)) & (train['drugB'].isin(train_drugs))]
        # Test data: rows where either DrugA or DrugB is in the test drugs (and not in training set)
        idx_cv_test = train.index[~train.index.isin(idx_cv_train)]

        # Convert the index labels to integer positions for compatibility with .iloc
        idx_cv_train_iloc = np.where(train.index.isin(idx_cv_train))[0]
        idx_cv_test_iloc = np.where(train.index.isin(idx_cv_test))[0]

        idx_list.append([idx_cv_train_iloc, idx_cv_test_iloc])
    return idx_list


def train_test_split_drugdata(input_type: Literal["raw","processed"], dataset: Literal["ONeil"],
                              setting: Literal["LTO", "LPO", "LDO", "LCO"],
                     seed: int=123):
    with as_file(files('pimogp.data.ONeil').joinpath('drug_latents.csv')) as f:
        drugs = pd.read_csv(f)
    if dataset == "ONeil":
        if input_type == "raw":
            with as_file(files('pimogp.data.ONeil').joinpath('raw.csv')) as f:
                data = pd.read_csv(f)
        elif input_type == "processed":
            with as_file(files('pimogp.data.ONeil').joinpath('processed.csv')) as f:
                data = pd.read_csv(f,sep=";")

    # Processing the data
    X = drugs.iloc[:, 9:]
    X = X.apply(zscore)
    drugs = drugs.iloc[:, 0:1]
    XA = X.add_prefix("A")
    XB = X.add_prefix("B")

    drugsA = pd.concat([drugs, XA], axis=1)
    drugsB = pd.concat([drugs, XB], axis=1)

    data = pd.merge(data, drugsA, left_on="drugA", right_on="Name")
    data = pd.merge(data, drugsB, left_on="drugB", right_on="Name")
    data = data.drop(columns=["Name_x", "Name_y"])
    data.dropna(inplace=True)

    # Adding an indicator for the cell_lines (tasks)
    data["task_index"] = pd.Categorical(data['cell_line']).codes



    # Split according to the various settings
    if setting == "LTO":
        data["id"] = data.cell_line.map(str) + ":" + data.drugA.map(str) + "_" + data.drugB.map(str)
        train_id, test_id = train_test_split(pd.DataFrame(data.id.unique()), test_size=0.2, random_state=seed)
        train = data[data["id"].isin(train_id[0].tolist())]
        test = data[data["id"].isin(test_id[0].tolist())]
        # Now drop the combination column
        ids = train["id"]
        train = train.drop(columns=["id"])
        test = test.drop(columns=["id"])
    if setting == "LPO":
        data["id"] = data.drugA.map(str) + "_" + data.drugB.map(str)
        train_id, test_id = train_test_split(pd.DataFrame(data.id.unique()), test_size=0.2, random_state=seed)
        train = data[data["id"].isin(train_id[0].tolist())]
        test = data[data["id"].isin(test_id[0].tolist())]
        # Now drop the combination column
        ids = train["id"]
        train = train.drop(columns=["id"])
        test = test.drop(columns=["id"])
    if setting == "LDO":
        # Training set should contain only combinations where both drugs are in the training ids
        drugs = np.unique(np.concatenate([data["drugA"].unique(), data["drugB"].unique()]))
        train_id, test_id = train_test_split(drugs, test_size=0.2, random_state=seed)
        # Pull out those where drugA is in the list of training drugs
        train = data[data["drugA"].isin(train_id)]
        # Pull out from these again, the ones where drugB is also on the list
        train = train[train["drugB"].isin(train_id)]
        # Test set is anything left over
        # Identify observations in data that are not in train
        test = data.merge(train, how='left', indicator=True)
        # Filter to keep only rows that are in data but not in train
        test = test[test['_merge'] == 'left_only']
        # Drop the merge indicator column
        test = test.drop(columns=['_merge'])
        # We won't use the ids in this setting but return one anyway
        ids = 1
    if setting == "LCO":
        data["id"] = data.cell_line.map(str)
        train_id, test_id = train_test_split(pd.DataFrame(data.id.unique()), test_size=0.2, random_state=seed)
        train = data[data["id"].isin(train_id[0].tolist())]
        test = data[data["id"].isin(test_id[0].tolist())]
        # Now drop the combination column
        ids = train["id"]
        train = train.drop(columns=["id"])
        test = test.drop(columns=["id"])
    return data, train, test, ids


def write_to_csv(filename, header, data):
    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(header)

        # Write the data rows
        writer.writerow(data)





def cross_validate(input_type: Literal["raw","processed"], predtarget: Literal["viability", "latent"],
                   dataset: Literal["ONeil"], setting: Literal["LTO", "LPO", "LDO", "LCO"],
                   model_type: Literal["nc", "mkl"], vardistr: Literal["mf","nat","chol"],
                   weighted: bool,
                   G: List[int], num_latents: List[int], num_inducing: List[int],
                   batch_size: int=256, num_epochs: int=12, seed: int=123):

    # Define what the targets are
    if input_type == "raw":
        targets = "viability"
    elif input_type == "processed":
        if predtarget == "viability":
            targets = "fMean"
        elif predtarget == "latent":
            targets = "GPMean"

    # Message
    print("Performing 5-fold cross validation!")
    print("Over: G=" + str(G) + " num_latent=" + str(num_latents) + " num_inducing=" + str(num_inducing))
    print("Model: " + model_type)
    print("Dataset: " + dataset)
    print("Using " + input_type + " data, with " + predtarget + " as target.")

    # Create some directories if they do not exist
    Path("results/plots/"+setting).mkdir(parents=True, exist_ok=True)
    Path("results/models/"+setting).mkdir(parents=True,exist_ok=True)
    # Fetch a train_test split of the data
    data, train, test, ids = train_test_split_drugdata(input_type=input_type,dataset=dataset,setting=setting,seed=seed)
    #exit()
    # Pull out the actual validation dataset
    # y_test, X_test, test_index, test_noise, test_weights = prepdata(test,targets,predtarget)
    # Commented out because it is never actually used here

    # Create the unique character string we'll use for later
    fname = "{0}{1}_data={2}_input={3}_target={4}_weighted={5}vardistr={6}_batchsize={7}_epochs={8}".format(setting,
                                                                                                            model_type,
                                                                                                            dataset,
                                                                                                            input_type,
                                                                                                            predtarget,
                                                                                                            str(weighted),
                                                                                                            vardistr,
                                                                                                            str(batch_size),
                                                                                                            str(num_epochs))

    # Now for the cross-validation itself:
    gkf = GroupKFold(n_splits=5)

    for g in G:
        for n_latent in num_latents:
            for n_inducing in num_inducing:
                fold = 1
                for train_idx, test_idx in (
                gkf.split(train, groups=ids) if setting != "LDO" else LDO_CV_split(train, gkf)):
                    # Message
                    print("G=" + str(g) + ", num_latent=" + str(n_latent) + ", num_inducing=" + str(n_inducing) + " fold=" + str(fold))
                    # Pull out the data
                    cv_y_train, cv_X_train, cv_train_indices, cv_train_noise, cv_train_weights = prepdata(
                        train.iloc[train_idx], targets, predtarget)
                    cv_y_test, cv_X_test, cv_test_indices, cv_test_noise, cv_test_weights = prepdata(
                        train.iloc[test_idx], targets, predtarget)
                    # Train model and predict
                    yhat = runmodel(x_train=cv_X_train, y_train=cv_y_train,
                                    y_noise=cv_train_noise, y_weights=cv_train_weights,
                                    train_indices=cv_train_indices, cell_covars=None,  # TODO fix cell covars
                                    x_test=cv_X_test, test_indices=cv_test_indices,
                                    pred_target=predtarget,
                                    G=g, num_latents=n_latent, num_inducing=n_inducing,
                                    batch_size=batch_size,
                                    num_tasks=data.task_index.max() + 1, model_type=model_type, num_epochs=num_epochs,
                                    vardistr=vardistr, weighted=weighted,
                                    fname="{0}_G={1}_num_latent={2}_num_inducing={3}_cvfold{4}".format(
                                        fname, str(g), str(n_latent), str(n_inducing), str(fold)),
                                    setting=setting)

                    # Move this to the same device as cv_y_test
                    yhat = yhat.to(cv_y_test.device)
                    # Calculate errors
                    rmse = (cv_y_test - yhat).square().mean().sqrt()
                    wrmse = (cv_y_test - yhat).square().mul(cv_test_weights).sum().div(cv_test_weights.sum()).sqrt()
                    # Now we write this to a csv file
                    write_to_csv("cv_results.csv",
                                 ["Setting", "Model", "Data", "input", "target", "weighted", "G", "num_latent", "num_inducing", "fold","RMSE", "wRMSE"],
                                 [setting, model_type, dataset, input_type, predtarget, weighted, str(g), str(n_latent), str(n_inducing),
                                  str(fold), str(rmse.item()), str(wrmse.item())])

                    fold = fold + 1








if __name__ == '__main__':
    # Create arg parser
    nparser = argparse.ArgumentParser(prog='cv',
                                      description="Perform 5-fold cross validation for model hypers")

    nparser.add_argument('--input_type', type=str, help='Which input type to use, processed or raw')
    nparser.add_argument('--predtarget', type=str, help='Which target to use, latent or viability')
    nparser.add_argument('--dataset', type=str, help='Which dataset to use, ONeil')
    nparser.add_argument('--setting', type=str, help='Which setting to predict in, LTO, LPO, LDO, LCO')
    nparser.add_argument('--model_type', type=str, help='Which model to use, nc or mkl')
    nparser.add_argument('--vardistr', type=str, help='Which type of variational distribution to use, mf, nat, chol')
    nparser.add_argument('--weighted', type=bool, help='Weighting observations by noise?')
    nparser.add_argument('--G', type=int, nargs='+', help='Which values of G to CV over?')
    nparser.add_argument('--num_latents', type=int, nargs='+', help='Which values of num_latent to CV over?')
    nparser.add_argument('--num_inducing', type=int, nargs='+', help='Which values of num_inducing to CV over?')
    nparser.add_argument('--batch_size', type=int, help='Batch size')
    nparser.add_argument('--num_epochs', type=int, help='No. of epochs')
    nparser.add_argument('--seed', type=int, help='Random seed')
    nparser.set_defaults(input_type="processed", predtarget="latent",
                   dataset="ONeil", setting= "LTO",
                   model_type="nc", vardistr="mf",
                   weighted=True,
                   G=[2], num_latents=[10], num_inducing=[50],
                   batch_size=256, num_epochs=1, seed=123)

    args = nparser.parse_args()


    cross_validate(input_type=args.input_type, predtarget=args.predtarget,
                   dataset=args.dataset, setting= args.setting,
                   model_type=args.model_type, vardistr=args.vardistr,
                   weighted=args.weighted,
                   G=args.G, num_latents=args.num_latents, num_inducing=args.num_inducing,
                   batch_size=args.batch_size, num_epochs=args.num_epochs, seed=args.seed)



