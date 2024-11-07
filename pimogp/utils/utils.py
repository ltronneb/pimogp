import csv
import os
from copy import deepcopy
import gc
from importlib.resources import as_file, files
from typing import List, Literal

import numpy as np
import pandas as pd
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO
import torch
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import GroupKFold, train_test_split


def better_varelbo_init(x, y, noise, task_indices, model, likelihood, num_inits=100):
    print("Trying to find a better initialization in " + str(num_inits) + " attempts!")
    with torch.no_grad():
        device = next(model.parameters()).device # Need to make sure everything stays on the same device
        #print("before reinit")
        #print(next(model.parameters()).device)
        model.reinit()
        model.to(device)
        newmodel = deepcopy(model)
        init_loss = []
        model.train()
        likelihood.train()
        #print("after reinit")
        #print(next(model.parameters()).device)
        mll = VariationalELBO(likelihood, model, num_data=y.size(0))
        output = model(x, task_indices=task_indices)
        if isinstance(likelihood, FixedNoiseGaussianLikelihood):
            loss = -mll(output, y,noise=noise)
        else:
            loss = -mll(output, y)
        for i in range(num_inits):
            newmodel.reinit()
            newmodel.to(device)
            newmodel.train()
            newmll = VariationalELBO(likelihood, newmodel, num_data=y.size(0))
            newoutput = newmodel(x, task_indices=task_indices)
            if isinstance(likelihood, FixedNoiseGaussianLikelihood):
                newloss = -newmll(newoutput, y,noise=noise)
            else:
                newloss = -newmll(newoutput, y)
            # If this loss is better overwrite
            if newloss.item() < loss.item():
                #print("Old loss: ", loss.item())
                #print("New loss: ", newloss.item())
                model = deepcopy(newmodel)
                loss = deepcopy(newloss)

            # Delete everything
            gc.collect()
    print("Initialization done!")
    return model, likelihood


def plot_loss(train_loss: List,filename: chr,setting: chr):
    # Calculate a shift constant
    min_loss = min(train_loss)
    shift = abs(min_loss) + 1e-6  # Adding a small epsilon to avoid log(0)

    # Shift the loss values to be positive
    shifted_loss = [loss + shift for loss in train_loss]

    # Plotting the training loss on a logarithmic scale
    plt.plot(shifted_loss, "r-", label='Training Loss')  # 'r-' means red line
    plt.yscale('log')  # Set the y-axis to log scale
    plt.xlabel('Iterations')  # Label for the x-axis
    plt.ylabel('Loss (Log Scale)')  # Label for the y-axis
    plt.title('Training Loss over Iterations')  # Title of the plot
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better readability
    # plt.show()  # Display the plot
    fname = "results/plots/" + setting + "/loss_" + filename + ".png"
    plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def prepdata(dataset,targets,predtarget):
    y = torch.tensor(dataset[targets].values).float()
    conc = dataset[['drugA_conc','drugB_conc']]
    task_indices = torch.tensor(dataset["task_index"].values).long()
    dataset = dataset.drop(columns=["task_index"])
    drugcovars = dataset.iloc[:,-(50*2):]
    X = torch.tensor(pd.concat([conc,drugcovars],axis=1).values).float()
    # Noise and weights
    if predtarget == "latent":
        noise = torch.tensor(dataset["GPVar"].values).float() # These are for implicitly weighting observations through likelihood
        weights = 1.0/noise # These are for sampling (during minibatching, and sampling inducing points)
    else:
        noise = torch.zeros(y.shape) # These are for implicitly weighting observations through likelihood
        weights = 1.0/(noise + 1.0)# These are for sampling (during minibatching, and sampling inducing points)
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

    if dataset == "ONeil":
        with as_file(files('pimogp.data.ONeil').joinpath('drug_latents.csv')) as f:
            drugs = pd.read_csv(f)
        if input_type == "raw":
            with as_file(files('pimogp.data.ONeil').joinpath('raw.csv')) as f:
                data = pd.read_csv(f)
        elif input_type == "processed":
            with as_file(files('pimogp.data.ONeil').joinpath('processed.csv')) as f:
                data = pd.read_csv(f,sep=";")
    if dataset == "GDSC7x7":
        with as_file(files('pimogp.data.GDSC_7x7').joinpath('drug_latents.csv')) as f:
            drugs = pd.read_csv(f)
        if input_type == "raw":
            with as_file(files('pimogp.data.GDSC_7x7').joinpath('raw.csv')) as f:
                data = pd.read_csv(f)

    # Processing the data
    X = drugs.iloc[:, 9:]
    #X = X.apply(zscore)
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

