import argparse
from pathlib import Path
from typing import Literal, List

import gpytorch.settings
import torch
from sklearn.model_selection import GroupKFold

from pimogp.utils.utils import train_test_split_drugdata, LDO_CV_split, prepdata, write_to_csv
from pimogp.utils.runModel import runmodel


def cross_validate(input_type: Literal["raw","processed"], predtarget: Literal["viability", "latent"],
                   dataset: Literal["ONeil","GDSC7x7"], setting: Literal["LTO", "LPO", "LDO", "LCO"],
                   model_type: Literal["nc", "mkl"], vardistr: Literal["mf","nat","chol"],
                   weighted: bool,
                   G: List[int], num_latents: List[int], num_inducing: List[int],
                   batch_size: int=256, num_epochs: int=12, seed: int=123,
                   num_inits: int=10,initial_lr=0.01):

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
                    yhat, _, _ = runmodel(x_train=cv_X_train, y_train=cv_y_train,
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
                                    setting=setting,num_inits=num_inits,
                                    initial_lr=initial_lr)

                    # Move this to the same device as cv_y_test
                    yhat = yhat.to(cv_y_test.device)
                    # Calculate errors
                    rmse = (cv_y_test - yhat).square().mean().sqrt()
                    wrmse = (cv_y_test - yhat).square().mul(1.0/cv_test_noise).sum().div((1.0/cv_test_noise).sum()).sqrt()
                    # Now we write this to a csv file
                    write_to_csv("cv_results.csv",
                                 ["Setting", "Model", "Data", "input", "target", "weighted", "G", "num_latent", "num_inducing", "num_epochs", "batch_size",  "fold","RMSE", "wRMSE"],
                                 [setting, model_type, dataset, input_type, predtarget, weighted, str(g), str(n_latent), str(n_inducing), str(num_epochs), str(batch_size),
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
    nparser.add_argument('--weighted', type=bool, action=argparse.BooleanOptionalAction, help='Weighting observations by noise?')
    nparser.add_argument('--G', type=int, nargs='+', help='Which values of G to CV over?')
    nparser.add_argument('--num_latents', type=int, nargs='+', help='Which values of num_latent to CV over?')
    nparser.add_argument('--num_inducing', type=int, nargs='+', help='Which values of num_inducing to CV over?')
    nparser.add_argument('--batch_size', type=int, help='Batch size')
    nparser.add_argument('--num_epochs', type=int, help='No. of epochs')
    nparser.add_argument('--seed', type=int, help='Random seed')
    nparser.add_argument('--num_inits', type=int, help='How many random initialisations?')
    nparser.add_argument("--initial_lr", type=float, help='Initial learning rate for optimizer')
    nparser.set_defaults(input_type="processed", predtarget="latent",
                   dataset="ONeil", setting= "LTO",
                   model_type="nc", vardistr="mf",
                   weighted=True,
                   G=[2], num_latents=[10], num_inducing=[50],
                   batch_size=256, num_epochs=4, seed=123, num_inits=10,
                   initial_lr=0.01)

    args = nparser.parse_args()


    with (gpytorch.settings.max_root_decomposition_size(2000),
          gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False),
          gpytorch.settings.linalg_dtypes(torch.float)):
        cross_validate(input_type=args.input_type, predtarget=args.predtarget,
                   dataset=args.dataset, setting= args.setting,
                   model_type=args.model_type, vardistr=args.vardistr,
                   weighted=args.weighted,
                   G=args.G, num_latents=args.num_latents, num_inducing=args.num_inducing,
                   batch_size=args.batch_size, num_epochs=args.num_epochs, seed=args.seed, num_inits=args.num_inits,
                       initial_lr=args.initial_lr)



