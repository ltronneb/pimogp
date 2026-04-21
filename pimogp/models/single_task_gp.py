import gpytorch
import torch
import math
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import torch
from pimogp.utils.processing import load_raw_data, load_processed_data, filter_by_cell_line, filter_by_drug_pair_and_cell_line, get_unique_drug_pairs, get_unique_drugs, get_unique_cell_lines

class SingleTaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingleTaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Use ARD for 2 input dimensions (2 lengthscales)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model(model, train_x, train_y, likelihood, num_epochs=100):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        # Logging
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        noise = likelihood.noise.detach().cpu().item()
        if (i+1) % 250 == 0:
            print(f"Epoch {i+1}/{num_epochs} | -MLL: {loss.item():.4f} | Lengthscale: {lengthscale} | Noise variance: {noise:.4f}")
        losses.append(loss.item())
    return model, likelihood, losses

def plot_drug_response(df, drugA, drugB, cell_line):
    """
    Plots a 2D scatter plot of drug concentrations vs viability for a given drug pair and cell line.

    Args:
        df (pd.DataFrame): The raw data DataFrame.
        drugA (str): Name of the first drug.
        drugB (str): Name of the second drug.
        cell_line (str): Name of the cell line.
    """
    subset = filter_by_drug_pair_and_cell_line(df, drugA, drugB, cell_line)
    x = subset["drugA_conc"]
    y = subset["drugB_conc"]
    viability = subset["viability"]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=viability, cmap='viridis', s=60, edgecolor='k')
    plt.colorbar(scatter, label='Viability')
    plt.xlabel(f'{drugA} concentration')
    plt.ylabel(f'{drugB} concentration')
    plt.title(f'Viability for {drugA} + {drugB} on {cell_line}')
    plt.show()

def plot_gp_surface(model, likelihood, train_x, train_y, drugA_name, drugB_name, cell_line=None, resolution=30, save=True):
    """
    Plots the GP mean prediction and uncertainty as 2D contour plots with training data points superimposed.
    """
    model.eval()
    likelihood.eval()

    # Create a grid over the range of concentrations
    x_min, x_max = train_x[:, 0].min().item(), train_x[:, 0].max().item()
    y_min, y_max = train_x[:, 1].min().item(), train_x[:, 1].max().item()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    grid_tensor = torch.tensor(grid, dtype=train_x.dtype).to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(grid_tensor))
        mean = preds.mean.cpu().numpy().reshape(xx.shape)
        stddev = preds.stddev.cpu().numpy().reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot mean prediction as contour
    contour1 = ax1.contourf(xx, yy, mean, levels=20, cmap='viridis')
    ax1.set_title('GP Mean Prediction')
    ax1.set_xlabel(f'{drugA_name} concentration')
    ax1.set_ylabel(f'{drugB_name} concentration')
    cbar1 = fig.colorbar(contour1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Predicted Viability')

    # Plot uncertainty as contour
    contour2 = ax2.contourf(xx, yy, stddev, levels=20, cmap='plasma')
    ax2.set_title('GP Uncertainty (Stddev)')
    ax2.set_xlabel(f'{drugA_name} concentration')
    ax2.set_ylabel(f'{drugB_name} concentration')
    cbar2 = fig.colorbar(contour2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('Uncertainty')

    # Superimpose training data points on both plots
    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()
    
    # Color points by their viability values
    scatter1 = ax1.scatter(train_x_np[:, 0], train_x_np[:, 1], marker='x', 
                          s=50, edgecolors='black', linewidth=0.5, color='k', label='Training data')
    scatter2 = ax2.scatter(train_x_np[:, 0], train_x_np[:, 1], marker='x', 
                          s=50, edgecolors='black', linewidth=0.5, color='k', label='Training data')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    # Add suptitle with drug pair and cell line
    if cell_line is not None:
        suptitle = f"{drugA_name} + {drugB_name}\nCell line: {cell_line}"
    else:
        suptitle = f"{drugA_name} + {drugB_name}"
    fig.suptitle(suptitle, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # directories to save the surface data and plots
    surface_dir = f"pimogp/surfaces/"   # directory to save the surface data
    plots_dir = f"pimogp/plots/"        # directory to save the plots
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    np.savetxt(f"{surface_dir}/{drugA_name}_{drugB_name}_{cell_line}.txt", mean)
    np.savetxt(f"{surface_dir}/{drugA_name}_{drugB_name}_{cell_line}_stddev.txt", stddev)
    # save the plot if save is True
    if save:
        plt.savefig(f"{plots_dir}/{drugA_name}_{drugB_name}_{cell_line}.png")
        plt.close(fig)
    else:
        return fig

def fit_and_plot_all_pairs_per_cell_line(df, cell_line, plots_dir='plots', num_epochs=100):
    import os
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subset = filter_by_cell_line(df, cell_line)
    pairs = subset[['drugA', 'drugB']].drop_duplicates().values.tolist()

    for drugA, drugB in pairs:
        print(f"Processing: {drugA}, {drugB}, {cell_line}")
        subset = filter_by_drug_pair_and_cell_line(df, drugA, drugB, cell_line)
        if len(subset) < 3:
            print(f"  Skipping: not enough data points ({len(subset)})")
            continue

        train_x = torch.tensor(subset[["drugA_conc", "drugB_conc"]].values, dtype=torch.float32, device=device)
        train_y = torch.tensor(subset["viability"].values, dtype=torch.float32, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = SingleTaskGP(train_x, train_y, likelihood).to(device)

        model, likelihood, losses = train_model(model, train_x, train_y, likelihood, num_epochs=num_epochs)

        fig = plot_gp_surface(model, likelihood, train_x, train_y, drugA, drugB, cell_line)
        plot_filename = f'{drugA}_{drugB}_{cell_line}.png'.replace('/', '-')
        fig.savefig(os.path.join(plots_dir, plot_filename))
        plt.close(fig)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data

    df = load_raw_data()
    subset = filter_by_drug_pair_and_cell_line(df, "5-FU", "ABT-888", "A2058")

    train_x = torch.tensor(subset[["drugA_conc", "drugB_conc"]].values, dtype=torch.float32, device=device)
    train_y = torch.tensor(subset["viability"].values, dtype=torch.float32, device=device)

    # Testing a  single triplet

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleTaskGP(train_x, train_y, likelihood).to(device)

    model, likelihood, losses = train_model(model, train_x, train_y, likelihood, num_epochs=100)

    unique_drugs = get_unique_drugs(df)
    unique_cell_lines = get_unique_cell_lines(df)
    drugA = unique_drugs[0]
    drugB = unique_drugs[1]
    cell_line = unique_cell_lines[0]
    
    plot_drug_response(df, drugA, drugB, cell_line)

    plot_gp_surface(model, likelihood, train_x, train_y, drugA, drugB, cell_line)

    # Testing all triplets per cell line

    cell_line = unique_cell_lines[0]
    fit_and_plot_all_pairs_per_cell_line(df, cell_line)

    ## Leiv's data

    df_subset = pd.read_csv('notebooks/A2780_5-FU_Bortezomib.csv')
    subset = filter_by_drug_pair_and_cell_line(df_subset, "5-FU", "Bortezomib", "A2780")

    train_x = torch.tensor(subset[["drugA_conc", "drugB_conc"]].values, dtype=torch.float32, device=device)
    train_y = torch.tensor(subset["fMean"].values, dtype=torch.float32, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleTaskGP(train_x, train_y, likelihood).to(device)

    model, likelihood, losses = train_model(model, train_x, train_y, likelihood, num_epochs=100)

    fig = plot_gp_surface(model, likelihood, train_x, train_y, "5-FU", "Bortezomib", "A2780", save=False)
    fig.savefig("leiv_fmean_regress.png")

    ## Loading processed data and assembling a data frame of all fmeans for all drug pairs and cell lines

    df = load_processed_data()
    unique_pairs = get_unique_drug_pairs(df)    
    unique_cell_lines = get_unique_cell_lines(df)

    cell_line = ['KPL1']
    for pair in unique_pairs:   # looping over all drug pairs
            print(f"Processing: {pair[0]} + {pair[1]} on {cell_line[0]}")
            subset = filter_by_drug_pair_and_cell_line(df, pair[0], pair[1], cell_line[0])
            train_x = torch.tensor(subset[["drugA_conc", "drugB_conc"]].values, dtype=torch.float32, device=device)
            train_y = torch.tensor(subset["fMean"].values, dtype=torch.float32, device=device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = SingleTaskGP(train_x, train_y, likelihood).to(device)
            model, likelihood, losses = train_model(model, train_x, train_y, likelihood, num_epochs=1000)
            plot_gp_surface(model, likelihood, train_x, train_y, pair[0], pair[1], cell_line[0], save=True)
