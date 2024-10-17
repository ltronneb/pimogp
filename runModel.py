from os import abort
from typing import Literal, Optional, List
from matplotlib import pyplot as plt
import gpytorch
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader
from tqdm import tqdm

from pimogp.models.models import DrugComboLMC_NC, DrugComboLMC_MKL
from pimogp.utils.utils import better_varelbo_init


def runmodel(x_train: Tensor, y_train: Tensor,
             y_noise: Tensor, y_weights: Tensor,
             train_indices: Tensor, cell_covars: Optional[List[Tensor]],
             x_test: Tensor, test_indices: Tensor,
             pred_target: Literal["viability","latent"],
             G: int, num_latents: int, num_inducing: int, batch_size: int,
             num_tasks: int, model_type: Literal["nc", "mkl"], num_epochs: int,
             vardistr: Literal["mf","nat","chol"],
             weighted: bool, fname: chr):
    print("INSIDE RUNMODEL!!!")
    #exit()
    """

    @param x_train: X locations of the training dataset
    @param y_train: y targets of the training dataset
    @param y_noise: noise associated with each observation
    @param y_weights: weights associated to sample minibatches
    @param train_indices: output index for the training dataset
    @param cell_covars: covariates for the cell lines, should be a list of tensors
    @param x_test: X locations of the test set, where to predict
    @param test_indices: output index for the test dataset
    @param pred_target: what the prediction target is, viability or the latent GP
    @param G: Number of components in the LMC
    @param num_latents: Number of latent functions per G
    @param num_inducing: Number of inducing points per latent function
    @param batch_size: Size of the minibatches
    @param num_tasks: Number of tasks to predict (no of cell lines)
    @param model_type: Which model we are using, MKL or NC?
    @param num_epochs: Number of epochs for training
    @param vardistr: Type of variational distribution
    @param weighted: Are we weighting observations by their noise?
    @param fname: Unique string to save models and plots
    @return:
    """
    # Set the device if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # And move everything to this device if needed
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    y_noise = y_noise.to(device)
    y_weights = y_weights.to(device)
    train_indices = train_indices.to(device)
    test_indices = test_indices.to(device)
    if cell_covars is not None:
        cell_covars = [cell_covar.to(device) for cell_covar in cell_covars]
    x_test = x_test.to(device)

    # First thing we do is set up minibatching
    train_dataset = TensorDataset(x_train,y_train,y_noise,train_indices)
    if pred_target == "latent":
        sampler = WeightedRandomSampler(y_weights,len(y_weights),replacement=True)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    # Now setting up the model
    # First set up the permutation and the dimensions needed
    permutation = torch.cat([torch.tensor([1]), torch.tensor([0]),
                             torch.linspace((2 + 50), 1 + (2 * 50), 50), torch.linspace(2, 1 + 50, 50)]).long()
    conc_dims = torch.tensor([0, 1])
    drug_dims = torch.linspace(2, 2 * 50 + 1, 2 * 50).long()

    # Parameters for the model
    param_dict = {"permutation": permutation,
                  "conc_dims": conc_dims,
                  "drug_covar_dims": drug_dims,
                  "num_tasks": num_tasks,
                  "num_latents": num_latents,
                  "num_inducing": num_inducing,
                  "sample_inducing_from": x_train,
                  "inducing_weights": y_weights,
                  "vardistr": vardistr}

    # Tack on cell line covariates if needed
    if model_type == "mkl":
        param_dict[cell_covars] = cell_covars

    # Set up G models with these parameters
    params = [param_dict.copy() for i in range(G)]

    # Now initialise models
    model = None
    likelihood = None
    if model_type == "nc":
        model = DrugComboLMC_NC(params)
    elif model_type == "mkl":
        model = DrugComboLMC_MKL(params)
    # Set up likelihood, depending on if we weight by noise
    if weighted:
        likelihood = FixedNoiseGaussianLikelihood(noise=y_noise,
                                                  learn_additional_noise=True)  # This warning is fine!
    else:
        likelihood = GaussianLikelihood()
    # Quick check here that everything is initialised
    if model is None:
        raise ValueError("Model is not initialised")
    if likelihood is None:
        raise ValueError("Likelihood is not initialised")

    # Here we try to reinitialise the model in an attempt to get find a better position for inducing points
    idx = y_weights.multinomial(num_samples=1000, replacement=False)
    X_mll = x_train[idx]
    y_mll = y_train[idx]
    noise_mll = y_noise[idx]
    task_indices_mll = train_indices[idx]
    model, likelihood = better_varelbo_init(X_mll, y_mll, noise_mll, task_indices_mll, model, likelihood)

    # Now here also move model and likelihood to GPU if this hasn't been done automatically
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Now model training
    model.train()
    likelihood.train()
    hyperparameter_optimizer = None
    variational_ngd_optimizer = None
    scheduler_hypers = None
    scheduler_variational = None
    if vardistr == "nat":
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr = 0.01)
        variational_ngd_optimizer = gpytorch.optim.NGD(
            model.variational_parameters(),
            num_data=y_train.size(0), lr=0.1)
        scheduler_variational = MultiStepLR(variational_ngd_optimizer,
                                            milestones=[int(0.5*num_epochs),int(0.75*num_epochs)],
                                            gamma=0.1)
    else:
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

    scheduler_hypers = MultiStepLR(hyperparameter_optimizer,
                                   milestones=[int(0.5 * num_epochs), int(0.75 * num_epochs)],
                                   gamma=0.1)

    # Defining the loss:
    mll = VariationalELBO(likelihood,model,num_data=y_train.size(0))

    # Tracking loss
    train_loss = []



    # Training!
    with gpytorch.settings.cholesky_max_tries(12):
        epochs_iter = tqdm(range(num_epochs),desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch, y_noise_batch, task_batch in minibatch_iter:
                # Zero gradients
                hyperparameter_optimizer.zero_grad()
                if variational_ngd_optimizer is not None:
                    variational_ngd_optimizer.zero_grad()

                # Get outputs
                output = model(x_batch,task_indices=task_batch)

                # Compute loss
                if isinstance(likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
                    loss = -mll(output, y_batch, noise=y_noise_batch)
                else:
                    loss = -mll(output, y_batch)
                # Call backward on loss
                loss.backward()
                # Take a step
                hyperparameter_optimizer.step()
                if variational_ngd_optimizer is not None:
                    variational_ngd_optimizer.step()

                # Update counter
                minibatch_iter.set_postfix(loss=loss.item())

                # Track loss
                train_loss.append(loss.item())

            # Step learning rate scheduler
            scheduler_hypers.step()
            if scheduler_variational is not None:
                scheduler_variational.step()

    # Saving a plot of the training loss
    plot_loss(train_loss,fname)

    # Now predict also in minibatch
    test_dataset = TensorDataset( x_test, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # Set into eval mode
    model.eval()
    likelihood.eval()
    yhat = []
    with torch.no_grad(), gpytorch.settings.cholesky_max_tries(12):
        minibatch_iter = tqdm(test_loader,desc="Minibatch", leave=False)
        for x_batch, task_batch in minibatch_iter:
            predictions = likelihood(model(x_batch, task_indices=task_batch))
            mean = predictions.mean
            yhat.append(mean)

    yhat_vector = torch.cat(yhat, 0)

    # Will also save the model here
    torch.save(model.state_dict(),"model_"+fname)
    torch.save(likelihood.state_dict(),"likelihood"+fname)

    # Now return the prediction
    return yhat_vector.clone().detach()



def plot_loss(train_loss: List,filename: chr):
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
    fname = "loss_" + filename + ".png"
    plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
    plt.close()
