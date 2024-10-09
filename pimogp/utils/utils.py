from copy import deepcopy
import gc

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO
import torch


def better_varelbo_init(x, y, noise, task_indices, model, likelihood, num_inits=100):
    with torch.no_grad():
        model.reinit()
        newmodel = deepcopy(model)
        init_loss = []
        model.train()
        likelihood.train()
        mll = VariationalELBO(likelihood, model, num_data=y.size(0))
        output = model(x, task_indices=task_indices)
        if isinstance(likelihood, FixedNoiseGaussianLikelihood):
            loss = -mll(output, y,noise=noise)
        else:
            loss = -mll(output, y)
        for i in range(num_inits):
            newmodel.reinit()
            newmodel.train()
            newmll = VariationalELBO(likelihood, newmodel, num_data=y.size(0))
            newoutput = newmodel(x, task_indices=task_indices)
            if isinstance(likelihood, FixedNoiseGaussianLikelihood):
                newloss = -newmll(newoutput, y,noise=noise)
            else:
                newloss = -newmll(newoutput, y)
            # If this loss is better overwrite
            if newloss.item() < loss.item():
                print("Old loss: ", loss.item())
                print("New loss: ", newloss.item())
                model = deepcopy(newmodel)
                loss = deepcopy(newloss)

            # Delete everything
            gc.collect()
    return model, likelihood
