from copy import deepcopy
import gc

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO
import torch


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
