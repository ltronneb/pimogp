from typing import Optional, Literal, List, Dict

import gpytorch.models
from botorch.models.utils.inducing_point_allocators import GreedyVarianceReduction
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel, ProductKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, LMCVariationalStrategy, \
    NaturalVariationalDistribution, CholeskyVariationalDistribution
import torch
from torch import Tensor

from pimogp.kernels.permutation_invariant_rbf import PermutationInvariantRBFKernel
from pimogp.variational.sum_variational_strategy import SumVariationalStrategy
from pimogp.variational.outputcovariance_lmc_variational_strategy import OutputCovarianceLMCVariationalStrategy
from pimogp.variational.permutation_invariant_variational_strategy import PermutationInvariantVariationalStrategy


class DrugComboICM_NC(ApproximateGP):
    r"""
    DrugComboModelICM_NC is the model for drug combination prediction that utilizes drug covariates, but
    no cell line covariates

    It wraps a PermutationInvariantVariationalStrategy in an LMCVariationalStrategy for a multi-output GP.
    The parameters of the LMC are learned "free-form" as opposed to making use of cell line information

    :param permutation: A Tensor giving the permutation the function should be invariant to
    :param conc_dims: The dimensions of the data input that corresponds to the drug concentrations
    :param drug_covar_dims: The dimension of the data input that corresponds to the drug covariates
    :param num_tasks: The number of tasks / outputs / cell lines
    :param num_latents: The number of latent functions in the LMC
    :param num_inducing: The number of inducing points per latent
    :param sample_inducing_from: A Tensor to sample inducing points from, usually this would be the training inputs
    :param inducing_weights: A vector of weights used in sampling of the inducing points
    :param vardistr: The variational distribution to use, mf=MeanField (Default), "nat"=Natural, "chol"=Cholesky
    """

    def __init__(self, permutation: Tensor,
                 conc_dims: Tensor,
                 drug_covar_dims: Tensor,
                 num_tasks: int,
                 num_latents: int,
                 num_inducing: int,
                 sample_inducing_from: Tensor,
                 inducing_weights: Tensor,
                 vardistr: Literal["mf", "nat", "chol"] = "mf"):

        # Define kernels here
        #covar_module  = []
        # Initialize a set to store seen pairs
        #seen_pairs = set()

        #for i, j in enumerate(permutation):
        #    # Sort the pair so that (i, j) and (j, i) look the same in the set
        #    pair = tuple(sorted((i, j.item())))

            # Process the pair only if it hasn't been seen before
        #    if pair not in seen_pairs:
        #        print(f"Unique pair (i, j): {pair}")
                # Initialise and add to list
        #        covar_module.append(gpytorch.kernels.RBFKernel(active_dims=pair))
                # Mark this pair as seen
        #        seen_pairs.add(pair)
        #covar_module = ProductKernel(*covar_module)
        covar_module = PermutationInvariantRBFKernel(permutation=permutation,
                                                     ard_num_dims=int((conc_dims.shape[-1] + drug_covar_dims.shape[-1])))

        #covar_module_concentrations = ScaleKernel(
        #    RBFKernel(active_dims=tuple(conc_dims.tolist()))
        #)
        #covar_module_drugs = RBFKernel(active_dims=tuple(drug_covar_dims.tolist()),
        #                               ard_num_dims=drug_covar_dims.shape[0])

        # Initialise the lengthscales
        lengthscales_init = sample_inducing_from.var(dim=0).sqrt().div(0.5)
        covar_module.lengthscale = lengthscales_init
        #covar_module_drugs.lengthscale = lengthscales_init[2:]


        # Inducing points where the action is!
        #p = inducing_weights.div(inducing_weights.sum())
        #idx = p.multinomial(num_samples=num_inducing * num_latents, replacement=False).reshape(num_latents,
        #                                                                                       num_inducing)
        #inducing_points = sample_inducing_from[idx]
        # Setting up inducing points using the GreedyVarianceReduction allocator from botorch
        ind_allocator = GreedyVarianceReduction()
        inducing_points = torch.empty((0, num_inducing, sample_inducing_from.size(1)))
        for i in range(num_latents):
            sample = ind_allocator.allocate_inducing_points(sample_inducing_from, covar_module,
                                                num_inducing=num_inducing, input_batch_shape=torch.Size([]))
            inducing_points = torch.cat((inducing_points, sample.unsqueeze(0)), dim=0)
            matches = (sample_inducing_from.unsqueeze(1) == sample).all(dim=2)
            row_matches = matches.any(dim=1)
            sample_inducing_from = sample_inducing_from[~row_matches]
        #inducing_points = ind_allocator.allocate_inducing_points(sample_inducing_from.unsqueeze(0).repeat(num_latents,1,1),
        #                                                   covar_module_concentrations*covar_module_drugs,num_inducing=400,
        #                                                         input_batch_shape=torch.Size([num_latents]))

        # The default is a MeanFieldDistribution
        variational_distribution = MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        # If not we can use a natural distribution
        if vardistr == "nat":
            variational_distribution = NaturalVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        # Or a cholesky distribution
        if vardistr == "chol":
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

        # Variational Strategy
        variational_strategy = LMCVariationalStrategy(
            PermutationInvariantVariationalStrategy(
                self, inducing_points, variational_distribution, permutation=permutation,
                learn_inducing_locations=True),
            num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1)

        super(DrugComboICM_NC, self).__init__(variational_strategy)

        # Store permutation here
        self.permutation = permutation
        self.conc_dims = conc_dims
        self.drug_covar_dims = drug_covar_dims
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_inducing = num_inducing
        self.sample_inducing_from = sample_inducing_from
        self.inducing_weights = inducing_weights
        self.vardistr = vardistr

        # Mean and covariance modules
        # Standard zero-mean
        self.mean_module = ZeroMean()
        # Covar over the concentrations, simple RBF
        #self.covar_module_concentrations = covar_module_concentrations
        # Covar over the drugs, RBF + ARD
        #self.covar_module_drugs = covar_module_drugs
        self.covar_module = covar_module



    def forward(self, x):
        mean_x = self.mean_module(x)
        # Final covariance is a simple product.
        #covar_x = self.covar_module_concentrations(x) * self.covar_module_drugs(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self):
        self.__init__(permutation=self.permutation,
                      conc_dims=self.conc_dims,
                      drug_covar_dims=self.drug_covar_dims,
                      num_tasks=self.num_tasks, num_latents=self.num_latents,
                      num_inducing=self.num_inducing, sample_inducing_from=self.sample_inducing_from,
                      inducing_weights=self.inducing_weights, vardistr=self.vardistr)


class DrugComboICM_MKL(ApproximateGP):
    r"""
    DrugComboModelNC is a model for drug combination prediction that utilizes drug covariates as well as
    cell line covariates via Multiple Kernel Learning

    It wraps a PermutationInvariantVariationalStrategy in an OutputCovarianceVariationalStrategy for a
    permutation invariant ICM which learns the parameters of the ICM from cell line information

    :param permutation: A Tensor giving the permutation the function should be invariant to
    :param conc_dims: The dimensions of the data input that corresponds to the drug concentrations
    :param drug_covar_dims: The dimension of the data input that corresponds to the drug covariates
    :param cell_covars: A list of cell line covariates for learning the LMC coefficients
    :param num_tasks: The number of tasks / outputs / cell lines
    :param num_latents: The number of latent functions in the LMC
    :param num_inducing: The number of inducing points per latent
    :param sample_inducing_from: A Tensor to sample inducing points from, usually this would be the training inputs
    :param inducing_weights: A vector of weights used in sampling of the inducing points
    :param vardistr: The variational distribution to use, mf=MeanField (Default), "nat"=Natural, "chol"=Cholesky
    """

    def __init__(self, permutation: Tensor,
                 conc_dims: Tensor,
                 drug_covar_dims: Tensor,
                 cell_covars: List[Tensor],
                 num_tasks: int,
                 num_latents: int,
                 num_inducing: int,
                 sample_inducing_from: Tensor,
                 inducing_weights: Tensor,
                 vardistr: Literal["mf", "nat", "chol"] = "mf"):
        # Inducing points where the action is!
        p = inducing_weights.div(inducing_weights.sum())
        idx = p.multinomial(num_samples=num_inducing * num_latents, replacement=False).reshape(num_latents,
                                                                                               num_inducing)
        inducing_points = sample_inducing_from[idx]

        # The default is a MeanFieldDistribution
        variational_distribution = MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        # If not we can use a natural distribution
        if vardistr == "nat":
            variational_distribution = NaturalVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        # Or a cholesky distribution
        if vardistr == "chol":
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

        # Here set up multiple-kernel learning
        startdim = 0
        tmp = cell_covars[0]
        ncols = int(tmp.shape[-1].long())
        active = tuple(torch.linspace(startdim, ncols - 1, ncols - 1).long())
        K = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
            active_dims=active,
            ard_num_dims=ncols))
        for i in range(1,len(cell_covars)):
            tmp = cell_covars[i]
            ncols = int(tmp.shape[-1].long())
            active = tuple(torch.linspace(startdim, ncols - 1, ncols - 1).long())
            K = K + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                active_dims=active,
                ard_num_dims=ncols))
            startdim = startdim + tmp.shape[-1]

        output_covars = torch.concat(cell_covars,dim=-1)
        output_kernel = K

        # Variational Strategy
        variational_strategy = OutputCovarianceLMCVariationalStrategy(
            PermutationInvariantVariationalStrategy(
                self, inducing_points, variational_distribution, permutation=permutation,
                learn_inducing_locations=True),
            output_kernel=output_kernel,
            output_covars=output_covars,
            num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1)

        super(DrugComboICM_MKL, self).__init__(variational_strategy)

        # Store permutation here
        self.permutation = permutation
        self.conc_dims = conc_dims
        self.drug_covar_dims = drug_covar_dims
        self.cell_covars = cell_covars
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_inducing = num_inducing
        self.sample_inducing_from = sample_inducing_from
        self.inducing_weights = inducing_weights

        # Mean and covariance modules
        # Standard zero-mean
        self.mean_module = ZeroMean()
        # Covar over the concentrations, simple RBF
        self.covar_module_concentrations = ScaleKernel(
            RBFKernel(active_dims=tuple(conc_dims.tolist()))
        )
        # Covar over the drugs, RBF + ARD
        self.covar_module_drugs = RBFKernel(active_dims=tuple(drug_covar_dims.tolist()),
                                            ard_num_dims=drug_covar_dims.shape[0])

        # Initialise the lengthscales
        self.covar_module_concentrations.base_kernel.lengthscale = torch.rand(1)
        self.covar_module_drugs.lengthscale = torch.rand(drug_covar_dims.shape[0])
        # TODO initialise cell covariance too!

    def forward(self, x):
        mean_x = self.mean_module(x)
        # Final covariance is a simple product.
        covar_x = self.covar_module_concentrations(x) * self.covar_module_drugs(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self):
        self.__init__(permutation=self.permutation,
                      conc_dims=self.conc_dims,
                      drug_covar_dims=self.drug_covar_dims,
                      cell_covars=self.cell_covars,
                      num_tasks=self.num_tasks, num_latents=self.num_latents,
                      num_inducing=self.num_inducing, sample_inducing_from=self.sample_inducing_from,
                      inducing_weights=self.inducing_weights)


class DrugComboLMC_NC(gpytorch.models.ApproximateGP):
    r"""
    Simple wrapper that constructs a proper LMC model from a list of ICM models.

    This specific model wraps the NC, or "No Cell" version of the model, where the LMC parameters are learnged
    free form
    """

    def __init__(self, params: List[Dict]):
        G = len(params)
        models = torch.nn.ModuleList([DrugComboICM_NC(**params[i]) for i in range(G)])

        variational_strategy = SumVariationalStrategy(models)

        super(DrugComboLMC_NC, self).__init__(variational_strategy)

        self.G = G
        self.models = models
        self.params = params

    def forward(self, x):
        raise NotImplementedError

    def reinit(self):
        self.__init__(self.params)


class DrugComboLMC_MKL(gpytorch.models.ApproximateGP):
    r"""
    Simple wrapper that constructs a proper LMC model from a list of ICM models.

    This specific model wraps the Full, version of the model, where the LMC parameters are learned
    from a covariance function over cell line covariates.
    """
    def __init__(self, params: List[Dict]):
        G = len(params)
        models = torch.nn.ModuleList([DrugComboICM_MKL(**params[i]) for i in range(G)])

        variational_strategy = SumVariationalStrategy(models)



        super(DrugComboLMC_MKL, self).__init__(variational_strategy)

        self.G = G
        self.models = models
        self.params = params

    def forward(self, x):
        raise NotImplementedError

    def reinit(self):
        self.__init__(self.params)
