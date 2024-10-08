from typing import Optional, Union

import linear_operator
import torch
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.utils.interpolation import left_interp
from torch import LongTensor, Tensor

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.kernels import Kernel


def _select_lmc_coefficients(lmc_coefficients: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    Given a list of indices for ... x N datapoints,
      select the row from lmc_coefficient that corresponds to each datapoint

    lmc_coefficients: torch.Tensor ... x num_latents x ... x num_tasks
    indices: torch.Tesnor ... x N
    """
    batch_shape = torch.broadcast_shapes(lmc_coefficients.shape[:-1], indices.shape[:-1])

    # We will use the left_interp helper to do the indexing
    lmc_coefficients = lmc_coefficients.expand(*batch_shape, lmc_coefficients.shape[-1])[..., None]
    indices = indices.expand(*batch_shape, indices.shape[-1])[..., None]
    res = left_interp(
        indices,
        torch.ones(indices.shape, dtype=torch.long, device=indices.device),
        lmc_coefficients,
    ).squeeze(-1)
    return res


class OutputCovarianceLMCVariationalStrategy(_VariationalStrategy):
    r"""

    """

    def __init__(
        self,
        base_variational_strategy: _VariationalStrategy,
        output_kernel: Kernel,
        output_covars: Tensor,
        num_tasks: int,
        num_latents: int = 1,
        latent_dim: int = -1,
        jitter_val: Optional[float] = None,
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        batch_shape = self.base_variational_strategy._variational_distribution.batch_shape

        # Check if no functions
        if latent_dim >= 0:
            raise RuntimeError(f"latent_dim must be a negative indexed batch dimension: got {latent_dim}.")
        if not (batch_shape[latent_dim] == num_latents or batch_shape[latent_dim] == 1):
            raise RuntimeError(
                f"Mismatch in num_latents: got a variational distribution of batch shape {batch_shape}, "
                f"expected the function dim {latent_dim} to be {num_latents}."
            )
        # Check if number of latents larger than number of outputs
        if num_tasks < num_latents:
            raise RuntimeError(f"number of latents must be smaller than number of outputs!")
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Make the batch_shape
        self.batch_shape = list(batch_shape)
        del self.batch_shape[self.latent_dim]
        self.batch_shape = torch.Size(self.batch_shape)

        # LCM coefficients generated from the kernel over the outputs
        # actual definition is in lmc_coefficient property below
        self.output_kernel = output_kernel
        self.output_covars = output_covars

        if jitter_val is None:
            self.jitter_val = settings.variational_cholesky_jitter.value(
                self.base_variational_strategy.inducing_points.dtype
            )
        else:
            self.jitter_val = jitter_val

    @property
    def lmc_coefficients(self) -> Tensor:
        evals, evecs = self.output_kernel(self.output_covars).symeig(eigenvectors=True)
        evecs = evecs[:, -self.num_latents:]
        evals = linear_operator.operators.DiagLinearOperator(evals[-self.num_latents:])
        return evecs.matmul(evals.sqrt()).t().evaluate()

    @property
    def prior_distribution(self) -> MultivariateNormal:
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self) -> MultivariateNormal:
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self) -> bool:
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self) -> Tensor:
        return super().kl_divergence().sum(dim=self.latent_dim)

    def __call__(
        self, x: Tensor, prior: bool = False, task_indices: Optional[LongTensor] = None,
            task_covars: Optional[Tensor] = None, **kwargs
    ) -> Union[MultitaskMultivariateNormal, MultivariateNormal]:
        r"""
        Computes the variational (or prior) distribution
        :math:`q( \mathbf f \mid \mathbf X)` (or :math:`p( \mathbf f \mid \mathbf X)`).
        There are two modes:

        1.  Compute **all tasks** for all inputs.
            If this is the case, the task_indices attribute should be None.
            The return type will be a (... x N x num_tasks)
            :class:`~gpytorch.distributions.MultitaskMultivariateNormal`.
        2.  Compute **one task** per inputs.
            If this is the case, the (... x N) task_indices tensor should contain
            the indices of each input's assigned task.
            The return type will be a (... x N)
            :class:`~gpytorch.distributions.MultivariateNormal`.

        :param x: (... x N x D) Input locations to evaluate variational strategy
        :param task_indices: (Default: None) Task index associated with each input.
            If this **is not** provided, then the returned distribution evaluates every input on every task
            (returns :class:`~gpytorch.distributions.MultitaskMultivariateNormal`).
            If this **is** provided, then the returned distribution evaluates each input only on its assigned task.
            (returns :class:`~gpytorch.distributions.MultivariateNormal`).
        :param prior: (Default: False) If False, returns the variational distribution
            :math:`q( \mathbf f \mid \mathbf X)`.
            If True, returns the prior distribution
            :math:`p( \mathbf f \mid \mathbf X)`.
        :return: :math:`q( \mathbf f \mid \mathbf X)` (or the prior),
            either for all tasks (if `task_indices == None`)
            or for a specific task (if `task_indices != None`).
        :rtype: ~gpytorch.distributions.MultitaskMultivariateNormal (... x N x num_tasks)
            or ~gpytorch.distributions.MultivariateNormal (... x N)
        """
        latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        if task_indices is None:
            num_dim = num_batch + len(latent_dist.event_shape)

            # Every data point will get an output for each task
            # Therefore, we will set up the lmc_coefficients shape for a matmul
            if task_covars is None:
                lmc_coefficients = self.lmc_coefficients.expand(*latent_dist.batch_shape,
                                                                self.lmc_coefficients.size(-1))
            else:
                k_star = self.output_kernel(task_covars,self.output_covars)
                anew = torch.linalg.solve(self.lmc_coefficients.matmul(self.lmc_coefficients.t()),
                                          self.lmc_coefficients.matmul(k_star.evaluate().unsqueeze(-1))).squeeze(-1).t()
                lmc_coefficients = anew.expand(*latent_dist.batch_shape, anew.size(-1))
            # Mean: ... x N x num_tasks
            latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
            mean = latent_mean @ lmc_coefficients.permute(
                *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
            )

            # Covar: ... x (N x num_tasks) x (N x num_tasks)
            latent_covar = latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(self.jitter_val)

            # Done!
            function_dist = MultitaskMultivariateNormal(mean, covar)

        else:
            # Each data point will get a single output corresponding to a single task
            # Therefore, we will select the appropriate lmc coefficients for each task
            if task_covars is None:
                lmc_coefficients = _select_lmc_coefficients(self.lmc_coefficients, task_indices)
            else:
                k_star = self.output_kernel(task_covars, self.output_covars)
                anew = torch.linalg.solve(self.lmc_coefficients.matmul(self.lmc_coefficients.t()),
                                          self.lmc_coefficients.matmul(k_star.evaluate().unsqueeze(-1))).squeeze(-1).t()
                lmc_coefficients = _select_lmc_coefficients(anew, task_indices)

            # Mean: ... x N
            mean = (latent_dist.mean * lmc_coefficients).sum(latent_dim)

            # Covar: ... x N x N
            latent_covar = latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = (latent_covar * lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(self.jitter_val)

            # Done!
            function_dist = MultivariateNormal(mean, covar)

        return function_dist