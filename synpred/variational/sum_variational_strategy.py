# I need a new variational strategy
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import _VariationalStrategy
from torch import Tensor, cat
from typing import List


def _concat_distributions(distr: List[MultivariateNormal], jitter_val) -> MultivariateNormal:
    mean = cat([dist.mean for dist in distr], 0)
    covar = cat([dist.lazy_covariance_matrix.add_jitter(jitter_val).evaluate() for dist in distr], 0)
    return MultivariateNormal(mean, covar)


class SumVariationalStrategy(_VariationalStrategy):
    def __init__(self, model1: ApproximateGP, model2: ApproximateGP):
        Module.__init__(self)

        self.model1 = model1
        self.model2 = model2
        self.jitter_val = gpytorch.settings.variational_cholesky_jitter.value(
            self.model1.variational_strategy.base_variational_strategy.inducing_points.dtype
        )

    @property
    def prior_distribution(self) -> MultivariateNormal:
        A = self.model1.variational_strategy.prior_distribution
        B = self.model2.variational_strategy.prior_distribution
        return _concat_distributions([A, B], self.jitter_val)

    @property
    def variational_params_initialized(self) -> bool:
        return self.model1.variational_strategy.variational_params_initialized

    @property
    def variational_distribution(self) -> MultivariateNormal:
        A = self.model1.variational_strategy.variational_distribution
        B = self.model2.variational_strategy.variational_distribution
        return _concat_distributions([A, B], self.jitter_val)

    def kl_divergence(self) -> Tensor:
        return super().kl_divergence().sum(-1)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> gpytorch.distributions.MultivariateNormal:
        latent_dist1 = self.model1(x, prior=prior, **kwargs)
        latent_dist2 = self.model2(x, prior=prior, **kwargs)
        function_dist = latent_dist1 + latent_dist2
        return function_dist