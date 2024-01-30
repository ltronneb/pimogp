# I need a new variational strategy
import gpytorch
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import _VariationalStrategy
from torch import Tensor, cat
from typing import List

from torch.nn import ModuleList


def _concat_distributions(distr: List[gpytorch.distributions.MultivariateNormal], jitter_val) -> MultivariateNormal:
    mean = cat([dist.mean for dist in distr], 0)
    covar = cat([dist.lazy_covariance_matrix.add_jitter(jitter_val).evaluate() for dist in distr], 0)
    return MultivariateNormal(mean, covar)


class SumVariationalStrategy(_VariationalStrategy):
    # def __init__(self, model1: gpytorch.models.ApproximateGP, model2: gpytorch.models.ApproximateGP):
    def __init__(self, models: ModuleList):
        Module.__init__(self)
        self.models = models
        self.jitter_val = gpytorch.settings.variational_cholesky_jitter.value(
            (self.models[0]).variational_strategy.base_variational_strategy.inducing_points.dtype
        )

    @property
    def prior_distribution(self) -> MultivariateNormal:
        return _concat_distributions([model.variational_strategy.prior_distribution for model in self.models],
                                     self.jitter_val)

    @property
    def variational_params_initialized(self) -> bool:
        return self.models[0].variational_strategy.variational_params_initialized

    @property
    def variational_distribution(self) -> MultivariateNormal:
        return _concat_distributions([model.variational_strategy.variational_distribution for model in self.models],
                                     self.jitter_val)

    def kl_divergence(self) -> Tensor:
        return super().kl_divergence().sum(-1)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> gpytorch.distributions.MultivariateNormal:
        function_dist = sum([model(x, prior=prior, **kwargs) for model in self.models])

        return function_dist
