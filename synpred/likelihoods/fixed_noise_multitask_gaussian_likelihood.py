from typing import Any, Optional

from gpytorch import Module
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
import torch
from linear_operator.operators import DiagLinearOperator, ZeroLinearOperator
from torch import Tensor


class FixedNoiseMultitaskGaussianLikelihood(GaussianLikelihood):
    """
        A multitask extension of FixedNoiseGaussianLikelihood
        """

    def __init__(self,
                 num_tasks,
                 noise: Tensor,
                 batch_shape=torch.Size(),
                 noise_prior=None,
                 noise_constraint=None,
                 has_global_noise=True,
                 **kwargs: Any) -> None:
        super().__init__(noise_covar=MultitaskFixedGaussianNoise(noise=noise, num_tasks=num_tasks))
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        if has_global_noise:
            self.register_parameter(name="raw_global_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_global_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_global_noise_prior", noise_prior, lambda m: m.noise)
        self.has_global_noise = has_global_noise
        self.num_tasks = num_tasks


class MultitaskFixedGaussianNoise(Module):
    def __init__(self, noise: Tensor, num_tasks=1) -> None:
        super().__init__()
        self.noise = noise
        self.num_tasks = num_tasks

    def forward(
            self, *params: Any, shape: Optional[torch.Size] = None, noise: Optional[Tensor] = None, **kwargs: Any
    ) -> DiagLinearOperator:

        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        if noise is not None:
            return DiagLinearOperator(noise)
        if shape == self.noise.reshape(-1).shape:
            return DiagLinearOperator(self.noise.reshape(-1))
        elif shape[-2] == self.noise.shape[-2]:
            return DiagLinearOperator(self.noise.reshape(-1))
        else:
            return ZeroLinearOperator()

    def _apply(self, fn):
        self.noise = fn(self.noise)
        return super(MultitaskFixedGaussianNoise, self)._apply(fn)
