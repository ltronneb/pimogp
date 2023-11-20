from typing import Any, Optional, Union

from gpytorch import Module, settings
from gpytorch.constraints import GreaterThan
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import _GaussianLikelihoodBase
import torch
import warnings
from gpytorch.utils.warnings import GPInputWarning
from linear_operator import LinearOperator
from linear_operator.operators import DiagLinearOperator, ZeroLinearOperator
from linear_operator.utils.warnings import NumericalWarning
from torch import Tensor


class FixedNoiseMultitaskGaussianLikelihood(_GaussianLikelihoodBase):
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


    @property
    def global_noise(self):
        nnn = self.raw_global_noise_constraint.transform(self.raw_global_noise)
        return nnn

    @global_noise.setter
    def global_noise(self, value):
        self.initialize(raw_global_noise=self.raw_global_noise_constraint.inverse_transform(value))

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()
        covar_kron_lt = self._shaped_noise_covar(mean.shape, *params, add_noise=True, **kwargs)
        covar = covar + covar_kron_lt
        return function_dist.__class__(mean, covar)

    def _shaped_noise_covar(self, shape, *params, add_noise=True,  **kwargs):
        noise = self.noise_covar(*params, shape=shape, **kwargs)
        # Here now, we add a diagonal for the added global noise
        if add_noise and self.has_global_noise:
            glob_noise = DiagLinearOperator(self.global_noise)
            noise = noise + glob_noise
        elif isinstance(noise, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )
        return noise


class MultitaskFixedGaussianNoise(Module):
    def __init__(self, noise: Tensor, num_tasks=1) -> None:
        super().__init__()
        min_noise = settings.min_fixed_noise.value(noise.dtype)
        if noise.lt(min_noise).any():
            warnings.warn(
                "Very small noise values detected. This will likely "
                "lead to numerical instabilities. Rounding small noise "
                f"values up to {min_noise}.",
                NumericalWarning,
            )
            noise = noise.clamp_min(min_noise)
        self.noise = noise
        self.num_tasks = num_tasks

    def forward(
            self, *params: Any, shape: Optional[torch.Size] = None, noise: Optional[Tensor] = None, **kwargs: Any
    ) -> Union[DiagLinearOperator, ZeroLinearOperator]:

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
