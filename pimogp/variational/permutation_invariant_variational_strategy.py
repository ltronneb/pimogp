# Lets first set up our own Variational Strategy, just so we can see the flow of things:
# just copy and paste in the default one with a new name:
import warnings
from typing import Any, Dict, Iterable, Optional

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    LinearOperator,
    MatmulLinearOperator,
    SumLinearOperator,
)
from torch import Tensor

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import pop_from_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational import _VariationalDistribution, VariationalStrategy


def _ensure_updated_strategy_flag_set(
        state_dict: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: Iterable[str],
        unexpected_keys: Iterable[str],
        error_msgs: Iterable[str],
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class PermutationInvariantVariationalStrategy(VariationalStrategy):
    def __init__(self,
                 model: ApproximateGP,
                 inducing_points: torch.Tensor,
                 variational_distribution: _VariationalDistribution,
                 permutation: torch.Tensor,
                 learn_inducing_locations: bool = True,
                 jitter_val: Optional[float] = None,
                 ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True
        ## TODO add a check to see if permutation is valid
        self.permutation = permutation

    def forward(
            self,
            x: Tensor,
            inducing_points: Tensor,
            inducing_values: Tensor,
            variational_inducing_covar: Optional[LinearOperator] = None,
            **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x, torch.index_select(x, -1, self.permutation)], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix
        # Covariance terms
        num_induc = inducing_points.size(-2)
        num_x = x.size(-2)
        test_mean = full_output.mean[..., num_induc:(num_induc + num_x)]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = (full_covar[..., :num_induc, num_induc:(num_induc + num_x)] +
                            full_covar[..., :num_induc, (num_induc + num_x):]).to_dense()
        data_data_covar = (full_covar[..., num_induc:(num_induc + num_x), num_induc:(num_induc + num_x)] +
                           full_covar[..., num_induc:(num_induc + num_x), (num_induc + num_x):] +
                           full_covar[..., (num_induc + num_x):, num_induc:(num_induc + num_x)] +
                           full_covar[..., (num_induc + num_x):, (num_induc + num_x):])

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                    data_data_covar.add_jitter(self.jitter_val).to_dense()
                    + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)