import warnings
from typing import Optional, Tuple

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.functions import RBFCovariance
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from gpytorch.priors import Prior
from gpytorch.settings import trace_mode
from torch import Tensor


class PermutationInvariantRBFKernel(Kernel):
    has_lengthscale = True

    # Overwrite the init method here because we need to change some stuff
    def __init__(
        self,
        permutation: Tensor,
        permute_forward: bool=False,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super(PermutationInvariantRBFKernel, self).__init__()

        # Initialise the indexing for the permutation
        seen_pairs = set()
        indices = torch.empty(permutation.shape[0]).long()
        idx = 0
        for i, j in enumerate(permutation):
            # Sort the pair so that (i, j) and (j, i) look the same in the set
            pair = tuple(sorted((i, j.item())))

            # Process the pair only if it hasn't been seen before
            if pair not in seen_pairs:
                print("Index", idx, "given to: ", pair)
                # Set corresponding entry in indices
                indices[list(pair)] = idx
                # Mark this pair as seen
                seen_pairs.add(pair)
                idx += 1
        self.indices = indices

        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = eps

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'lengthscale_constraint' instead.",
                DeprecationWarning,
            )
        # We divide ard_num_dims by 2 to deal with the permutation
        lengthscale_num_dims = 1 if ard_num_dims is None else int(ard_num_dims/2.0)
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
        )
        if lengthscale_prior is not None:
            if not isinstance(lengthscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
            self.register_prior(
                "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
            )

        self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

        self.permutation = permutation

        self.permute_forward = permute_forward

    @property
    def lengthscale(self) -> Tensor:
        # This returns the correct doubled version of the lengthscale
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale.index_select(1, self.indices))

    @lengthscale.setter
    def lengthscale(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        reduced_inits = torch.empty(self.indices.max() + 1)
        for i in range(self.indices.max() + 1):
            mask = self.indices == i
            reduced_inits[i] = value[mask].mean()

        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(reduced_inits))


    def forward(self, x1, x2, diag=False, **params):
        if (
                x1.requires_grad
                or x2.requires_grad
                or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                or diag
                or params.get("last_dim_is_batch", False)
                or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            if self.permute_forward:
                return (postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)) +
                        postprocess_rbf(
                            self.covar_dist(torch.index_select(x1_, -1, self.permutation), x2_, square_dist=True,
                                            diag=diag, **params)) #+
                        #postprocess_rbf(
                        #    self.covar_dist(x1_, torch.index_select(x2_, -1, self.permutation), square_dist=True,
                        #                    diag=diag, **params)) +
                        #postprocess_rbf(self.covar_dist(torch.index_select(x1_, -1, self.permutation),
                        #                                torch.index_select(x2_, -1, self.permutation), square_dist=True,
                        #                                diag=diag, **params)))
                        )
            else:
                return postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))

        if self.permute_forward:

            return (RBFCovariance.apply(x1, x2, self.lengthscale, lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params)) +
                    RBFCovariance.apply(torch.index_select(x1,-1,self.permutation), x2, self.lengthscale, lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params)) +
                    RBFCovariance.apply(x1, torch.index_select(x2,-1,self.permutation), self.lengthscale, lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params)) +
                    RBFCovariance.apply(torch.index_select(x1,-1,self.permutation),  torch.index_select(x2,-1,self.permutation), self.lengthscale, lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params)))



        else:
            return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params)
        )