from typing import Callable, Optional, Tuple, Union

from linear_operator.operators import SumLinearOperator
from linear_operator.operators.diag_linear_operator import DiagLinearOperator
import torch
from jaxtyping import Float
from linear_operator import LinearOperator


class GPattKroneckerSumAddedDiagLinearOperator(SumLinearOperator):
    """
        Linear operator to deal with the case where the base-kernel is a GPattKroneckerSumLinearOperator
        This is the case when we work with the invariant-kernels used in the DrugCombinationKernel
    """

    def __init__(
        self,
        *linear_ops: Union[Tuple[LinearOperator, DiagLinearOperator], Tuple[DiagLinearOperator, LinearOperator]],
        preconditioner_override: Optional[Callable] = None,
    ):
        linear_ops = list(linear_ops)
        super(GPattKroneckerSumAddedDiagLinearOperator, self).__init__(*linear_ops, preconditioner_override=preconditioner_override)
        if len(linear_ops) > 2:
            raise RuntimeError("An AddedDiagLinearOperator can only have two components")

        if isinstance(linear_ops[0], DiagLinearOperator) and isinstance(linear_ops[1], DiagLinearOperator):
            raise RuntimeError(
                "Trying to lazily add two DiagLinearOperators. Create a single DiagLinearOperator instead."
            )
        elif isinstance(linear_ops[0], DiagLinearOperator):
            self._diag_tensor = linear_ops[0]
            self._linear_op = linear_ops[1]
        elif isinstance(linear_ops[1], DiagLinearOperator):
            self._diag_tensor = linear_ops[1]
            self._linear_op = linear_ops[0]
        else:
            raise RuntimeError(
                "One of the LinearOperators input to AddedDiagLinearOperator must be a DiagLinearOperator!"
            )

        self.preconditioner_override = preconditioner_override
        self.missing_idx = (self._diag_tensor._diagonal() >= 500.).clone().detach()  # Hardcoded which is not optimal
        self.n_missing = self.missing_idx.sum()
        self.n_total = self.missing_idx.numel()
        self.n_obs = self.n_total - self.n_missing

    def add_diag(self, added_diag):
        raise RuntimeError("Tried to add diag to GPattKroneckerSumAddedDiagLinearOperator")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
        else:
            inv_quad_term = None
        logdet_term = self._logdet() if logdet else None
        return inv_quad_term, logdet_term

    def _logdet(self):
        """
        Log-determinant computed uses an approximation via Weyl's inequality
        """
        print("computing log-det with Weyl's inequality!")
        # Compute eigenvectors for gradients
        # It suffices to eigen-decomp the second term in the KroneckerSum
        evals_unsorted, _ = self._lazy_tensor.lazy_tensors[1].symeig(eigenvectors=False)
        evals = evals_unsorted.sort(descending=True)[0]
        # Clamp to zero
        evals = evals.clamp_min(0.0)
        # And multiply by two
        evals = 2 * evals
        # Pull out the constant diagonal
        noise_unsorted = self._diag_tensor.diag()
        noise_unsorted = noise_unsorted.masked_fill(self.missing_idx, 0)  # Mask large variances
        noise = noise_unsorted.sort(descending=True)[0]
        # Apply Weyl's inequality
        print(len(evals))
        weyl = torch.zeros(evals.shape, device=self.device)
        weyl[0::2] = evals[0:int(len(evals) / 2):1] + noise[0:int(len(evals) / 2):1]
        weyl[1::2] = evals[1:int(len(evals) / 2 + 1):1] + noise[0:int(len(evals) / 2):1]
        top_evals = (self.n_obs / self.n_total) * weyl[:self.n_obs]
        logdet = torch.log(top_evals).sum(dim=-1)
        return logdet

    def _preconditioner(self):
        def precondition_closure(tensor):
            return tensor / self._diag_tensor.diag().unsqueeze(-1)

        return precondition_closure, None, None

    def _solve(
        self: Float[LinearOperator, "... N N"],
        rhs: Float[torch.Tensor, "... N C"],
        preconditioner: Optional[Callable[[Float[torch.Tensor, "... N C"]], Float[torch.Tensor, "... N C"]]] = None,
        num_tridiag: Optional[int] = 0,
    ) -> Union[
        Float[torch.Tensor, "... N C"],
        Tuple[
            Float[torch.Tensor, "... N C"],
            Float[torch.Tensor, "..."],  # Note that in case of a tuple the second term size depends on num_tridiag
        ],
    ]:
        # CG for solves
        return super()._solve(rhs, preconditioner=preconditioner, num_tridiag=num_tridiag)

    def __add__(self, other):
        if isinstance(other, DiagLinearOperator):
            return self.__class__(self._lazy_tensor, self._diag_tensor + other)
        else:
            raise RuntimeError("Only DiagLazyTensors can be added to a GPattKroneckerSumAddedDiagLazyTensor!")
