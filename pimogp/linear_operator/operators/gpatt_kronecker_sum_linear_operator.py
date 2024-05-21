from linear_operator.operators.sum_linear_operator import SumLinearOperator
from linear_operator.operators.diag_linear_operator import DiagLinearOperator

from .gpatt_kronecker_sum_added_diag_linear_operator import GPattKroneckerSumAddedDiagLinearOperator


class GPattKroneckerSumLinearOperator(SumLinearOperator):
    """
        Class to wrap a sum of Kronecker products, but ensure we stay inside the GPatt family of operators
        Simple extension of SumLazyTensor with custom __add__ routine, ensuring that a pass through the likelihood
        yields a GPattKroneckerSumAddedDiagLinearOperator
    """

    def __add__(self, other):
        if isinstance(other, DiagLinearOperator):
            return GPattKroneckerSumAddedDiagLinearOperator(self, other, preconditioner_override=None)
        else:
            raise RuntimeError("Invalid addition")
