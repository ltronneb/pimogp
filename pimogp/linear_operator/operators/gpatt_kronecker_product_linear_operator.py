from linear_operator.operators.kronecker_product_linear_operator import KroneckerProductLinearOperator
from .gpatt_kronecker_sum_linear_operator import GPattKroneckerSumLinearOperator


class GPattKroneckerProductLinearOperator(KroneckerProductLinearOperator):
    """
        Simple class to wrap a Kronecker product such that we can define a custom pre-conditioner and log-determinant
        calculation
    """

    def __init__(self, linop):
        if not isinstance(linop, KroneckerProductLinearOperator):
            raise RuntimeError("The GPattKroneckerProductLinearOperator can only wrap a KroneckerProductLinearOperator")
        super().__init__(linop)

    def __add__(self, other):
        if isinstance(other, GPattKroneckerProductLinearOperator):
            return GPattKroneckerSumLinearOperator(self, other)
        else:
            raise RuntimeError("Invalid addition")
