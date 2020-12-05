import torch
from torch import nn
import geotorch.parametrize as P
from geotorch.constructions import Manifold
from geotorch.exceptions import NonSquareError
from geotorch.symmetric import Symmetric

try:
    from torch import matrix_exp as expm
except ImportError:
    from geotorch.linalg.expm import expm


class PSDReductive(Manifold):
    def __init__(self, size, lower=True):
        r"""
        Manifold of symmetric positive definite matrices implemented through the exponential map

        Args:
            size (torch.size): Size of the tensor to be applied to
            lower (bool): Optional. Uses the lower triangular part of the matrix to parametrize
                the skew-symmetric matrices. Default: `True`
        """
        super().__init__(dimensions=2, size=size)
        if self.n != self.k:
            raise NonSquareError(self.__class__.__name__, size)

        # Precompose with Skew
        self.chain(Symmetric(size=size, lower=lower))
        self.wishart_init_()

    def trivialization(self, X):
        B = self.base
        # Compute B^{-1}X
        B1X = torch.solve(X, B).solution
        return B @ expm(B1X)

    def wishart_init_(self):
        r""" Wishart is a distribution on PSSD matrices, but will do for now,
        as the set of singualr matrices is of measure zero
        """
        B = self.base
        # Gain of 0.5 because we're going to compute its "square"
        torch.nn.init.xavier_normal_(B, gain=0.5)
        with torch.no_grad():
            B.data = B.transpose(-2, -1) @ B
            if self.is_registered():
                self.original_tensor().zero_()

    def extra_repr(self):
        return "n={}, triv={}".format(self.n, self.triv.__name__)


def positive_definite(module, tensor_name):
    r"""Adds a positive definiteness constraint to the tensor
    ``module[tensor_name]``.
    """
    size = getattr(module, tensor_name).size()
    M = PSDReductive(size)
    P.register_parametrization(module, tensor_name, M)
    with torch.no_grad():
        M.original_tensor().zero_()