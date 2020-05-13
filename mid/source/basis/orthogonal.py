from typing import Callable

import numpy as np

from source.basis.geneators import cosine_basis

BASIS_GENERATORS = {
    "cos": cosine_basis,
    "cosine": cosine_basis,
}


class OrthogonalBasis:
    """
    Class holds interface to callable basis function set
    """
    def __init__(self, n_functions: int, basis_generator: Callable[[int], list]):
        """
        :param n_functions: number of functions to generate
        :param basis_generator: basis generator function, should return list of callable objects
        """
        self.basis = basis_generator(n_functions)

    def _point_basis(self, x):
        return np.array([self.basis[index](x) for index in range(len(self.basis))])

    def __call__(self, values: float):
        return np.vectorize(self._point_basis, signature="()->(n)")(values)
