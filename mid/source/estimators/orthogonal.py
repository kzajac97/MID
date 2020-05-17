from typing import Union

import numpy as np
from sklearn.exceptions import NotFittedError

from source.basis.generators import BASIS_GENERATORS
from source.basis.orthogonal import OrthogonalBasis


class OrthogonalEstimator:
    """
    Orthogonal Regression Estimator
    """
    def __init__(self, n_functions: int, basis: Union[str, OrthogonalBasis]):
        """
        :param n_functions: number of functions used for building orthogonal base
        :param basis: str or build orthogonal base
        """
        self._basis = OrthogonalBasis(n_functions, BASIS_GENERATORS[basis]) if type(basis) is str else basis
        self._alphas = None
        self._betas = None

    def fit(self, x, y):
        """
        Fit estimator to feature, label pairs

        :param x: input value for kernel regression
        :param y: targets for kernel regression
        """
        self._alphas = (1 / x.shape[0]) * np.sum(self._basis(x), axis=0)
        self._betas = (1 / x.shape[0]) * np.sum((self._basis(x).T * y).T, axis=0)

    def _estimate(self, x):
        if self._alphas is None:
            raise NotFittedError

        return np.sum(self._betas * self._basis(x)) / np.sum(self._alphas * self._basis(x))

    def predict(self, values):
        """
        Returns cumulative distribution function value at any point

        :param values: value or array of values to compute histogram for

        :raises NotFittedError: when called prior to calling fit function

        :return: array of histogram values
        """
        return np.vectorize(self._estimate)(values)
