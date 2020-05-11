from typing import Callable, Union

import numpy as np
from sklearn.exceptions import NotFittedError

from source.kernels.density import DENSITY_KERNELS


class KernelRegressor:
    """

    """
    def __init__(
            self, kernel: Union[str, Callable[[float, float], float]], width: float = None, precision: int = 10_000
    ):
        """
        :param kernel: str or callable kernel, if str valid options are:
                       `gau` or `gaussian`: gaussian_kernel
                       `ep` or `epenechikov`: epenechikov_kernel
                       `cos` or `cosine`: cosine kernel
                       `top` or `tophat`: histogram like kernel
                       `lin` or `linear`: linear monotonic kernel
                       `exp` or `exponential`: exponential kernel
        :param width: width of estimators kernel
        :param precision: optional parameter defining precision of rounding point during predict
        """
        # public properties
        self.width = width
        # private params
        self._kernel = DENSITY_KERNELS[kernel] if type(kernel) is str else kernel
        self._precision = precision
        # placeholders
        self._spacing = None
        self._values = None
        self._targets = None

    def fit(self, x: np.array, y: np.array) -> None:
        """
        Fit estimator to feature, label pairs

        :param x: input value for kernel regression
        :param y: targets for kernel regression
        """
        self._spacing = np.linspace(x.min(), x.max(), self._precision)
        self._values = np.sum(np.vectorize(self._point_kernel, signature="()->(n)")(x), axis=0)
        self._targets = np.sum((self._values.T * y).T)  # double transpose multiplication

    def _point_kernel(self, point: float):
        return self._kernel((self._spacing - point) / self.width, self.width)

    def _estimate(self, x: float) -> float:
        if self._targets is None:
            raise NotFittedError

        nearest_value = np.argmin(np.abs(self._spacing - x))
        return self._targets[nearest_value] / self._values[nearest_value]

    def predict(self, values: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param values: value or array of values to compute histogram for

        :raises NotFittedError: when called prior to calling fit function

        :return: array of histogram values
        """
        return np.vectorize(self._estimate)(values)
