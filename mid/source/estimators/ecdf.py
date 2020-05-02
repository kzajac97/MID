import numpy as np


class ECDF:
    """
    Empirical Cumulative Distribution Function Estimator
    """
    def __init__(self, realization: np.array):
        """
        :param realization: random variable realization for which
                            cumulative distribution will be estimated,
                            must have at least 2 samples
        """
        self.density = np.sort(realization)

    def _value(self, x):
        # np.where works like indicator function
        return np.sum(np.where(x >= self.density)[0].shape[0]) / self.density.shape[0]

    def __call__(self, x: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param x: value or array of values to compute CDF for

        :return: array of CDF values
        """
        return np.vectorize(self._value)(x)
