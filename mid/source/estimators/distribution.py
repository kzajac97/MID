import numpy as np


FIRST_ITEM_INDEX = 0
SECOND_ITEM_INDEX = 1


class ECDF:
    """
    Empirical Cumulative Distribution Function Estimator

    example:
        >>> estimator = ECDF(realization)
        >>> cdf = estimator(np.linspace(-3, 3, 1000))
    """

    def __init__(self, realization: np.array):
        """
        :param realization: random variable realization for which
                            cumulative distribution will be estimated,
                            must have at least 2 samples
        """
        self.density = np.sort(realization)

    def _value(self, x: float) -> float:
        # np.where works like indicator function
        return np.sum(np.where(x >= self.density)[0].shape[0]) / self.density.shape[0]

    def __call__(self, x: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param x: value or array of values to compute CDF for

        :return: array of CDF values
        """
        return np.vectorize(self._value)(x)


class Histogram:
    """
    Histogram Probability Density Estimator

    example:
        >>> estimator = Histogram(realization, bins=10)
        >>> histogram = estimator(np.linspace(0, 5, 1000))

    :warning: bins or width must be specified for histogram constructor
    """

    def __init__(self, realization: np.array, bins: int = None, width: float = None, normalized: bool = True):
        """
        :param realization: random variable realization for which
                            probability density  will be estimated,
                            must have at least 2 samples
        :param bins: number of bins in histogram
        :param width: width of histogram
        :param normalized: if True values will be divided by number of samples times histogram width
        """
        self.distribution = realization
        self.grid = (
            np.arange(realization.min(), realization.max(), width)
            if width
            else np.linspace(realization.min(), realization.max(), bins)
        )
        self.normalization_factor = (1 / (self.width * self.distribution.shape[0])) if normalized else 1

    @property
    def width(self):
        """
        :return: spacing between grid elements
        """
        return self.grid[SECOND_ITEM_INDEX] - self.grid[FIRST_ITEM_INDEX]

    def _value(self, x: float) -> float:
        nearest_value_index = np.abs(self.grid - x).argmin()  # grid item nearest to called value
        return self.normalization_factor * np.count_nonzero(
            # number of points from distribution between two grid items
            np.logical_and(
                (self.distribution <= self.grid[nearest_value_index]),
                (self.grid[nearest_value_index] <= self.distribution + self.width),
            ).astype(np.int)
        )

    def __call__(self, x: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param x: value or array of values to compute histogram for

        :return: array of histogram values
        """
        return np.vectorize(self._value)(x)


class KernelEstimator:
    """
    Kernel Probability Density Estimator

    example:
        >>> estimator = Histogram(realization, bins=10)
        >>> histogram = estimator(np.linspace(0, 5, 1000))

    :warning: bins or width must be specified for histogram constructor
    """

    def __init__(self, realization: np.array):
        ...

    def _value(self, x: float) -> float:
        ...

    def __call__(self, x: float) -> float:
        ...
