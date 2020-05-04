import numpy as np
from sklearn.exceptions import NotFittedError


FIRST_ITEM_INDEX = 0
SECOND_ITEM_INDEX = 1


class ECDF:
    """
    Empirical Cumulative Distribution Function Estimator
    """
    def __init__(self):
        self._density = None

    def _estimate(self, x: float) -> float:
        # np.where works like indicator function
        if self._density is None:
            raise NotFittedError

        return np.sum(np.where(x >= self._density)[FIRST_ITEM_INDEX].shape[0]) / self._density.shape[0]

    def fit(self, realization: np.array) -> None:
        """
        :param realization: random variable realization for which
                            cumulative distribution will be estimated,
                            must have at least 2 samples
        """
        self._density = np.sort(realization)

    def predict(self, values: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param values: value or array of values to compute CDF for

        :raises NotFittedError: when called prior to calling fit function

        :return: array of CDF values
        """
        return np.vectorize(self._estimate)(values)


class Histogram:
    """
    Histogram Probability Density Estimator

    :warning: bins or width must be specified for histogram constructor
    """
    def __init__(self, width: float = None, bins: int = None, normalized: bool = True):
        """
        :param bins: number of bins in histogram
        :param width: width of histogram
        :param normalized: if True values will be divided by number of samples times histogram width
        """
        self._distribution = None
        self._spacing = None
        self.bins = bins
        self.width = width
        self.normalized = normalized

    def fit(self, realization: np.array) -> None:
        """
        :param realization: random variable realization for which
                            probability density  will be estimated,
                            must have at least 2 samples
        """
        self._distribution = realization

        if self.width:
            self._spacing = np.arange(realization.min(), realization.max(), self.width)
            # number of bins is the number of point generated in spacing
            self.bins = self._spacing.shape[FIRST_ITEM_INDEX]

        if self.bins:
            self._spacing = np.linspace(realization.min(), realization.max(), self.bins)
            # width is difference between point in spacing
            self.width = self._spacing[SECOND_ITEM_INDEX] - self._spacing[FIRST_ITEM_INDEX]

    @property
    def normalization_factor(self) -> float:
        """
        :return: normalization factor is (1 / N h),
                 where N is number of samples in fitted distribution and h is histogram width
        """
        if self.normalized:
            return 1.0 / (self.width * self._distribution.shape[FIRST_ITEM_INDEX])

        return 1.0

    def _estimate(self, x: float) -> float:
        if self._distribution is None:
            raise NotFittedError

        nearest_value_index = np.abs(self._spacing - x).argmin()  # grid item nearest to called value
        return self.normalization_factor * np.count_nonzero(
            # number of points from distribution between two grid items
            np.logical_and(
                (self._distribution <= self._spacing[nearest_value_index]),
                (self._spacing[nearest_value_index] <= self._distribution + self.width),
            ).astype(np.int)
        )

    def predict(self, values: float) -> float:
        """
        Returns cumulative distribution function value at any point

        :param values: value or array of values to compute histogram for

        :raises NotFittedError: when called prior to calling fit function

        :return: array of histogram values
        """
        return np.vectorize(self._estimate)(values)


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
