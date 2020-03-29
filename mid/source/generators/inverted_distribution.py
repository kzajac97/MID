from typing import Any, Callable, Tuple, Optional

import numpy as np
from numba import jit
from scipy import integrate

from source.functions.cumulative import (
    cumulative_cauchy,
    cumulative_exponential,
    cumulative_laplace,
    cumulative_logistic,
    cumulative_triangle,
)
from source.functions.inverse import (
    inverse_cauchy,
    inverse_exponential,
    inverse_laplace,
    inverse_logistic,
    inverse_triangle,
)

INTEGRAL_VALUE_INDEX = 0
INTEGRAL_ERROR_INDEX = 1
LOWER_RANGE_INDEX = 0
UPPER_RANGE_INDEX = 1

_DISTRIBUTION_TO_CUMULATIVE_DISTRIBUTION_MAPPING = {
    "cauchy": cumulative_cauchy,
    "exponential": cumulative_exponential,
    "laplace": cumulative_laplace,
    "logistic": cumulative_logistic,
    "triangle": cumulative_triangle,
}

_DISTRIBUTION_TO_INVERTED_CUMULATIVE_DISTRIBUTION_MAPPING = {
    "cauchy": inverse_cauchy,
    "exponential": inverse_exponential,
    "laplace": inverse_laplace,
    "logistic": inverse_logistic,
    "triangle": inverse_triangle,
}


@jit
def cumulative_distribution_function(
    random_distribution_function: Callable[[Tuple[float, ...]], float],
    x: float,
    infinity_approximation: float = np.inf,
    *args
) -> np.array:
    """
    :param random_distribution_function: random numbers distribution function
    :param x: point at which to evaluate
    :param infinity_approximation:
    :param args: additional args to pdf function

    :return: cumulative distribution function value at x
    """
    return integrate.quad(random_distribution_function, -1 * infinity_approximation, x, args)[INTEGRAL_VALUE_INDEX]


def _sample_analytic_distribution(
    n_samples: int,
    distribution: str,
    sample_range: Tuple[float, float] = (0.0, 1.0),
    uniform_sampler: Optional[Callable[[Any], np.array]] = None,
    **kwargs
) -> np.array:
    """
    Sample array of random variables distributed with chosen function

    :param n_samples: number of generated data points
    :param distribution: probability density function which will be generated, valid options are:
                         `cauchy`,
                         `exponential`
                         `laplace`
                         `logistic`
                         `triangle`
    :param sample_range: range in which numbers will be generated
    :param uniform_sampler: function generating uniform samples
    :param kwargs: additional arguments passed to inverse_distribution function

    :return: array with random data points following chosen distribution
    """
    uniform_sampler = uniform_sampler or np.random.rand
    x_samples = (sample_range[UPPER_RANGE_INDEX] - sample_range[LOWER_RANGE_INDEX]) * uniform_sampler(
        n_samples
    ) + sample_range[LOWER_RANGE_INDEX]

    samples = [
        _DISTRIBUTION_TO_INVERTED_CUMULATIVE_DISTRIBUTION_MAPPING[distribution](sample, **kwargs)
        for sample in x_samples
    ]
    return np.array(samples)
