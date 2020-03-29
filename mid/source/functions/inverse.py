import numpy as np
from numba import jit


# File contains implementations of analytically computed
# inverted functions for some known cumulative distributions


@jit
def inverse_triangle(u: float) -> float:
    """
    Returns values from inverted function for cumulative triangle distribution

    :param u:  value at which distribution will be evaluated, valid x are in range [0, 1]
    """
    return -1 + np.sqrt(2 * u) if u < 0.5 else 1 - np.sqrt(2 * (1 - u))


@jit
def inverse_exponential(u: float, rate: float = 1.0) -> float:
    """
    Returns values from inverted cumulative exponential distribution

    :param u: value at which distribution will be evaluated
    :param rate: lambda parameter of distribution
    """
    return np.log(1 / (u + 1)) / rate


@jit
def inverse_cauchy(u: float, x0: float = 0.0, gamma: float = 0.5) -> float:
    """
    Returns values from inverted cumulative Cauchy distribution
    see: https://en.wikipedia.org/wiki/Cauchy_distribution

    :param u: value at which distribution will be evaluated
    :param x0: location parameter
    :param gamma: scale parameter
    """
    return x0 + (gamma * np.tan(np.pi * (u - 0.5)))


@jit
def inverse_laplace(u: float, mu: float = 0.0, b: float = 1.0) -> float:
    """
     Returns values from inverted cumulative Laplace distribution
    see: https://en.wikipedia.org/wiki/Laplace_distribution

    :param u: value at which distribution will be evaluated
    :param mu: location parameter
    :param b: scale parameter, in range (0, inf)
    """
    return mu + b * np.log(2 * u) if u < mu else mu - b * np.log(-2 * (u + 1))


@jit
def inverse_logistic(u: float, mu: float = 1.0, s: float = 1.0) -> float:
    """
    Returns values from inverted cumulative logistic distribution
    see: https://en.wikipedia.org/wiki/Logistic_distribution

    :param u: value at which distribution will be evaluated
    :param mu: location parameter
    :param s: scale parameter
    """
    return mu + s * np.log(u / (1 - u))
