import numpy as np
from numba import jit


# File contains implementations of analytically computed
# cumulative distributions for some known probability density functions


@jit
def cumulative_triangle(x: float) -> float:
    """
    Returns values from cumulative triangle distribution function

    :param x:  value at which distribution will be evaluated
    """
    if x < -1:
        return 0

    if x > 1:
        return 1

    return 0.5 + x + 0.5 * x ** 2 if x < 0 else 0.5 + x - 0.5 * x ** 2


@jit
def cumulative_exponential(x: float, rate: float = 1.0) -> float:
    """
    Returns values from cumulative exponential distribution

    :param x: value at which distribution will be evaluated
    :param rate: lambda parameter of distribution
    """
    return 1 - np.exp(-1 * rate * x) if x > 0 else 0


@jit
def cumulative_cauchy(x: float, x0: float = 0.0, gamma: float = 0.5) -> float:
    """
    Returns values from cumulative Cauchy distribution
    see: https://en.wikipedia.org/wiki/Cauchy_distribution

    :param x: value at which distribution will be evaluated
    :param x0: location parameter
    :param gamma: scale parameter
    """
    ratio = (x - x0) / gamma
    return 0.5 + (np.arctan(ratio) / np.pi)


@jit
def cumulative_laplace(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """
     Returns values from cumulative Laplace distribution
    see: https://en.wikipedia.org/wiki/Laplace_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param b: scale parameter, in range (0, inf)
    """
    exponent = -1 * np.abs(x - mu) / b
    return 0.5 * np.exp(exponent) if x < mu else 1 - 0.5 * np.exp(exponent)


@jit
def cumulative_logistic(x: float, mu: float = 1.0, s: float = 1.0) -> float:
    """
    Returns values from cumulative logistic distribution
    see: https://en.wikipedia.org/wiki/Logistic_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param s: scale parameter
    """
    exponent = -(x - mu) / s
    return 1 / (1 + np.exp(exponent))
