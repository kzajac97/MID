import numpy as np
from numba import jit


# File contains implementations of analytically computed
# probability density functions for some known distributions


@jit
def triangle(x: float) -> float:
    """
     Returns values from triangle distribution, described by formula:
     .. math:: 0 for x in (-inf, -1) and [1, inf)
                x + 1 for x in [-1, 0)
               -x + 1 for x in [0, 1)

    :param x:  value at which distribution will be evaluated
    """
    if x < -1 or x > 1:
        return 0

    return x + 1 if x < 0 else -x + 1


@jit
def exponential(x: float, rate: float = 1.0) -> float:
    """
    Returns values from exponential distribution, described by formula:
    .. math:: lambda exp(-lambda x)

    :param x: value at which distribution will be evaluated
    :param rate: lambda parameter of distribution
    """
    return rate * np.exp(-1 * rate * x) if x > 0 else 0


@jit
def cauchy(x: float, x0: float = 0.0, gamma: float = 0.5) -> float:
    """
    Returns values from Cauchy distribution
    see: https://en.wikipedia.org/wiki/Cauchy_distribution

    :param x: value at which distribution will be evaluated
    :param x0: location parameter
    :param gamma: scale parameter
    """
    ratio = ((x - x0) / gamma) ** 2
    return 1 / (np.pi * gamma * (1 + ratio))


@jit
def laplace(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """
    Returns values from Laplace distribution
    see: https://en.wikipedia.org/wiki/Laplace_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param b: scale parameter, in range (0, inf)
    """
    exponent = -1 * np.abs(x - mu) / b
    return 1 / (2 * b) * np.exp(exponent)


@jit
def logistic(x: float, mu: float = 1.0, s: float = 1.0) -> float:
    """
    Returns values from logistic distribution
    see: https://en.wikipedia.org/wiki/Logistic_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param s: scale parameter
    """
    exponent = -(x - mu) / s
    return np.exp(exponent) / (s * (1 + np.exp(exponent) ** 2))
