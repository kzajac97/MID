import numpy as np
from numba import jit


# File contains implementations of analytically computed
# inverted functions for some known cumulative distributions


@jit
def inverse_triangle(x: float) -> float:
    """
    Returns values from inverted function for cumulative triangle distribution

    :param x:  value at which distribution will be evaluated
    """
    if x <= -1:
        return 0

    if x >= 1:
        return 1

    return -1 + np.sqrt(2 * x) if x <= 0 else 1 - np.sqrt(2 * x)


@jit
def inverse_exponential(x: float, rate: float = 1.0) -> float:
    """
    Returns values from inverted cumulative exponential distribution

    :param x: value at which distribution will be evaluated
    :param rate: lambda parameter of distribution
    """
    return np.log(1 / (x + 1)) / rate


@jit
def inverse_cauchy(x: float, x0: float = 0.0, gamma: float = 0.5) -> float:
    """
    Returns values from inverted cumulative Cauchy distribution
    see: https://en.wikipedia.org/wiki/Cauchy_distribution

    :param x: value at which distribution will be evaluated
    :param x0: location parameter
    :param gamma: scale parameter
    """
    return x0 + (gamma * np.tan(np.pi * (x - 0.5)))


@jit
def inverse_laplace(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """
     Returns values from inverted cumulative Laplace distribution
    see: https://en.wikipedia.org/wiki/Laplace_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param b: scale parameter, in range (0, inf)
    """
    ...


@jit
def inverse_logistic(x: float, mu: float = 1.0, s: float = 1.0) -> float:
    """
    Returns values from inverted cumulative logistic distribution
    see: https://en.wikipedia.org/wiki/Logistic_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param s: scale parameter
    """
    return s * np.log((x - 1) / x) + mu
