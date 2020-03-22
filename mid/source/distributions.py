from typing import Callable

import numpy as np
from numba import jit
from scipy import integrate

INTEGRAL_VALUE_INDEX = 0
INTEGRAL_ERROR_INDEX = 1


@jit
def exponential(x: float, rate: float = 1.0) -> float:
    """
    Returns values from exponential distribution, described by formula:
    .. math:: lambda exp(-lambda x)

    :param x: value at which distribution will be evaluated
    :param rate: lambda parameter of distribution
    """
    return rate * np.exp(-1*rate*x) if x > 0 else 0


@jit
def logistic(x: float, mu: float, s: float) -> float:
    """
    Returns values from logistic distribution
    see: https://en.wikipedia.org/wiki/Logistic_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param s: scale parameter
    """
    exponent_value = -(x - mu)/s
    return np.exp(exponent_value) / (s * (1 + np.exp(exponent_value)**2))


@jit
def cauchy(x: float, x0: float, gamma: float) -> float:
    """
    Returns values from Cauchy distribution
    see: https://en.wikipedia.org/wiki/Cauchy_distribution

    :param x: value at which distribution will be evaluated
    :param x0: location parameter
    :param gamma: scale parameter
    """
    ratio = ((x - x0) / gamma)**2
    return 1 / (np.pi * gamma * (1 + ratio))


@jit
def laplace(x: float, mu: float, b: float) -> float:
    """
    Returns values from Laplace distribution
    see: https://en.wikipedia.org/wiki/Laplace_distribution

    :param x: value at which distribution will be evaluated
    :param mu: location parameter
    :param b: scale parameter, in range (0, inf)
    """
    exponent = -1 * np.abs(x - mu) / b
    return 1/(2*b) * np.exp(exponent)


@jit
def cdf(pdf: Callable[[float, ...], float], x: float) -> np.array:
    """
    :param pdf: probability distribution function
    :param x: point at which to evaluate

    :return: cumulative distribution function value at x
    """
    return integrate.quad(pdf, -1*np.inf, x)[INTEGRAL_VALUE_INDEX]
