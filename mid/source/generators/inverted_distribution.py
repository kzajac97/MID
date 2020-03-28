from typing import Callable, Tuple

import numpy as np
from numba import jit
from scipy import integrate

INTEGRAL_VALUE_INDEX = 0
INTEGRAL_ERROR_INDEX = 1


@jit
def cdf(pdf: Callable[[Tuple[float, ...]], float], x: float, infinity_approximation: float = np.inf, *args) -> np.array:
    """
    :param pdf: probability distribution function
    :param x: point at which to evaluate
    :param infinity_approximation:
    :param args: additional args to pdf function

    :return: cumulative distribution function value at x
    """
    return integrate.quad(pdf, -1 * infinity_approximation, x, args)[INTEGRAL_VALUE_INDEX]
