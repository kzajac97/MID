import numpy as np


"""

Kernel Function for density estimation
For detailed formulas see: https://scikit-learn.org/stable/modules/density.html

All follow the same signature:
    :param x: point at which kernel function is evaluated 
    :param h: width of kernel
    
    :return: values of kernel at a point
    
    Should be applied as vectorized functions for better interface
    
    example:
        >>> np.vectorize(kernel_fn, signature="()->(n)")(linspace)
"""


def gaussian_kernel(x, h):
    return np.exp(-1*(np.power(x, 2) / (2 * np.power(h, 2))))


def epenechikov_kernel(x, h):
    value = 1 - (np.power(x, 2) / np.power(h, 2))
    return (value > 0) * value


def cosine_kernel(x, h):
    return (np.abs(x) - h < 0) * np.cos((np.pi * x) / (2 * h))


def tophat_kernel(x, h):
    return 1 * (np.abs(x) - h < 0)


def linear_kernel(x, h):
    return (x > 0) * (1 - x / h) * (x - h < 0) + (x < 0) * (1 + x / h) * (x + h > 0)


def exponential_kernel(x, h):
    return np.exp(-x/h) * (x > 0) + np.exp(x/h) * (x < 0)


DENSITY_KERNELS = {
    "gau": gaussian_kernel,
    "gaussian": gaussian_kernel,
    "epenechikov": epenechikov_kernel,
    "ep": epenechikov_kernel,
    "cos": cosine_kernel,
    "cosine": cosine_kernel,
    "top": tophat_kernel,
    "tophat": tophat_kernel,
    "lin": linear_kernel,
    "linear": linear_kernel,
    "exp": exponential_kernel,
    "exponential": exponential_kernel,
}
