import numpy as np


def estimate_expectation(rvs: np.array) -> float:
    """
    :param rvs: array with realization of random distribution
    """
    return np.sum(rvs) / rvs.shape[0]


def estimate_variance(rvs: np.array, biased: bool = False) -> float:
    """
    :param rvs: array with realization of random distribution
    :param biased: if True used biased variance estimator
    """
    n = rvs.shape[0]
    if biased:
        return np.sum(np.power(rvs - estimate_expectation(rvs), 2)) / n

    return (1 / (n - 1)) * np.sum(np.power(rvs - estimate_expectation(rvs), 2))


def estimate_covariance(x_rvs: np.array, y_rvs: np.array) -> float:
    """
    :param x_rvs: array with realization of random distribution
    :param y_rvs: array with realization of random distribution

    :return: covariance between given random variables
    """
    return estimate_expectation(x_rvs * y_rvs) - (estimate_expectation(x_rvs) * estimate_expectation(y_rvs))


def estimate_correlation(x_rvs: np.array, y_rvs: np.array) -> float:
    """
    :param x_rvs: array with realization of random distribution
    :param y_rvs: array with realization of random distribution

    :return: Pearsons correlation coefficient between given random variables
    """
    return estimate_covariance(x_rvs, y_rvs) / (np.std(x_rvs) * np.std(y_rvs))
