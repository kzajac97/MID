import numpy as np
from scipy import stats


def white_noise(n_samples: int, mean: float = 0.0, variance: float = 1.0, amplitude: float = 1.0) -> np.array:
    """
    Generate white noise signal

    :param n_samples: number of samples to generate
    :param mean: mean of norma distribution of noise signal
    :param variance: variance of normal distribution signal
    :param amplitude: amplitude of normal generated signal

    :return: array with random samples from noise signal
    """
    return amplitude * stats.norm.rvs(size=n_samples, loc=mean, scale=variance)


def multivariate_white_noise(n_samples: int, mean: float = 0.0, covariance: np.array = None, amplitude: float = 1.0) -> np.array:
    """
    Generate white noise signal

    :param n_samples: number of samples to generate
    :param mean: mean of norma distribution of noise signal
    :param covariance: covariance matrix of multivariate normal distribution signal
    :param amplitude: amplitude of normal generated signal

    :return: array with random samples from noise signal, with shape (n_samples, covariance_size)
    """
    covariance = covariance or np.eye(10)
    return amplitude * stats.multivariate_normal(size=n_samples, mean=mean, cov=covariance)


def disturbance(
        n_samples: int, mean: np.array = None, covariance: np.array = None, amplitude: float = 1.0, s: float = 0.5
) -> np.array:
    """
    Generate disturbance signal

    :param n_samples: number of samples to generate
    :param mean: mean of normal distribution of noise signal
    :param covariance: covariance matrix of multivariate normal distribution signal
    :param amplitude: amplitude of normal generated signal
    :param s: past samples influence factor

    :return: array with random samples from noise signal, with shape (n_samples, covariance_size)
    """
    covariance = covariance or np.eye(10)
    random_vars = stats.multivariate_normal.rvs(size=(1 + n_samples), mean=mean, cov=covariance)
    return amplitude * (random_vars[:-1] + s * random_vars[1:])
