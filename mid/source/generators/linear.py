import numpy as np
from scipy import signal


def _sine_distribution(n_points: int, omega: float = 10 * np.pi, first_sample: float = None) -> np.array:
    """
    Generate random numbers from sine distribution

    :param n_points: number of generated data points
    :param omega: sine frequency
    :param first_sample: starting point for drawing distribution

    :return: numpy array with random distribution samples
    """
    distribution = np.zeros(n_points)
    distribution[0] = first_sample or np.random.rand()

    for index in range(1, n_points):
        distribution[index] = 0.5 * (np.sin(omega * distribution[index - 1]) + 1)

    return distribution


def _sawtooth_distribution(n_points: int, period: float = 10 * np.pi, first_sample: float = None) -> np.array:
    """
    Generate random numbers from sawtooth distribution

    :param n_points: number of generated data points
    :param period: sawtooth period
    :param first_sample: starting point for drawing distribution

    :return: numpy array with random distribution samples
    """
    distribution = np.zeros(n_points)
    distribution[0] = first_sample or np.random.rand()

    for index in range(1, n_points):
        distribution[index] = signal.sawtooth(period * distribution[index - 1])

    return distribution


def _fibonacci_distribution(n_points: int, p: int = 8, q: int = 4, m: int = 100) -> np.array:
    """
    Generate random number with fibonacci distribution

    :param n_points: number of generated data points
    :param p: fibonacci param, defaults to 8, must be greater than q
    :param q: fibonacci param, defaults to 4, must be less than p
    :param m: modulo param, defaults to 100, should be in range <100, 10_000>

    :return: numpy array with random distribution samples
    """
    distribution = np.zeros(n_points)
    distribution[:p] = np.random.rand(p)

    for index in range(p, n_points):
        distribution[index] = (distribution[index - p] + distribution[index - q]) % m

    return distribution


distribution_name_mapping = {
    'uniform': np.random.rand,
    'sine': _sine_distribution,
    'sawtooth': _sawtooth_distribution,
    'fibonacci': _fibonacci_distribution
}


def random_distribution(n_points: int, distribution: str, **kwargs) -> np.array:
    """
    Generate array of random number with chosen distribution

    :param n_points: number of generated data points
    :param distribution: distribution type, accepted values are:
                         `uniform`
                         `sine`
                         `sawtooth`
                         `fibonacci`

    :return: numpy array with random distribution samples
    """
    return distribution_name_mapping[distribution](n_points, **kwargs)
