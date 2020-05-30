import numpy as np
from scipy import stats

_NOISE_TYPES = {
    "normal": stats.norm.rvs,
    "uniform": stats.uniform.rvs,
}


class MISOSystem:
    """
    Class holding implementation of array aware MISO system
    """
    def __init__(self, coefficients: np.array, noise: str = "normal"):
        """
        :param coefficients: Vector of systems coefficients
        :param noise: type of noise in system
        """
        self.coefficients = coefficients
        self._noise = _NOISE_TYPES[noise]

    @property
    def n_inputs(self):
        """
        :return: number of inputs system has
        """
        return self.coefficients.shape[0]

    def __call__(self, input_values: np.array, mean: float = 0.0, variance: float = 1.0) -> np.array:
        """
        :param input_values: array of points to evaluate system
        :param mean: noise mean
        :param variance: noise variance

        :return: system output for given input values
        """
        return input_values.T @ self.coefficients + self._noise(size=1, loc=mean, scale=variance)

    def apply(self, input_sequence: np.array,  mean: float = 0.0, variance: float = 1.0) -> np.array:
        """
        Apply system to a sequence of input values in one call

        :param input_sequence: sequence of input values
        :param mean: noise mean
        :param variance: noise variance

        :return: array of systems outputs for given inputs
        """
        return np.apply_along_axis(self, 1, input_sequence, mean, variance)


class ARXSystem:
    """
    Class holding statistical ARX system
    """
    def __init__(self, alphas: np.array, betas: np.array, noise: str = "normal"):
        """
        :param alphas: autoregressive vector parameters
        :param betas: input parameter
        :param noise: type of noise in ARX system
        """
        self.alphas = alphas
        self.betas = betas
        self._noise = _NOISE_TYPES[noise]

    def _previous_time_steps(self, array: np.array, index: int, n_steps: int) -> np.array:
        """
        Utility for slicing time series array and getting previous samples

        :param array: times series array to slice
        :param index: current time step index
        :param n_steps: number of steps to look back at

        :return: time series array slice
        """
        return array[index - n_steps: index]

    def __call__(self, u: np.array, mean: float = 0.0, variance: float = 1.0) -> np.array:
        """
        Get ARX system output sequence

        :param u: inputs for X part of system

        :warning: first N elements are padded with zeros

        :return: array containing a sequence of system outputs
        """
        v = np.zeros(len(u))
        y = np.array([sum(
                self.alphas * self._previous_time_steps(v, index=index, n_steps=len(self.alphas)) +
                self.betas * self._previous_time_steps(u, index=index, n_steps=len(self.betas))
            ) for index in range(len(self.alphas), len(u))])

        return np.concatenate([np.zeros(len(self.alphas)), y]) + self._noise(size=len(u), scale=variance, loc=mean)
