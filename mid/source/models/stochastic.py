import numpy as np
from scipy import stats

_NOISE_TYPES = {
    "normal": stats.norm.rvs,
    "uniform": stats.uniform.rvs,
}


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
        self.noise = _NOISE_TYPES[noise]

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

        return np.concatenate([np.zeros(len(self.alphas)), y]) + self.noise(size=len(u), scale=variance, loc=mean)
