import numpy as np
from sklearn.exceptions import NotFittedError


class LeastSquaresEstimator:
    """
    Lest Squares Estimator implementation

    It can be used to estimate characteristics of linear static MISO systems.
    """
    def __init__(self):
        self.coefficients = None

    def fit(self, x: np.array, y: np.array) -> None:
        """
        Fit systems coefficients to data

        :param x: inputs to system, array with shape (num_samples, system_n_dim)
        :param y: outputs of system, array with shape (num_samples,)
        """
        self.coefficients = np.linalg.inv(x.T @ x) @ x.T @ y

    @staticmethod
    def covariance(x):
        """
        :param x: inputs used to estimate MISO system

        :return: return estimators estimated covariance matrix
        """
        return np.power(np.std(x), 2) * np.linalg.inv(x.T @ x)

    def _estimate(self, x):
        return x.T @ self.coefficients

    def predict(self, values):
        """
        Predict system output using fitted coefficients

        :param values: vector of points to predict

        :raises: NotFittedError when called prior to calling fit

        :return: system outputs for fitted coefficients
        """
        if self.coefficients is None:
            raise NotFittedError

        return np.vectorize(self._estimate, signature="(n)->()")(values)
