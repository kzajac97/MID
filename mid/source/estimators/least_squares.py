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


class RecursiveLeastSquaresEstimator:
    def __init__(self, time_steps: int, forgetting: float = 1.0, initial_condition_scale: float = 500.0):
        self.time_steps = time_steps
        self.forgetting = forgetting
        self.initial_condition_scale = initial_condition_scale
        self.theta = None

    def fit_predict(self, x, y):
        self.theta = np.zeros(x.shape)
        covariance = self.initial_covariance_matrix

        for index in range(1, x.shape[0]):
            covariance = self.recursive_covariance(covariance, self.regression_vector(x, y, index - 1))
            self.theta[index] = (self.theta[index - 1] +
                                 covariance @
                                 self.regression_vector(x, y, index - 1) *
                                 self.one_step_error(y[index], self.theta[index - 1], self.regression_vector(x, y, index - 1)))

    def one_step_error(self, y, theta, phi):
        """

        :param y:
        :param theta:
        :param phi:
        :return:
        """
        return y - theta.T @ phi

    @property
    def initial_covariance_matrix(self) -> np.array:
        """
        :return: initial covariance identity matrix
        """
        return self.initial_condition_scale * np.eye(self.time_steps * 2)

    def regression_vector(self, x, y, index):
        if index > self.time_steps:
            return np.concatenate([x[index - self.time_steps: index], -y[index - self.time_steps: index]])

        return np.zeros(self.time_steps * 2)

    def recursive_covariance(self, previous_covariance, regression_vector):
        """

        :param previous_covariance:
        :param regression_vector:
        :return:
        """
        phi = regression_vector.reshape(-1, 1)
        return ((1 / self.forgetting) * (previous_covariance -
                                         ((previous_covariance @ phi @ phi.T @ previous_covariance) /
                                          (self.forgetting + phi.T @ previous_covariance @ phi))))

