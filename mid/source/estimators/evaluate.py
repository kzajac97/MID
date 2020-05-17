from typing import Any, Callable, Union

import numpy as np
from sklearn import metrics
from tqdm import tqdm


_METRICS_MAPPING = {
    "mse": metrics.mean_squared_error,
    "mae": metrics.mean_absolute_error,
    "log_mse": metrics.mean_squared_log_error,
    "r2": metrics.r2_score,
}


def evaluate_estimate(
        estimator: Callable[[np.array, Any], Any],
        distribution: Any,
        true_value: Any,
        n_samples: int,
        n_trials: int,
        silent: bool = True,
        metric: Union[str, Callable[[np.array, np.array], float]] = "mse",
        *args,
        **kwargs,
) -> float:
    """
    Evaluates random variable statistic estimator

    :param estimator: callable returning estimates
    :param distribution: distribution object returning random variables
    :param true_value: true value of estimated statistic
    :param n_samples: number of random values to generate
    :param n_trials: number of estimation to run
    :param silent: if False print progress bar
    :param metric: metrics used to evaluate estimator
                   can be str or callable
    :param args: positional arguments to estimator function
    :param kwargs: key word arguments to distribution random number generator

    :return: estimator error using given metric
    """
    estimates = []
    for _ in tqdm(range(n_trials), disable=silent):
        rvs = distribution.rvs(size=n_samples, **kwargs)
        estimates.append(estimator(rvs, *args))

    if metric in _METRICS_MAPPING.keys():
        return _METRICS_MAPPING[metric](estimates,  np.repeat(true_value, n_trials))

    return metric(estimates, n_trials * [true_value])
