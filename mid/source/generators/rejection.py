from typing import Any, Callable, Optional

import numpy as np


def rejection_sampling_distribution(
        n_samples: int,
        acceptance: Callable[[float, float], bool],
        x_sampler: Callable[[Optional[Any]], float],
        y_sampler:  Callable[[Optional[Any]], float],
) -> np.array:
    """

    :param n_samples:
    :param acceptance:
    :param x_sampler:
    :param y_sampler:

    :return:
    """
    generated_samples = []
    generated_samples_count = 0

    while generated_samples_count < n_samples:
        x_sample = x_sampler()
        y_sample = y_sampler()

        if acceptance(x_sample, y_sample):
            generated_samples.append(x_sample)
            generated_samples_count += 1

    return np.array(generated_samples)
