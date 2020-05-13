import numpy as np


def cosine_basis(n_functions: int) -> list:
    """
    Basis generator, should be used via OrthogonalBasis interface only

    :param n_functions: number of basis function to generate

    :return: list of lambda functions corresponding to basis functions in cosine base
    """
    callable_basis = [lambda x: np.sqrt(1 / (2 * np.pi))]
    # passing i=i is Python trick, this is poor design!
    callable_basis.extend([lambda x, i=i: np.sqrt(1 / np.pi) * np.cos(i*x) for i in range(1, n_functions)])

    return callable_basis
