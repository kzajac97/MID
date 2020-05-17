import numpy as np


"""
Basis generator, should be used via OrthogonalBasis interface only
All follow the same signature:
    :param n_functions: number of basis function to generate

    :return: list of lambda or np.poly1d functions corresponding to basis functions in chosen base
"""


def cosine_basis(n_functions: int) -> list:
    callable_basis = [lambda x: np.sqrt(1 / (2 * np.pi))]
    # passing i=i is Python trick, this is poor design!
    callable_basis.extend([lambda x, i=i: np.sqrt(1 / np.pi) * np.cos(i*x) for i in range(1, n_functions)])

    return callable_basis


# TODO: Update basis generators
def hermite_basis(n_functions: int, domain: tuple = (-1, 1)) -> list:
    return [np.polynomial.hermite.Hermite.basis(deg=n) for n in range(n_functions)]


BASIS_GENERATORS = {
    "cos": cosine_basis,
    "cosine": cosine_basis,
    "hermite": hermite_basis,
}
