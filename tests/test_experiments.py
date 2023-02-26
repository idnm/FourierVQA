import numpy as np

from experiments_utils import CoarsePauli, two_local_circuit, random_local_pauli, random_local_pauli_circuit


def test_binomial_stability():
    assert np.allclose(CoarsePauli.z_product_distribution(100, 50, 35).sum(), 1)
