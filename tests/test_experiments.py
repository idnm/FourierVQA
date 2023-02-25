import numpy as np

from experiments_utils import CoarsePauli


def test_binomial_stability():
    assert np.allclose(CoarsePauli.z_product_distribution(100, 50, 35).sum(), 1)

