import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator

from jax_utils import jax_tensor
from test_wave_expansion import random_clifford_phi
from wave_expansion import CliffordPhi


def equal_up_to_global_phase(s0, s1):
    first_nonzero_index = tuple([x[0] for x in s0.nonzero()])
    norm = s1[first_nonzero_index]/s0[first_nonzero_index]
    return np.allclose(norm*s0, s1)


def test_jax_unitary():

    qc = random_clifford_phi(4, 8)

    np.random.seed(0)
    parameters = np.random.rand(qc.num_parameters)
    qiskit_unitary = Operator(qc.bind_parameters(parameters)).data
    jax_unitary = jax_tensor(qc, 'id')(parameters)

    assert equal_up_to_global_phase(qiskit_unitary, jax_unitary)
