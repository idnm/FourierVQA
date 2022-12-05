import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector, random_clifford

from jax_utils import jax_tensor
from test_wave_expansion import random_clifford_phi
from wave_expansion import CliffordPhi


def equal_up_to_global_phase(s0, s1):
    index_of_max = np.unravel_index(np.abs(s0).argmax(), s0.shape)

    return np.allclose(s0/s0[index_of_max], s1/s1[index_of_max])


def test_jax_unitary(max_num_qubits=3, max_num_parameters=5):

    # Unitary test
    for num_qubits in range(2, max_num_qubits+1):
        for num_parameters in range(max_num_parameters+1):
            qc = random_clifford_phi(num_qubits, num_parameters)

            np.random.seed(0)
            parameters = np.random.rand(qc.num_parameters)
            qiskit_unitary = Operator(qc.bind_parameters(parameters)).data
            jax_unitary = jax_tensor(qc, 'id')(parameters)

            assert equal_up_to_global_phase(qiskit_unitary, jax_unitary)

    # State test
    for num_qubits in range(2, max_num_qubits+1):
        for num_parameters in range(max_num_parameters+1):
            qc = random_clifford_phi(num_qubits, num_parameters)

            np.random.seed(0)
            parameters = np.random.rand(qc.num_parameters)
            qiskit_state = Statevector.from_label('0'*num_qubits).evolve(qc.bind_parameters(parameters)).data
            jax_state = jax_tensor(qc, '0')(parameters)

            assert equal_up_to_global_phase(qiskit_state, jax_state)


