import numpy as np
from qiskit.quantum_info import Pauli

from experiments_utils import CoarsePauli, two_local_circuit, random_local_pauli, random_local_pauli_circuit
from fourier_vqa import FourierExpansionVQA, FourierStats, PauliCircuit


def test_binomial_stability():
    assert np.allclose(CoarsePauli.z_product_distribution(100, 50, 35).sum(), 1)


def test_monte_carlo_plain():

    num_qubits = 5
    num_paulis = 20

    fourier_computation = FourierExpansionVQA.random(num_qubits, num_paulis, seed=5757)
    fourier_computation.compute(check_admissible=False, verbose=False)

    all_full = len(fourier_computation.complete_nodes)
    nonzero_full = len([node for node in fourier_computation.complete_nodes if node.expectation_value != 0])
    num_all, num_nonzero = fourier_computation.estimate_node_count_monte_carlo(num_samples=30000, check_admissible=False)
    print('\n')
    print(f'all {num_all:.2e}, all full {all_full:.2e}, all random {1.5**num_paulis:.2e}, nonzero {num_nonzero:.2e}, nonzero_full {nonzero_full:.2e}')


def test_monte_carlo_with_filtering():
    num_qubits = 20
    num_paulis = 55

    fourier_computation = FourierExpansionVQA.random(num_qubits, num_paulis, seed=16)
    fourier_computation.compute(check_admissible=True, verbose=False)

    all_full = len(fourier_computation.complete_nodes)
    nonzero_full = len([node for node in fourier_computation.complete_nodes if node.expectation_value != 0])
    num_all, num_nonzero = fourier_computation.estimate_node_count_monte_carlo(num_samples=1000, check_admissible=True, seed=11)

    print('\n')
    print(f'all {num_all:.2e}, all full {all_full:.2e}, all random {1.5**(num_paulis-num_qubits):.2e}, nonzero {num_nonzero:.2e}, nonzero_full {nonzero_full:.2e}')
