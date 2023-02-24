import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_pauli, Pauli

from experiments_utils import RandomLocalCircuit, CoarsePauli, NodeDistribution
from wave_expansion import PauliCircuit


def test_local_from_pauli():
    qc = QuantumCircuit(3)
    qc.rzz(Parameter('zz'), 0, 1)
    qc.rx(Parameter('x'), 1)
    qc.ry(Parameter('y'), 2)

    pauli_circuit = PauliCircuit.from_parameterized_circuit(qc)
    random_local_circuit = RandomLocalCircuit.from_pauli_circuit(pauli_circuit)

    coarse_paulis = random_local_circuit.paulis
    assert coarse_paulis[0].z == 2 and coarse_paulis[0].x == 0
    assert coarse_paulis[1].z == 0 and coarse_paulis[1].x == 1
    assert coarse_paulis[2].z == 1 and coarse_paulis[2].x == 1


def test_intersection_distribution():

    num_qubits = 3
    n1 = 2
    n2 = 3

    assert np.allclose(CoarsePauli.z_product_distribution(num_qubits, n1, n2), [0, 1, 0, 0])

    num_qubits = 3
    n1 = 2
    n2 = 1

    assert np.allclose(CoarsePauli.z_product_distribution(num_qubits, n1, n2), [0, 2/3, 0, 1/3])

    num_qubits = 4
    n1 = 3
    n2 = 2

    assert np.allclose(CoarsePauli.z_product_distribution(num_qubits, n1, n2), [0, 1/2, 0, 1/2, 0])

    num_qubits = 5
    n1 = 3
    n2 = 2

    assert np.allclose(CoarsePauli.z_product_distribution(num_qubits, n1, n2), [0, 3/10, 0, 3/5, 0, 1/10])


def test_node_distribution_update():
    observable = CoarsePauli.from_pauli(Pauli('IZY'))
    pauli = CoarsePauli.from_pauli(Pauli('III'))

    node_distribution = NodeDistribution.from_observable_and_pauli(observable, pauli)
    nonzero = np.where(node_distribution.counts_array != 0)

    assert nonzero[0] == 2 and nonzero[1] == 1

    observable = CoarsePauli.from_pauli(Pauli('IZY'))
    pauli = CoarsePauli.from_pauli(Pauli('IXI'))

    node_distribution = NodeDistribution.from_observable_and_pauli(observable, pauli)
    print(node_distribution)


def test_trivial():
    observable = CoarsePauli(1, 1, 0)
    pauli = CoarsePauli(1, 1, 1)
    print(observable.product_distribution(pauli))


def test_binomial_stability():
    assert np.allclose(CoarsePauli.z_product_distribution(100, 50, 35).sum(), 1)


def test_node_estimation():
    num_qubits = 20
    num_paulis = 80

    pauli_circuit = PauliCircuit.random(num_qubits, num_paulis, seed=4)
    observable = CoarsePauli.from_pauli(random_pauli(num_qubits, seed=94))

    random_local_circuit = RandomLocalCircuit.from_pauli_circuit(pauli_circuit)
    node_distribution = random_local_circuit.estimate_node_count(observable)
    print('\n')
    random_est = (3/2)**num_paulis
    local_random_est = node_distribution.counts_array.sum()
    print(f'random {random_est:.2e}, local {local_random_est:.2e}, ratio {random_est/local_random_est}')


    # for n_paulis in range(20, num_paulis):
    #     random_local_circuit = RandomLocalCircuit(num_qubits, coarse_paulis[:n_paulis])
    #     node_distribution = random_local_circuit.estimate_node_count(observable)
    #     print(f'n paulis {n_paulis}')
    #     print(f'terms {node_distribution.counts_array.sum()}')


def test_single_update():
    num_qubits = 40

    observable = CoarsePauli.from_pauli(random_pauli(num_qubits, seed=10))
    pauli1 = CoarsePauli.from_pauli(random_pauli(num_qubits, seed=30))
    pauli2 = CoarsePauli.from_pauli(random_pauli(num_qubits, seed=40))

    # observable = CoarsePauli(num_qubits, 0, 1)
    # pauli1 = CoarsePauli(num_qubits, 1, 2)
    # pauli2 = CoarsePauli(num_qubits, 1, 1)

    node_distribution = NodeDistribution.from_observable_and_pauli(observable, pauli1)
    print(node_distribution.counts_array.sum())
    print(node_distribution)

    # probs = []
    # for z in range(num_qubits+1):
    #     for x in range(num_qubits + 1):
    #
    #         count = node_distribution.counts_array[z, x]
    #         prob = CoarsePauli(num_qubits, z, x).commuting_probability(pauli2)
    #         # print(f'z{z} x{x} count{count} prob {prob}')
    #         probs.append(prob*count)

    # print((sum(probs) / node_distribution.counts_array.sum()))
    # print(((0.5 / sum(probs) * node_distribution.counts_array.sum())))
    # print(((0.5/sum(probs)*node_distribution.counts_array.sum()))**100)
