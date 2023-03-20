import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate, RYGate, RZGate, RZZGate, RZXGate
from qiskit.quantum_info import random_clifford, Operator, Pauli, random_pauli

from fourier_vqa import PauliCircuit, FourierExpansionVQA, PauliSpace


def random_clifford_phi(num_qubits, num_parametric_gates, num_duplicate_parameters=0, seed=0):
    """Generate a random CliffordPhi circuit."""
    random.seed(seed)
    qc = QuantumCircuit(num_qubits)
    parameters = [Parameter(f'θ_{i}') for i in range(num_parametric_gates-num_duplicate_parameters)]
    for i in range(num_parametric_gates):
        clifford_gate = random_clifford(num_qubits, seed=seed+i)
        qc.append(clifford_gate.to_instruction(), range(num_qubits))

        if num_qubits == 1:
            parametric_gate = random.choice([RXGate, RZGate, RYGate])
        else:
            parametric_gate = random.choice([RXGate, RZGate, RYGate, RZZGate, RZXGate])
        parameter = parameters[i % (num_parametric_gates - num_duplicate_parameters)]
        position = random.sample(range(num_qubits), parametric_gate(0).num_qubits)

        qc.append(parametric_gate(parameter), position)

    return qc


def hilbert_schmidt_product(u, v):
    num_qubits = int(np.log2(u.shape[0]))
    return np.abs((u.conj() * v).sum())**2/4**num_qubits


def parametric_circuits_are_equivalent(qc0, qc1):

    np.random.seed(42)
    random_parameters = np.random.rand(10, qc0.num_parameters)
    return all([Operator(qc0.bind_parameters(p)).equiv(Operator(qc1.bind_parameters(p))) for p in random_parameters])


def test_pauli_circuit_reconstruction(num_qubits=3, num_parametric_gates=4):
    for num_qubits in range(1, 4):
        for num_parametric_gates in range(1, 15, 3):
            qc = random_clifford_phi(num_qubits, num_parametric_gates)
            pauli_qc = PauliCircuit.from_parameterized_circuit(qc)
            qc_reconstruction = pauli_qc.to_parameterized_circuit()

            assert parametric_circuits_are_equivalent(qc, qc_reconstruction)


def test_fourier_computation_plain():

    for num_qubits in range(1, 5):
        for num_parameters in range(1, 6):
            for seed in range(5):
                qc = random_clifford_phi(num_qubits, num_parameters, seed=seed)
                pauli_circuit = PauliCircuit.from_parameterized_circuit(qc)
                observable = random_pauli(num_qubits, seed=seed)

                try:
                    fourier_computation = FourierExpansionVQA(pauli_circuit, observable)
                    fourier_computation.compute(check_admissible=False, verbose=False)
                except ValueError as e:
                    # Occurs when all Pauli operators have no X components.
                    continue

                params = np.random.rand(num_parameters)
                assert np.allclose(
                    pauli_circuit.expectation_value(observable, params),
                    fourier_computation.evaluate_loss_at(params)
                )


def test_fourier_computation_with_filtering():
    for num_qubits in range(1, 5):
        for num_parameters in range(1, 6):
            for seed in range(5):
                qc = random_clifford_phi(num_qubits, num_parameters, seed=seed)
                pauli_circuit = PauliCircuit.from_parameterized_circuit(qc)
                observable = random_pauli(num_qubits, seed=seed)

                try:
                    fourier_computation = FourierExpansionVQA(pauli_circuit, observable)
                    fourier_computation.compute(check_admissible=True, verbose=False)
                except ValueError as e:
                    continue

                params = np.random.rand(num_parameters)
                assert np.allclose(
                    pauli_circuit.expectation_value(observable, params),
                    fourier_computation.evaluate_loss_at(params)
                )


def test_pauli_space():

    num_qubits = 5

    for seed in range(10):

        # Create three independent paulis.
        np.random.seed(0)
        seeds = np.random.randint(0, 1000, size=3)
        label0, label1, label2 = [list(random_pauli(num_qubits, seed=s).to_label()) for s in seeds]
        label0[0] = 'X'
        label1[0] = 'I'
        label2[0] = 'I'

        label0[1] = 'I'
        label1[1] = 'Y'
        label2[1] = 'I'

        label0[2] = 'I'
        label1[2] = 'Z'
        label2[2] = 'X'

        paulis = [Pauli(''.join(label)) for label in [label0, label1, label2]]

        # Generate more paulis with clear dependence.
        paulis = [paulis[0], paulis[0], paulis[0].compose(paulis[1]), paulis[1], paulis[2], paulis[1].compose(paulis[2])]

        pauli_space = PauliSpace(paulis)
        pauli_space.construct()

        assert pauli_space.independent_paulis == [0, 2, 4]
        assert pauli_space.dependent_paulis == [1, 3, 5]

        assert np.all(pauli_space.decomposition_matrix[0] == [True, False, False, False, False])
        assert np.all(pauli_space.decomposition_matrix[1] == [True, False, False, False, False])
        assert np.all(pauli_space.decomposition_matrix[3] == [True, True, False, False, False])
        assert np.all(pauli_space.decomposition_matrix[5] == [True, True, True, False, False])

        assert pauli_space.list_decomposition(pauli_space.decomposition_matrix[1]) == [0]
        assert pauli_space.list_decomposition(pauli_space.decomposition_matrix[3]) == [0, 2]
        assert pauli_space.list_decomposition(pauli_space.decomposition_matrix[5]) == [0, 2, 4]

        decomposition = pauli_space.compute_decomposition(Pauli('IIIII'), 5)
        assert np.all(decomposition == [False, False, False, False, False])
        assert pauli_space.list_decomposition(decomposition) == []

        observable = paulis[3]
        decomposition = pauli_space.compute_decomposition(observable, 2)
        assert pauli_space.list_decomposition(decomposition) is None
        decomposition = pauli_space.compute_decomposition(observable, 5)
        assert pauli_space.list_decomposition(decomposition) == [0, 2]

        assert pauli_space.list_decomposition(pauli_space.update_decomposition(decomposition, 0)) == [2]
        assert pauli_space.list_decomposition(pauli_space.update_decomposition(decomposition, 2)) == [0]
        assert pauli_space.list_decomposition(pauli_space.update_decomposition(decomposition, 4)) == [0, 2, 4]


def test_estimate_node_count_limited_volume():
    num_qubits = 15
    num_paulis = 60

    fourier_computation = FourierExpansionVQA.random(num_qubits, num_paulis, seed=0)

    volume_stats = fourier_computation.estimate_node_count_limited_volume(max_nodes=1000)
    mc_tot, _ = fourier_computation.estimate_node_count_monte_carlo(num_samples=1000)

    print('\n')

    volume_tot = volume_stats.sum()
    exact = 1.5**num_paulis

    print(f'vol {volume_tot:.2e}|{volume_tot/exact:.2%} mc {mc_tot:.2e}|{mc_tot/exact:.2%}')


def test_limits():
    num_qubits = 30
    num_paulis = 25
    fourier_expansion = FourierExpansionVQA.random(num_qubits, num_paulis)
    fourier_expansion.compute()