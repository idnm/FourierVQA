import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_clifford, Operator, Pauli, Statevector

from wave_expansion import CliffordPhi, PauliString, PauliRotation


def test_trivial():
    assert PauliString.trivial(2) == PauliString([0, 0], [0, 0])


def random_clifford_phi(num_qubits, num_parameters, seed=0):
    """Generate a random CliffordPhi circuit."""
    random.seed(seed)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_parameters):
        clifford_gate = random_clifford(num_qubits, seed=seed+i)
        qc.append(clifford_gate.to_instruction(), range(num_qubits))

        parametric_gate = random.choice(list(PauliRotation.pauli_gates.values()))[0]
        parameter = Parameter(f'θ_{i}')
        position = random.sample(range(num_qubits), parametric_gate(0).num_qubits)

        qc.append(parametric_gate(parameter), position)

    return CliffordPhi.from_quantum_circuit(qc)


def reconstruct_circuit(qc0):
    """Reconstruct the full circuit from clifford and parametric gates"""
    num_qubits = qc0.num_qubits
    clifford_gates, parametric_gates = qc0.gates()

    qc = QuantumCircuit(num_qubits)
    for i, clifford_gate in enumerate(clifford_gates):
        while parametric_gates and parametric_gates[0].after_clifford_num == i-1:
            parametric_gate = parametric_gates.pop(0)
            qc.append(parametric_gate.gate, parametric_gate.qubit_indices)
        qc.append(clifford_gate, range(num_qubits))

    # Remaining parametric gates.
    for parametric_gate in parametric_gates:
        qc.append(parametric_gate.gate, parametric_gate.qubit_indices)

    return qc


def test_circuit_reconstruction(num_qubits=3, num_parameters=4, seed=0):
    """Test reconstructing the full circuit from clifford and parametric gates"""
    qc0 = CliffordPhi.from_quantum_circuit(random_clifford_phi(num_qubits, num_parameters, seed=seed))
    qc = reconstruct_circuit(qc0)

    random_parameters = np.random.rand(10, qc.num_parameters)

    checks = [Operator(qc0.bind_parameters(p)).equiv(Operator(qc.bind_parameters(p))) for p in random_parameters]
    assert False not in checks


def qc_for_generator_testing():
    qc = CliffordPhi(3)
    p = [Parameter(f'p_{i}') for i in range(4)]
    qc.rx(p[0], 0)
    qc.h(1)
    qc.cx(0, 2)
    qc.rzz(p[1], 0, 1)
    qc.rzz(p[2], 0, 1)
    qc.x(0)
    qc.y(1)
    qc.z(2)
    qc.ry(p[3], 2)
    return qc


def test_generators_0():
    """A simple hand-crafted test for commutation of the generators to the vacuum state."""

    qc = qc_for_generator_testing()
    num_duplicates, generators = qc.generators_0()
    assert num_duplicates == 1
    assert set(generators) == {Pauli('IIX'), Pauli('IXI'), Pauli('XII')}


def test_generators_H():
    """A simple hand-crafted test for commutation of the generators to Hamiltonian."""

    qc = qc_for_generator_testing()
    num_duplicates, generators = qc.generators_H()
    assert num_duplicates == 1
    assert set(generators) == {Pauli('XIX'), Pauli('IZI'), Pauli('XII')}


def test_generators_loss():
    """A simple hand-crafted test for commutation of the generators to the loss operator."""

    qc = qc_for_generator_testing()

    # Test full support. Should catch qubit ordering issues.
    loss_support = [0, 1, 2]
    num_duplicates_H, generators_H = qc.generators_loss(loss_support)
    num_duplicates, generators = qc.generators_loss(loss_support)
    assert num_duplicates == num_duplicates_H
    assert set(generators) == set(generators_H)

    # Test partial support.
    loss_support = [2, 1]
    num_duplicates, generators = qc.generators_loss(loss_support)
    assert num_duplicates == 1
    assert set(generators) == {Pauli('IIX'), Pauli('IZI'), Pauli('III')}

    loss_support = [1, 0]
    num_duplicates, generators = qc.generators_loss(loss_support)
    assert num_duplicates == 2
    assert set(generators) == {Pauli('XII'), Pauli('IZI')}

    loss_support = [0]
    num_duplicates, generators = qc.generators_loss(loss_support)
    assert num_duplicates == 2
    assert set(generators) == {Pauli('XII'), Pauli('III')}


def test_pauli_group():
    """Test the pauli group is generated correctly."""

    generators = [Pauli('I')]
    multiplicity, pauli_group = CliffordPhi.pauli_group(generators)
    assert multiplicity == 2
    assert pauli_group == {Pauli('I')}

    generators = [Pauli('I'*10)]
    multiplicity, pauli_group = CliffordPhi.pauli_group(generators)
    assert multiplicity == 2
    assert pauli_group == {Pauli('I'*10)}

    generators = [Pauli('Y')]
    multiplicity, pauli_group = CliffordPhi.pauli_group(generators)
    assert multiplicity == 1
    assert pauli_group == {Pauli('I'), Pauli('Y')}

    generators = [Pauli('I'), Pauli('Z')]
    multiplicity, pauli_group = CliffordPhi.pauli_group(generators)
    assert multiplicity == 2
    assert pauli_group == {Pauli('I'), Pauli('Z')}

    generators = [Pauli('XZ'), Pauli('ZX'), Pauli('YY'), Pauli('II')]
    multiplicity, pauli_group = CliffordPhi.pauli_group(generators)
    assert multiplicity == 4
    assert pauli_group == {Pauli('II'), Pauli('XZ'), Pauli('ZX'), Pauli('YY')}


def matrix_average(state, pauli):
    """Compute the average of a pauli operator over a state."""
    state = state.data
    H = pauli.to_matrix()
    return np.real(state.conj().T @ H @ state)


def test_group_average():

    # Trivial circuit and group
    num_qubits = 3
    qc = CliffordPhi(num_qubits)
    state_0 = Statevector(qc)

    pauli_H = Pauli('ZZZ')
    multiplicity, group = CliffordPhi.pauli_group([Pauli('I'*num_qubits)])

    assert np.allclose(
        qc.group_average(group, pauli_H),
        matrix_average(state_0, pauli_H)
    )

    # Trivial circuit non-trivial group
    num_qubits = 4
    qc = CliffordPhi(num_qubits)
    state_0 = Statevector(qc)

    pauli_H = Pauli('IIII')
    multiplicity, group = CliffordPhi.pauli_group([Pauli('XYZI'), Pauli('ZZXX'), Pauli('IYIY')])

    average = 0
    for pauli in group:
        state = state_0.evolve(pauli)
        average += matrix_average(state, pauli_H)

    assert np.allclose(
        qc.group_average(group, pauli_H),
        average
    )

    # Non-trivial circuit trivial group
    num_qubits = 3
    qc = CliffordPhi.from_quantum_circuit(random_clifford(num_qubits, seed=0).to_circuit())
    state_0 = Statevector(qc)

    pauli_H = Pauli('I'*num_qubits)
    group = {Pauli('I'*num_qubits)}

    assert np.allclose(
        qc.group_average(group, pauli_H),
        matrix_average(state_0, pauli_H)
    )

    # Non-trivial circuit non-trivial group
    num_qubits = 4
    # qc = CliffordPhi.from_quantum_circuit(random_clifford(num_qubits, seed=1).to_circuit())
    qc = CliffordPhi(num_qubits)
    qc.cx(0, 1)
    qc.h(2)
    qc.s(3)
    qc.cz(3, 0)
    qc.sdg(3)
    state_qc = Statevector(qc)

    pauli_H = Pauli('IZIZ')
    multiplicity, group = CliffordPhi.pauli_group([Pauli('IIZI'), Pauli('IZIZ')])
    print('group', group)
    average = 0
    for pauli in group:
        state = state_qc.evolve(pauli)
        average += matrix_average(state, pauli_H)

    print('matrix average', average)
    print('group average', qc.group_average(group, pauli_H))
    assert np.allclose(
        qc.group_average(group, pauli_H),
        average
    )


