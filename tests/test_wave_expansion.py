import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_clifford, Operator, random_statevector, Pauli

from wave_expansion import CliffordPhi, PauliRotation, Loss, TrigonometricPolynomial, CliffordPhiVQA, FourierMode


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


def test_loss_from_state():
    """Test reconstructing density matrix from the loss function."""
    state = random_statevector(2**4, seed=0)
    loss = Loss.from_state(state)
    rho_matrix = np.outer(state.data, state.data.conj().T)

    assert np.allclose(rho_matrix, loss.matrix())


def test_average():
    """Test computing the average of a Pauli operator over a density matrix."""

    for num_qubits in range(2, 3):
        for num_parameters in range(4):
            qc = random_clifford_phi(num_qubits, num_parameters, seed=41)
            state = random_statevector(2 ** num_qubits, seed=42)

            loss = Loss.from_state(state)

            assert np.allclose(qc.lattice_average(loss), qc.average(loss))


def test_trigonometric_polynomial():
    """Test constructing and reconstructing trigonometric polynomial"""

    # Single variable
    def p(x):
        return 0.2 * np.cos(x[0]) - 1.3 * np.sin(x[0])

    tp = TrigonometricPolynomial.from_function(p, 1)

    xx = np.linspace(0, 2 * np.pi, 40)
    p_values = np.array([p([x]) for x in xx])
    tp_values = np.array([tp.evaluate_at([x]) for x in xx])

    assert np.allclose(p_values, tp_values)

    # Three variables
    def p(x):
        return 0.2*np.cos(x[0])*np.cos(x[1])*np.cos(x[2])\
               -2.*np.cos(x[0])*np.cos(x[1])*np.sin(x[2])\
               +12.34*np.cos(x[0])*np.sin(x[1])*np.cos(x[2])\
               -0.786*np.cos(x[0])*np.sin(x[1])*np.sin(x[2])\
               +np.pi*np.sin(x[0])*np.cos(x[1])*np.cos(x[2])\
               +np.e*np.sin(x[0])*np.cos(x[1])*np.sin(x[2])\
               -0.7112*np.sin(x[0])*np.sin(x[1])*np.cos(x[2])\
               -10.*np.sin(x[0])*np.sin(x[1])*np.sin(x[2])

    tp = TrigonometricPolynomial.from_function(p, 3)

    xx = 2*np.pi*np.random.rand(100, 3)
    p_values = [p(x) for x in xx]
    tp_values = [tp.evaluate_at(x) for x in xx]

    assert np.allclose(p_values, tp_values)


def test_pauli_root():

    for label in ['X', 'Y', 'Z', 'XY', 'XYZ', 'IXYZ', 'IXIYYX', 'IXIZZY']:
        pauli = Pauli(label)
        num_qubits = pauli.num_qubits

        # Take square root directly
        qc = QuantumCircuit(num_qubits)
        qc.append(pauli.to_instruction().power(1/2), range(num_qubits))

        # Use clifford square root
        qc_sqrt = QuantumCircuit(num_qubits)
        qc_sqrt.append(PauliRotation.pauli_root(pauli), range(num_qubits))

        assert Operator(qc).equiv(Operator(qc_sqrt))


def parametric_circuits_are_equivalent(qc0, qc1):

    np.random.seed(42)
    random_parameters = np.random.rand(10, qc0.num_parameters)
    return all([Operator(qc0.bind_parameters(p)).equiv(Operator(qc1.bind_parameters(p))) for p in random_parameters])


def test_fix_parameters():

    num_qubits = 4
    num_parameters = 6
    qc = random_clifford_phi(num_qubits, num_parameters)
    parameters = qc.parameters

    parameter_dict = {1: 0, 3: np.pi/2, 5: np.pi/2}

    qc_fix_direct = qc.bind_parameters({parameters[key]: value for key, value in parameter_dict.items()})
    qc_fix_clifford = qc.fix_parameters(parameter_dict)

    assert qc_fix_direct.num_parameters == num_parameters - 3
    assert qc_fix_clifford.num_parameters == num_parameters - 3
    assert parametric_circuits_are_equivalent(qc_fix_direct, qc_fix_clifford)


def test_fourier_mode_evaluation():
    def poly1(x):
        return np.cos(x[0]) * np.sin(x[1]) - np.sin(x[0]) * np.cos(x[1])

    def poly2(x):
        return 0.5 * np.sin(x[0]) * np.sin(x[1]) + 10 * np.sin(x[0]) * np.cos(x[1])

    def poly3(x):
        return -5 ** np.sin(x[0]) * np.cos(x[1])

    tp1, tp2, tp3 = [TrigonometricPolynomial.from_function(p, 2) for p in [poly1, poly2, poly3]]
    poly_dict = {(0, 1): tp1, (0, 2): tp2, (1, 2): tp3}
    fmode2 = FourierMode(poly_dict)

    x = np.array([1., 2., 3.])
    val = tp1.evaluate_at(x[[0, 1]]) + tp2.evaluate_at(x[[0, 2]]) + tp3.evaluate_at(x[[1, 2]])
    assert np.allclose(val, fmode2.evaluate_at(x))
    assert np.allclose(fmode2.average({1: 3., 2: 4.}), tp3.evaluate_at([3., 4.]))


def test_first_fourier():
    num_qubits = 2
    num_parameters = 1

    qc = random_clifford_phi(num_qubits, num_parameters)
    loss = Loss.from_state(random_statevector(2 ** num_qubits, seed=0))
    vqa = CliffordPhiVQA(qc, loss)
    vqa.compute_fourier_mode(1)

    vqa.evaluate_loss_at([1.])
