import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate, RZGate, RYGate
from qiskit.quantum_info import random_clifford, Operator

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

    return qc


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