import random
from functools import reduce
from itertools import product

from qiskit import QuantumCircuit, QiskitError

import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import RZZGate, RZGate, RYGate, RXGate
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Operator, Statevector, DensityMatrix, SparsePauliOp


class CliffordPhi(QuantumCircuit):

    def __init__(self, n):
        super().__init__(n)

    @staticmethod
    def from_quantum_circuit(circuit):
        """Construct a CliffordPhi from a quantum circuit."""
        qc = CliffordPhi(circuit.num_qubits)
        qc.data = circuit.data
        return qc

    def gates(self):
        clifford_gates = []
        parametric_gates = []

        clifford_gate = Clifford(QuantumCircuit(self.num_qubits))
        for gate, qargs, cargs in self.data:
            try:
                # If the gate is clifford extend the current one.
                gate = Clifford(gate)
                clifford_gate = clifford_gate.compose(gate, [q._index for q in qargs])
            except QiskitError:
                # If the gate is not clifford, add the current accumulated clifford one to the list and start a new one.
                clifford_gates.append(clifford_gate)
                clifford_gate = Clifford(QuantumCircuit(self.num_qubits))

                # If the gate is parametric add it to the parametric list.
                if self._is_parametric(gate):
                    pauli_rotation = PauliRotation(gate, qargs, len(clifford_gates) - 1)
                    parametric_gates.append(pauli_rotation)

                else:
                    raise TypeError(f"Gate {gate} is neither Clifford nor parametric Pauli")

        # Append the last clifford gate, whether trivial or not
        clifford_gates.append(clifford_gate)

        return clifford_gates, parametric_gates

    def clifford_gates(self):
        """Clifford gates in the circuit."""
        clifford_gates, parametric_gates = self.gates()
        return clifford_gates

    def parametric_gates(self):
        """Parametric gates in the circuit."""
        clifford_gates, parametric_gates  = self.gates()
        return parametric_gates

    @staticmethod
    def _is_parametric(gate):
        return gate.params and isinstance(gate.params[0], ParameterExpression)

    def empiric_average(self, loss, batch_size=100):
        parameters_batch = 2*np.pi*np.random.rand(batch_size, self.num_parameters)
        total_loss = 0
        for parameters in parameters_batch:
            qc = self.bind_parameters(parameters)
            state = Statevector(qc)
            losses = [c*state.expectation_value(p) for c, p in zip(loss.coefficients, loss.paulis)]
            total_loss += sum(losses)

        return total_loss/batch_size

    def lattice_average(self, loss):
        """Compute loss average by summing over all lattice points."""

        state_0 = Statevector.from_label('0' * self.num_qubits)

        def loss_func(params):
            qc = self.bind_parameters(params)
            state = state_0.evolve(qc)

            averages = [coeff * state.expectation_value(pauli) for coeff, pauli in zip(loss.coefficients, loss.paulis)]
            return sum(averages)

        grid = list(product(*[[0, 1]] * len(self.parameters)))
        grid = np.pi * np.array(grid)
        losses = [loss_func(p) for p in grid]

        return sum(losses) / len(losses)

    def average(self, loss):
        """Compute the average of the loss function over all parameters in the circuit."""
        individual_averages = [
            coeff * self.average_pauli(pauli) for coeff, pauli in zip(loss.coefficients, loss.paulis)
        ]
        return sum(individual_averages)

    def average_pauli(self, pauli):
        """Compute the average of a Pauli Hamiltonian over all parameters in the circuit."""
        state_0 = StabilizerState(QuantumCircuit(self.num_qubits))
        state = self.evolve_state_using_all_clifford_gates(state_0)
        bare_average = state.expectation_value(pauli)
        assert bare_average in [-1, 0, 1], f"Average of {pauli} is {bare_average}"
        if bare_average:
            generators = self.commuted_generators()
            all_commute = self.all_commute(generators, pauli)
            if all_commute:
                return bare_average

        return 0

    def commuted_generators(self):
        """Generators of the parametric gates commuted to the end of the circuit"""
        clifford_gates = self.clifford_gates()
        generators = []
        for gate in self.parametric_gates():
            generator = gate.pauli_operator()
            for clifford in clifford_gates[gate.after_clifford_num+1:]:
                generator = generator.evolve(clifford, frame='s')
            generators.append(generator)

        return generators

    @staticmethod
    def all_commute(generators, pauli):
        """Check if all generators commute with a Pauli operator."""
        for generator in generators:
            if not pauli.commutes(generator):
                return False
        return True

    @staticmethod
    def support(pauli):
        return [i for i, gate in enumerate(pauli.to_label()) if gate != 'I']

    def group_average(self, group, pauli):
        """Compute the average (unnormalized) of a Pauli Hamiltonian over a group of Clifford operators."""

        state_0 = StabilizerState(QuantumCircuit(self.num_qubits))
        group_sum = 0

        for element in group:
            state = self.evolve_state_using_all_clifford_gates(state_0)
            state = state.evolve(element)
            group_sum += state.expectation_value(pauli)

        return group_sum

    def evolve_pauli_using_all_clifford_gates(self, pauli):
        """Evolve a Pauli operator using all Clifford gates in the circuit."""
        for gate in self.clifford_gates():
            pauli = pauli.evolve(gate)
        return pauli

    def evolve_state_using_all_clifford_gates(self, state=None):
        """Evolve a state using all clifford gates ."""
        if state is None:
            state = StabilizerState(QuantumCircuit(self.num_qubits))

        for gate in self.clifford_gates():
            state = state.evolve(gate)

        return state

    @staticmethod
    def pauli_group(generators):
        """Compute the abelian group generated by a list of pauli operators."""
        if not generators:
            return None, {}

        trivial_generator = Pauli('I'*generators[0].num_qubits)
        non_trivial_generators = [g for g in generators if not g.equiv(trivial_generator)]
        num_dependent_generators = len(generators) - len(non_trivial_generators)

        group = [Pauli('I'*generators[0].num_qubits)]
        for generator in non_trivial_generators:
            for g in group:
                if g.equiv(generator):
                    num_dependent_generators += 1
                    break

            else:
                group_times_generator = [g.compose(generator) for g in group]
                group.extend(group_times_generator)

        multiplicity = 2**num_dependent_generators
        group_set = set(group)
        assert len(group) == len(group_set), 'Some generators are dependent'
        return multiplicity, group_set

    @staticmethod
    def restore_label(label, support, num_qubits):
        """Restore a label to the full length of the circuit."""
        restored_label = ['I'] * num_qubits
        for i, qubit in enumerate(support):
            restored_label[qubit] = label[i]
        return ''.join(restored_label)

    def generators_loss(self, loss_support):
        """Generators of a Pauli group relevant for the qubits at loss support.

        Note that loss support qubit ordering follows the qiskit little-endian convention.
        """
        num_duplicates, generators = self.generators_H()
        generator_labels = [g.to_label() for g in generators]
        projected_labels = [''.join([l[i] for i in loss_support]) for l in generator_labels]
        distinct_projected_labels = list(dict.fromkeys(projected_labels))

        num_duplicates += len(generator_labels) - len(distinct_projected_labels)

        restored_labels = [self.restore_label(l, loss_support, self.num_qubits) for l in distinct_projected_labels]
        distinct_generatros = [Pauli(l) for l in restored_labels]

        return num_duplicates, distinct_generatros

    def generators_H(self):
        """Generators of a Pauli subgroup relevant for a generic loss function"""
        num_duplicates_0, generators_0 = self.generators_0()
        generators = [self.evolve_pauli_using_all_clifford_gates(g) for g in generators_0]
        for generator in generators:
            generator.phase = 0

        return num_duplicates_0, generators

    def generators_0(self):
        """Generators of a Pauli subgroup acting non-trivially after commuted to the vacuum state"""
        all_generators = []
        clifford_gates = self.clifford_gates()

        # Commute all the generators to the vacuum state
        for gate in self.parametric_gates():
            generator = gate.pauli_operator()
            for clifford in clifford_gates[:gate.after_clifford_num+1]:
                generator = generator.evolve(clifford.adjoint())
            all_generators.append(generator)

        # Remove Z and phase factors from the generators.
        for generator in all_generators:
            generator.z = np.array([False]*self.num_qubits)
            generator.phase = 0

        # Remove duplicate generators
        generator_labels = [g.to_label() for g in all_generators]
        distinct_labels = list(dict.fromkeys(generator_labels))
        distinct_generators = [Pauli(l) for l in distinct_labels]

        num_duplicates = len(all_generators) - len(distinct_generators)
        return num_duplicates, distinct_generators


class PauliRotation:

    pauli_gates = {
        'rx': [RXGate, 'X'],
        'ry': [RYGate, 'Y'],
        'rz': [RZGate, 'Z'],
        'rzz': [RZZGate, 'ZZ'],
    }

    def __init__(self, gate, qargs, after_clifford_num):
        self.gate = gate
        self.qubit_indices = [q._index for q in qargs]
        self.after_clifford_num = after_clifford_num
        self.num_qubits = qargs[0]._register.size

    def pauli_operator(self):
        labels = ['I']*self.num_qubits
        for n, index in enumerate(self.qubit_indices):
            labels[index] = PauliRotation.pauli_gates[self.gate.name][1][n]
        labels = labels[::-1]  # Qiskits little-endian convention
        return Pauli(''.join(labels))


X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


class Loss:
    def __init__(self, coefficients, paulis):
        self.coefficients = coefficients
        self.paulis = paulis

    @staticmethod
    def from_state(state):
        rho = DensityMatrix(state)
        pauli_list = SparsePauliOp.from_operator(rho)

        return Loss(pauli_list.coeffs, pauli_list.paulis)

    def matrix(self):
        return sum(np.array([c * p.to_matrix() for c, p in zip(self.coefficients, self.paulis)]))


def multi_kronecker(matrix_list):
    return reduce(lambda m1, m2: np.kron(m1, m2), matrix_list)


class PauliString:
    def __init__(self, z, x):
        assert len(z) == len(x), f'Z and X stabilizers must have the same length, got Z:{len(z)} X:{len(x)}'
        self.z = z
        self.x = x
        self.num_qubits = len(z)

    def unitary(self):
        z_list = [Z if z == 1 else np.identity(2) for z in self.z]
        x_list = [X if x == 1 else np.identity(2) for x in self.x]
        return multi_kronecker(x_list) @ multi_kronecker(z_list)

    @staticmethod
    def product(s1, s2):
        z12 = (s1.z + s2.z) % 2
        x12 = (s1.x + s2.x) % 2
        return PauliString(z12, x12)

    @staticmethod
    def trivial(num_qubits):
        z = np.zeros(num_qubits, dtype=int)
        x = np.zeros(num_qubits, dtype=int)
        return PauliString(z, x)

    def __repr__(self):
        z_str = ''.join([str(z) for z in self.z])
        x_str = ''.join([str(x) for x in self.x])
        return f'PauliString({z_str}|{x_str})'

    def __eq__(self, other):
        if isinstance(other, PauliString):
            return np.array_equal(self.z, other.z) and np.array_equal(self.x, other.x)
        return False

    def __hash__(self):
        return int(''.join([str(i) for i in np.concatenate([self.z, self.x])]), 2)
