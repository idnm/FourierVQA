from itertools import product, combinations

import numpy as np
from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import RZZGate, RZGate, RYGate, RXGate, IGate
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Statevector, DensityMatrix, SparsePauliOp


class CliffordPhiVQA:
    def __int__(self, circuit, loss):
        self.circuit = circuit
        self.loss = loss
        self.fourier_modes = []

    def compute_fourier_mode(self, order):
        # Compute previous Fourier modes if necessary
        if len(self.fourier_modes) < order:
            self.compute_fourier_mode(order-1)

        poly_dict = {}
        for parameter_configuration in self.parameter_configurations(order):

            def pcircuit(p):
                parameter_dict = dict(zip(parameter_configuration, p))
                return self.circuit.fix_parameters(parameter_dict)

            def centered_average(p):
                average = pcircuit(p).average(self.loss)
                shift = sum([fourier_mode.evaluate_at(p) for fourier_mode in self.fourier_modes[:order]])
                return average - shift
            poly_dict[parameter_configuration] = TrigonometricPolynomial.from_function(centered_average)

        fourier_mode = FourierMode(poly_dict)
        self.fourier_modes.append[fourier_mode]

        return fourier_mode

    @staticmethod
    def parameter_configurations(num_parameters, order):
        return combinations(range(num_parameters), order)


class FourierMode:
    def __int__(self, poly_dict):
        self.poly_dict = poly_dict

    def evaluate_at(self, parameters):
        parameters = np.array(parameters)
        return sum([polynomial.evaluate_at(parameters[configuration]) for configuration, polynomial in self.poly_dict.items()])


class TrigonometricPolynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.num_parameters = int(np.log2(len(coefficients)))

    def evaluate_at(self, parameters):
        return (self.coefficients * np.array([m(parameters) for m in self.monomials(self.num_parameters)])).sum()

    @staticmethod
    def from_function(f, num_parameters):
        coefficients = [f(g) for g in np.pi/2*TrigonometricPolynomial.binary_grid(num_parameters)]
        return TrigonometricPolynomial(coefficients)

    @staticmethod
    def binary_grid(num_parameters):
        grid = list(product(*[[0, 1]] * num_parameters))
        return np.array(grid)

    @staticmethod
    def monomial(power):
        return lambda p: np.product(np.cos(p)**(1-power) * np.sin(p)**power)

    @staticmethod
    def monomials(num_parameters):
        return [TrigonometricPolynomial.monomial(power) for power in TrigonometricPolynomial.binary_grid(num_parameters)]


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
        for instruction in self.data:
            gate, qargs, cargs = instruction
            try:
                # If the gate is clifford extend the current one.
                gate = Clifford(gate)
                clifford_gate = clifford_gate.compose(gate, [q._index for q in qargs])
            except QiskitError:
                # If the gate is not clifford, add the current accumulated clifford one to the list and start a new one.
                clifford_gates.append(clifford_gate)
                clifford_gate = Clifford(QuantumCircuit(self.num_qubits))

                # If the gate is parametric add it to the parametric list.
                if instruction.operation.is_parameterized():
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

        grid = TrigonometricPolynomial.binary_grid(self.num_parameters)
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

    def evolve_state_using_all_clifford_gates(self, state=None):
        """Evolve a state using all clifford gates ."""
        if state is None:
            state = StabilizerState(QuantumCircuit(self.num_qubits))

        for gate in self.clifford_gates():
            state = state.evolve(gate)

        return state

    def fix_parameters(self, parameter_dict):

        qc = self.copy()
        parametric_instructions = [instruction for instruction in qc.data if instruction.operation.is_parameterized()]
        for parameter_index, parameter_value in parameter_dict.items():
            gate = parametric_instructions[parameter_index].operation
            gate = PauliRotation.fix_parameter(gate, parameter_value)

        return qc


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

    @staticmethod
    def fix_parameter(gate, parameter_value):
        num_qubits = gate.num_qubits
        if np.allclose(parameter_value, 0):
            gate = IGate()
            gate.num_qubits = num_qubits
        elif np.allclose(parameter_value, np.pi/2):
            gate_pauli = PauliRotation.pauli_gates[gate.name][1]
            gate = PauliRotation.pauli_root(Pauli(gate_pauli))
        return gate

    # @staticmethod
    # def pauli_root(pauli):
    #     """Construct a Clifford gate implementing (1-iP)/sqrt(2)"""
    #     qc = QuantumCircuit(pauli.num_qubits)
    #     for qubit, label in enumerate(pauli.to_label()):
    #         print(f'appendig root of {Pauli(label)}')
    #         qc.append(PauliRotation.pauli_root_single(Pauli(label)), [qubit])
    #     return qc.reverse_bits()

    @staticmethod
    def pauli_root(pauli):
        """Construct a Clifford gate implementing (1-iP)/sqrt(2)"""

        num_qubits = pauli.num_qubits
        z_generators = ['I' * i + 'Z' + 'I' * (num_qubits - i - 1) for i in range(num_qubits)][::-1]
        x_generators = ['I' * i + 'X' + 'I' * (num_qubits - i - 1) for i in range(num_qubits)][::-1]

        stabilizers = [Pauli(g) if Pauli(g).commutes(pauli) else -1j * Pauli(g).compose(pauli) for g in z_generators]
        destabilizers = [Pauli(g) if Pauli(g).commutes(pauli) else -1j * Pauli(g).compose(pauli) for g in x_generators]

        stabilizers = [s.to_label() for s in stabilizers]
        destabilizers = [s.to_label() for s in destabilizers]

        cliff = Clifford.from_dict({'stabilizer': stabilizers, 'destabilizer': destabilizers})
        return cliff.to_circuit()

    @staticmethod
    def pauli_root_single(pauli):
        qc = QuantumCircuit(1)
        label = pauli.to_label()
        if label == 'Z':
            qc.s(0)
        elif label == 'X':
            qc.h(0)
            qc.s(0)
            qc.h(0)
        elif label == 'Y':
            qc.sdg(0)
            qc.h(0)
            qc.s(0)
            qc.h(0)
            qc.s(0)
        return Clifford(qc).to_circuit()


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
