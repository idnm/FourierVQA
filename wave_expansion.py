import copy
from functools import reduce
from itertools import product, combinations
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import ParameterExpression, Instruction, Gate, CircuitInstruction, Parameter
from qiskit.circuit.library import RZZGate, RZGate, RYGate, RXGate, IGate, PauliEvolutionGate
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Statevector, DensityMatrix, SparsePauliOp, Operator, \
    pauli_basis, random_pauli
from scipy.special import binom


class PauliCircuit:
    def __init__(self, paulis, final_clifford=None, parameters=None):
        self.paulis = paulis
        self.final_clifford = final_clifford
        self.parameters = parameters

    @property
    def num_qubits(self):
        return self.paulis[0].num_qubits

    def to_parameterized_circuit(self):
        if self.parameters is not None:
            parameters = self.parameters
        else:
            parameters = [Parameter(f'x{i}') for i in range(len(self.paulis))]

        qc = QuantumCircuit(self.num_qubits)
        for pauli, parameter in zip(self.paulis, parameters):
            gate = PauliEvolutionGate(pauli, 1/2*parameter)
            qc.append(gate, range(qc.num_qubits))
        if self.final_clifford:
            qc.append(self.final_clifford.to_instruction(), range(qc.num_qubits))

        return qc

    def expectation_value(self, observable, parameters):
        qc = self.to_parameterized_circuit().bind_parameters(parameters)
        state = Statevector(qc)
        return state.expectation_value(observable)

    @staticmethod
    def from_parameterized_circuit(qc):
        gates, parameters = PauliCircuit.clifford_pauli_data(qc)
        pauli_gates, final_clifford = PauliCircuit.commute_all_cliffords_to_the_end(gates)
        return PauliCircuit(pauli_gates, final_clifford, parameters)

    @staticmethod
    def random(num_qubits, num_paulis, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(0, 10000, size=num_paulis)
        paulis = [random_pauli(num_qubits, seed=s) for s in seeds]

        return PauliCircuit(paulis)

    @staticmethod
    def commute_all_cliffords_to_the_end(gates):
        commuted_something = True
        while commuted_something:
            commuted_something, gates = PauliCircuit.commute_the_first_clifford(gates)
        num_pauli_gates = len([True for gate_type, _, _ in gates if gate_type == 'pauli'])
        pauli_gates = [gate for _, gate, _ in gates[:num_pauli_gates]]
        clifford_gates = gates[num_pauli_gates:]

        num_qubits = pauli_gates[0].num_qubits
        qc = QuantumCircuit(num_qubits)
        for _, gate, qubits in clifford_gates:
            qc.append(gate.to_instruction(), qubits)
        final_clifford = Clifford.from_circuit(qc)
        return pauli_gates, final_clifford

    @staticmethod
    def commute_the_first_clifford(gates):

        for i in range(len(gates) - 1):
            gate_type, gate, qubits = gates[i]
            next_gate_type, next_gate, next_qubits = gates[i + 1]
            if gate_type == 'clifford' and next_gate_type == 'pauli':
                gates[i] = 'pauli', next_gate.evolve(gate, qubits), next_qubits
                gates[i + 1] = 'clifford', gate, qubits
                commuted_something = True
                break
        else:
            commuted_something = False

        return commuted_something, gates

    @staticmethod
    def clifford_pauli_data(qc, parametric_to_pauli_dict=None):
        """Unroll the circuit according to circuit.data and identify each gate either as a Clifford gate
        or as a Pauli rotation."""

        if not parametric_to_pauli_dict:
            parametric_to_pauli_dict = {}
        gates = []
        parameters = []
        for gate, qargs, cargs in qc.data:
            qubits = [q._index for q in qargs]
            try:
                new_gate = Clifford(gate)
                gate_type = 'clifford'
            except (QiskitError, TypeError):
                if not gate.is_parameterized():
                    raise ValueError(f'Gate {gate.name} is neither clifford nor parametric.')
                if len(gate.params) != 1:
                    raise ValueError(
                        f'Parametric gates must have a single parameter. Got gate {gate.name} with gate.params={gate.params})')
                if gate.name in parametric_to_pauli_dict:
                    pauli = parametric_to_pauli_dict[gate.name]
                else:
                    pauli = PauliCircuit.pauli_generator_from_parametric_gate(gate)
                    parametric_to_pauli_dict[gate.name] = pauli

                parameter = gate.params[0]
                parameters.append(parameter)
                new_gate = PauliCircuit.full_pauli_generator(pauli, qubit_indices=qubits, num_qubits=qc.num_qubits)
                qubits = list(range(qc.num_qubits))
                gate_type = 'pauli'

            gates.append((gate_type, new_gate, qubits))

        return gates, parameters

    @staticmethod
    def pauli_generator_from_parametric_gate(gate):
        gate = copy.deepcopy(gate)
        gate.params = [Parameter('theta')]  # In case gate came with parameter like 0.5*Parameter('theta')
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))

        # Check that gate(pi) is a pauli gate.
        gate_at_pi = Operator(qc.bind_parameters([np.pi])).data

        for pauli in pauli_basis(gate.num_qubits):
            if np.allclose(gate_at_pi, -1j * pauli.to_matrix()):
                pauli = pauli
                break
        else:
            raise ValueError(f'Gate {gate.name} at pi is not a pauli gate.')

        # Check that gate(x) is exponential of the pauli gate for other parameters.
        # This will fail e.g. for exp(i x (Z1+Z2)).
        xx = np.linspace(0, 2 * np.pi, 19)
        gate_values = np.array([Operator(qc.bind_parameters([x])).data for x in xx])
        pauli_rotation_values = np.array([PauliRotation.matrix(pauli, x) for x in xx])

        if not np.allclose(gate_values, pauli_rotation_values):
            raise ValueError(f'Gate {gate.name} is not a Pauli rotation.')

        return pauli

    @staticmethod
    def full_pauli_generator(pauli, qubit_indices, num_qubits):
        short_generator = pauli.to_label()
        generator = ['I'] * num_qubits
        for label, q in zip(short_generator, qubit_indices[::-1]):
            generator[q] = label

        return Pauli(''.join(generator)[::-1])


class PauliSpace:
    def __init__(self, paulis):
        self.paulis = paulis

        self.independent_paulis = None
        self.dependent_paulis = None
        self.normal_form = None
        self.decomposition_matrix = None

    @property
    def dim(self):
        return self.paulis[0].num_qubits

    @property
    def num_paulis(self):
        return len(self.paulis)

    def rank(self, up_to_pauli_num):
        return len([n for n in self.independent_paulis if n <= up_to_pauli_num])

    def list_decomposition(self, decomposition):
        return [self.independent_paulis[n] for n, coefficient in enumerate(decomposition) if coefficient]

    def decomposition_contains_pauli(self, decomposition, num_pauli):
        return num_pauli in self.list_decomposition(decomposition)

    def is_independent(self, num_pauli):
        return num_pauli in self.independent_paulis

    def decomposition_requires_paulis(self, decomposition):
        if np.all(decomposition == 0):
            return 0
        return max(self.list_decomposition(decomposition))

    def compute_decomposition(self, observable, num_paulis):
        normal_form = self.normal_form[:num_paulis]
        decomposition_matrix = self.decomposition_matrix[:num_paulis]
        normal_form, decomposition_matrix = self.extend_normal_form(normal_form, decomposition_matrix, observable.x)
        return decomposition_matrix[-1]

    def construct(self):
        independent_paulis, dependent_paulis, normal_form, decomposition_matrix = self.basis_and_decompositions(self.paulis)
        self.independent_paulis = independent_paulis
        self.dependent_paulis = dependent_paulis
        self.normal_form = normal_form
        self.decomposition_matrix = decomposition_matrix

    @staticmethod
    def basis_and_decompositions(paulis):
        paulis_x = [pauli.x for pauli in paulis]

        independent_paulis = []
        dependent_paulis = []

        normal_form = []
        decomposition_matrix = []

        # Find first non-zero pauli.
        for n, pauli in enumerate(paulis_x):
            if np.all(pauli == 0):
                dependent_paulis.append(n)

                normal_form.append(np.zeros(len(pauli), dtype=bool))
                decomposition_matrix.append(np.zeros(len(pauli), dtype=bool))
            else:
                independent_paulis.append(n)

                d = np.zeros(len(pauli), dtype=bool)
                d[0] = True

                normal_form.append(pauli)
                decomposition_matrix.append(d)
                break
        else:
            raise ValueError('No non-trivial paulis provided.')

        for n, pauli in list(enumerate(paulis_x))[independent_paulis[0]+1:]:

            normal_form, decomposition_matrix = PauliSpace.extend_normal_form(normal_form, decomposition_matrix, pauli)

            if np.all(normal_form[-1] == 0):
                dependent_paulis.append(n)
            else:
                independent_paulis.append(n)
                decomposition_matrix[-1][len(independent_paulis)-1] = True

            if len(independent_paulis) == len(pauli):
                break

        return independent_paulis, dependent_paulis, normal_form, decomposition_matrix

    @staticmethod
    def extend_normal_form(normal_form, decomposition_matrix, pauli):
        pauli = copy.deepcopy(pauli)

        nonzero_rows = []
        starting_indices = []
        for n, row in enumerate(normal_form):
            if not np.all(row == 0):
                (i,) = row.nonzero()
                starting_indices.append(i[0])
                nonzero_rows.append(n)

        decomposition = np.zeros(len(pauli), dtype=bool)
        for i, n_row in sorted(zip(starting_indices, nonzero_rows)):
            if pauli[i]:
                pauli ^= normal_form[n_row]
                decomposition ^= decomposition_matrix[n_row]

        return normal_form+[pauli], decomposition_matrix+[decomposition]

    def update_decomposition(self, decomposition, num_pauli):
        if decomposition is None:
            return None
        return decomposition ^ self.decomposition_matrix[num_pauli]


class FourierComputation:
    def __init__(self, pauli_circuit, pauli_observable):
        self.pauli_circuit = pauli_circuit
        pauli_space = PauliSpace(pauli_circuit.paulis)
        pauli_space.construct()
        self.pauli_space = pauli_space

        self.original_observable = pauli_observable
        if self.pauli_circuit.final_clifford:
            self.observable = self.original_observable.evolve(self.pauli_circuit.final_clifford)
        else:
            self.observable = self.original_observable

        self.complete_nodes = []
        self.incomplete_nodes = []
        self.num_iterations = 0

    def run(self, check_admissible=True, max_order=None):

        # Initialize the computation if it wasn't.
        if not self.incomplete_nodes and not self.complete_nodes:
            root = FourierComputationNode(self.pauli_space.num_paulis, self.observable, ())
            root.remove_commuting_paulis(self.pauli_space)
            if root.is_complete:
                self.complete_nodes = [root]
            else:
                self.incomplete_nodes = [root]
            self.num_iterations = 0

        # If not provided, set max order to the number of parameters.
        if not max_order:
            max_order = self.pauli_space.num_paulis

        # If iterations exist number adjust the number of new iterations
        num_iterations = max_order-self.num_iterations

        # Run recursive algorithm. Each iteration computes all Fourier terms of at the next order.
        for _ in range(num_iterations):
            self.num_iterations += 1
            self.incomplete_nodes, self.complete_nodes = self.iteration(
                self.incomplete_nodes, self.complete_nodes, check_admissible)
            if len(self.incomplete_nodes) == 0:
                break

    def iteration(self, incomplete_nodes, complete_nodes, check_admissibility):
        if not incomplete_nodes:
            return [], complete_nodes

        new_incomplete_nodes = []
        for node in incomplete_nodes:
            incomplete_nodes, completed_nodes = node.branch_and_refine(self.pauli_space, check_admissibility)

            new_incomplete_nodes.extend(incomplete_nodes)
            complete_nodes.extend(completed_nodes)

        return new_incomplete_nodes, complete_nodes

    def _check_type(self, t):
        if self.type is None:
            self.type = t
        else:
            assert self.type == t, f'Continuing computation of a different type {self.type} is not allowed.'

    def evaluate_at(self, parameters):
        state0 = Statevector(QuantumCircuit(self.pauli_circuit.num_qubits))
        res = 0
        for node in self.complete_nodes:
            res += state0.expectation_value(node.observable)*node.monomial(parameters)

        return res

    def order_statistics(self):
        M = len(self.paulis)
        orders = [node.order for node in self.complete_nodes]
        return [orders.count(m) for m in range(M + 1)]

    def norm_statistics(self):
        return [n / (2 ** m) for m, n in enumerate(self.order_statistics())]

    def visualize(self):
        M = len(self.paulis)

        plt.scatter(range(M + 1), np.array(self.order_statistics()) / (3 / 2) ** M)
        plt.scatter(range(M + 1), self.norm_statistics())

        plt.plot([binom(M, m) * 2 ** (m - M) / (3 / 2) ** M for m in range(M + 1)], linestyle='--')
        plt.plot([binom(M, m) * 2 ** (-M) for m in range(M + 1)], linestyle='--')


class FourierComputationNode:
    def __init__(self, num_paulis, observable, branch_history):
        self.num_paulis = num_paulis
        self.observable = observable
        self.branch_history = branch_history
        self.is_admissible = None
        self.observable_decomposition = None

    @property
    def order(self):
        return len(self.branch_history)

    @property
    def is_complete(self):
        # The node is complete is no Paulis remain.
        return self.num_paulis == 0

    def __repr__(self):
        return f'{self.num_paulis} {self.observable} {self.branch_history}'

    def remove_commuting_paulis(self, pauli_space):
        paulis = pauli_space.paulis[:self.num_paulis]
        num_commuting = 0
        for pauli in paulis[::-1]:
            if pauli.commutes(self.observable):
                num_commuting += 1
            else:
                break
        self.num_paulis -= num_commuting

    def branch_cos(self, pauli_space):
        node_cos = FourierComputationNode(
            self.num_paulis-1,
            self.observable,
            self.branch_history+((self.num_paulis-1, 0), ))
        node_cos.observable_decomposition = self.observable_decomposition
        return node_cos

    def branch_sin(self, pauli_space):
        node_sin = FourierComputationNode(
            self.num_paulis - 1,
            1j * self.observable.compose(pauli_space.paulis[self.num_paulis-1]),
            self.branch_history + ((self.num_paulis-1, 1), ))
        decomposition = pauli_space.update_decomposition(self.observable_decomposition, self.num_paulis-1)
        node_sin.observable_decomposition = decomposition
        return node_sin

    def branch_and_refine(self, pauli_space, check_admissibility):

        # Each node processes here if at the branching point. Before branching,
        # we can check admissibility of the current node itself, and if admissible, of its two branches.

        cos_admissible = True
        sin_admissible = True

        # Admissibility is checked if the corresponding flag is true,
        # and the remaining pauli operators are insufficient to decompose any observable.

        if check_admissibility and pauli_space.rank(self.num_paulis) < pauli_space.dim:
            self.is_admissible = False

            # If the observable was not decomposed before, decompose it now.
            if self.observable_decomposition is None:
                self.observable_decomposition = pauli_space.compute_decomposition(self.observable, self.num_paulis)

            # If decomposition was successful.
            if self.observable_decomposition is not None:
                # If the number of available pauli matrices is enough for decomposition, the node is admissible.
                if pauli_space.decomposition_requires_paulis(self.observable_decomposition) <= self.num_paulis:
                    self.is_admissible = True

            # If the node is not admissible, in the end, abort the computation.
            if not self.is_admissible:
                complete_nodes = [self]
                incomplete_nodes = []
                return incomplete_nodes, complete_nodes

            # If the branching Pauli is dependent, both branches are admissible.
            # If it is independent a single branch is admissible.
            num_branching_pauli = self.num_paulis
            if not pauli_space.is_independent(num_branching_pauli):
                # If observable requires the branching Pauli in its decomposition the cos branch is not admissible.
                if pauli_space.decomposition_contains_pauli(self.observable_decomposition, num_branching_pauli):
                    cos_admissible = False
                # Else the sin branch is not admissible.
                else:
                    sin_admissible = False

        # Branch and remove commuting paulis from each branch.
        nodes = []
        if cos_admissible:
            nodes.append(self.branch_cos(pauli_space))
        if sin_admissible:
            nodes.append(self.branch_sin(pauli_space))

        for node in nodes:
            node.remove_commuting_paulis(pauli_space)

        incomplete_nodes = []
        complete_nodes = []
        for node in nodes:
            if node.is_complete:
                complete_nodes.append(node)
            else:
                incomplete_nodes.append(node)

        return incomplete_nodes, complete_nodes

    def monomial(self, parameters):
        res = 1
        for num_parameter, branch in self.branch_history:
            x = parameters[num_parameter]
            if branch == 0:
                res *= np.cos(x)
            elif branch == 1:
                res *= np.sin(x)

        return res

#######################################################################################


class CliffordPhiVQA:
    def __init__(self, circuit, loss):
        self.circuit = loss.construct_circuit(circuit)
        self.loss = loss
        self.fourier_modes = []

    def evaluate_loss_at(self, parameters):
        return self.circuit.evaluate_loss_at(self.loss.hamiltonian, parameters)

    def compute_fourier_mode(self, order):
        # Compute previous Fourier modes if necessary
        if len(self.fourier_modes) < order:
            self.compute_fourier_mode(order-1)

        # Order zero is simply a constant
        if order == 0:
            tp = TrigonometricPolynomial.from_function(lambda _: self.circuit.average(self.loss.hamiltonian), 0)
            fourier_mode = FourierMode({(): tp})
            self.fourier_modes.append(fourier_mode)
            return fourier_mode

        # Each parameter configuration like (0, 2, 5) corresponds to a trigonometric polynomial.
        # We iterate over all parameter configurations and find the corresponding polynomials via their coefficients.
        poly_dict = {}
        for parameter_configuration in self.parameter_configurations(self.circuit.num_parameters, order):

            def pcircuit(fixed_parameter_values):
                fixed_parameters_dict = dict(zip(parameter_configuration, fixed_parameter_values))
                return self.circuit.fix_parameters(fixed_parameters_dict)

            def centered_average(fixed_parameter_values):
                fixed_parameters_dict = dict(zip(parameter_configuration, fixed_parameter_values))
                average = pcircuit(fixed_parameter_values).average(self.loss.hamiltonian)
                shift = sum([fourier_mode.average(fixed_parameters_dict) for fourier_mode in self.fourier_modes[:order]])
                return average - shift

            poly_dict[parameter_configuration] = TrigonometricPolynomial.from_function(centered_average, order)

        fourier_mode = FourierMode(poly_dict)
        self.fourier_modes.append(fourier_mode)

        return fourier_mode

    @staticmethod
    def parameter_configurations(num_parameters, order):
        return combinations(range(num_parameters), order)

    def fourier_expansion(self, up_to_order=None):
        max_order = self.circuit.num_parameters
        if up_to_order is None:
            up_to_order = max_order

        if up_to_order > max_order:
            raise ValueError(f'Requested order higher than the maximal {max_order}.')

        if len(self.fourier_modes) <= up_to_order:
            self.compute_fourier_mode(up_to_order)

        def f(parameters):
            return sum(fourier_mode.evaluate_at(parameters) for fourier_mode in self.fourier_modes[:up_to_order+1])

        return f


class FourierMode:
    def __init__(self, poly_dict):
        self.poly_dict = poly_dict

    def average(self, fixed_parameters_dict):
        total = 0
        for parameter_configuration, polynomial in self.poly_dict.items():
            if all([p in fixed_parameters_dict.keys() for p in parameter_configuration]):
                parameters = np.array([fixed_parameters_dict[p] for p in parameter_configuration])
                total += polynomial.evaluate_at(parameters)

        return total

    def evaluate_at(self, parameters):
        return self.average({i: p for i, p in enumerate(parameters)})


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

    def __init__(self, num_qubits, parametric_gates_to_pauli_gates_dict={}):
        super().__init__(num_qubits)
        self.parametric_gates_to_pauli_gates_dict = parametric_gates_to_pauli_gates_dict

    @property
    def clifford_pauli_data(self):
        data = []

        for gate, qargs, cargs in self.data:
            q_indices = [q._index for q in qargs]
            try:
                gate = Clifford(gate)
                instruction = [gate, q_indices]
            except (QiskitError, TypeError):
                if gate.is_parameterized():
                    if gate.name not in self.parametric_gates_to_pauli_gates_dict:
                        new_pauli_rotation_gate = PauliRotation(gate)
                        pauli = new_pauli_rotation_gate.pauli
                        pauli_root = new_pauli_rotation_gate.pauli_root
                        self.parametric_gates_to_pauli_gates_dict[gate.name] = (pauli, pauli_root)

                    pauli_rotation = PauliRotation(gate, self.parametric_gates_to_pauli_gates_dict)
                    instruction = [pauli_rotation, q_indices]
                else:
                    raise TypeError(f"Gate {gate} is neither Clifford nor parametric")

            data.append(instruction)

        return data

    @staticmethod
    def from_quantum_circuit(circuit):
        """Construct a CliffordPhi from a quantum circuit."""
        qc = CliffordPhi(circuit.num_qubits)
        qc.data = circuit.data
        return qc

    @staticmethod
    def clifford_gates(data):
        """Clifford gates in the circuit."""
        return [(gate, qargs) for gate, qargs in data if CliffordPhi.is_clifford(gate)]

    @staticmethod
    def pauli_rotation_gates(data):
        """Clifford gates in the circuit."""
        return [(gate, qargs) for gate, qargs in data if CliffordPhi.is_pauli_rotation(gate)]

    @staticmethod
    def is_pauli_rotation(gate):
        return isinstance(gate, PauliRotation)

    @staticmethod
    def is_clifford(gate):
        return not isinstance(gate, PauliRotation)
        # try:
        #     Clifford(gate)
        #     return True
        # except (QiskitError, TypeError):
        #     return False

    @staticmethod
    def parametric_gates(gate_list):
        """Parametric gates in the circuit."""
        return [gate for gate in gate_list if gate.operation.is_parametric()]

    @property
    def all_parameters(self):
        return [instruction.operation.params[0] for instruction in self.data if instruction.operation.is_parameterized()]

    def evaluate_loss_at(self, hamiltonian, parameters):
        qc = self.bind_parameters(parameters)
        state = Statevector(qc)
        pauli_losses = [c * state.expectation_value(p) for c, p in zip(hamiltonian.coeffs, hamiltonian.paulis)]
        return sum(pauli_losses)

    def empiric_average(self, hamiltonian, batch_size=100):
        parameters_batch = 2*np.pi*np.random.rand(batch_size, self.num_parameters)
        losses = [self.evaluate_loss_at(hamiltonian, p) for p in parameters_batch]
        return sum(losses)/batch_size

    def lattice_average(self, hamiltonian):
        """Compute loss average by summing over all lattice points."""

        state_0 = Statevector.from_label('0' * self.num_qubits)

        def loss_func(params):
            qc = self.bind_parameters(params)
            state = state_0.evolve(qc)

            averages = [coeff * state.expectation_value(pauli) for coeff, pauli in zip(hamiltonian.coeffs, hamiltonian.paulis)]
            return sum(averages)

        grid = TrigonometricPolynomial.binary_grid(self.num_parameters)
        grid = np.pi * np.array(grid)
        losses = [loss_func(p) for p in grid]

        return sum(losses) / len(losses)

    def average(self, hamiltonian):
        """Compute the average of the loss function over all parameters in the circuit."""
        individual_averages = [
            coeff * self.average_pauli(pauli) for coeff, pauli in zip(hamiltonian.coeffs, hamiltonian.paulis)
        ]
        return sum(individual_averages)

    def average_pauli(self, pauli):
        """Compute the average of a Pauli Hamiltonian over all parameters in the circuit."""
        state = self.stabilizer_state
        bare_average = state.expectation_value(pauli)
        assert np.allclose(np.imag(bare_average), 0), f"Average of {pauli} is not real {bare_average}"
        assert bare_average in [-1, 0, 1], f"Average of {pauli} is {bare_average}"
        bare_average = np.real(bare_average)

        if bare_average:
            generators = self.commuted_generators
            all_commute = self.all_commute(generators, pauli)
            if all_commute:
                return bare_average

        return 0

    @property
    def stabilizer_state(self):
        state_0 = StabilizerState(QuantumCircuit(self.num_qubits))
        return self.evolve_state_using_all_clifford_gates(state_0)

    @property
    def commuted_generators(self):
        """Generators of the parametric gates commuted to the end of the circuit"""
        clifford_pauli_gates = self.clifford_pauli_data
        generators = []
        for i, (gate, qubit_indices) in enumerate(clifford_pauli_gates):
            if self.is_clifford(gate):
                continue
            generator = gate.full_pauli_generator(qubit_indices, self.num_qubits)
            for clifford, qubits in self.clifford_gates(clifford_pauli_gates[i:]):
                generator = generator.evolve(clifford, qubits, frame='s')
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

        for gate, qargs in self.clifford_gates(self.clifford_pauli_data):
            state = state.evolve(gate, qargs)

        return state

    def fix_parameters(self, parameter_dict):
        data = copy.deepcopy(self.data)

        parametric_instructions = [instruction for instruction in data if instruction.operation.is_parameterized()]
        pauli_rotations = self.pauli_rotation_gates(self.clifford_pauli_data)
        for parameter_index, parameter_value in parameter_dict.items():
            # gate, qargs, cargs = parametric_instructions[parameter_index]
            num_instruction = self.num_instruction_from_num_parameter[parameter_index]
            parametric_instructions[num_instruction].operation = pauli_rotations[num_instruction][0].fix_parameter(parameter_value)

        qc = CliffordPhi(self.num_qubits)
        qc.data = data

        return qc

    @property
    def num_parameter_from_num_instruction(self):
        parameters = list(self.parameters)
        parametric_instructions = [gate for gate, _, _ in self.data if gate.is_parameterized()]
        return [parameters.index(gate.params[0]) for gate in parametric_instructions]

    @property
    def num_instruction_from_num_parameter(self):
        parameters = self.parameters
        parametric_instructions = [gate for gate, _, _ in self.data if gate.is_parameterized()]
        instruction_parameters = [gate.params[0] for gate in parametric_instructions]

        return [instruction_parameters.index(p) for p in parameters]


class PauliRotation:

    def __init__(self, gate, rotation_gates_dict={}):
        self.gate = gate
        if gate.name in rotation_gates_dict:
            self.pauli = rotation_gates_dict[gate.name][0]
            self.pauli_root = rotation_gates_dict[gate.name][1]
        else:
            self.pauli = PauliRotation.pauli_generator_from_gate(gate)
            self.pauli_root = PauliRotation.pauli_root_from_pauli(self.pauli)

    @staticmethod
    def pauli_generator_from_gate(gate):
        gate = copy.deepcopy(gate)
        gate.params = [Parameter('theta')]  # In case gate came with parameter like 0.5*Parameter('theta')
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))

        # Check that gate(pi) is a pauli gate.
        gate_at_pi = Operator(qc.bind_parameters([np.pi])).data

        for pauli in pauli_basis(gate.num_qubits):
            if np.allclose(gate_at_pi, -1j*pauli.to_matrix()):
                pauli = pauli
                break
        else:
            return None

        # Check that gate(x) is exponential of the pauli gate for other parameters.
        # This will fail e.g. for exp(i x (Z1+Z2)).
        xx = np.linspace(0, 2*np.pi, 19)
        gate_values = np.array([Operator(qc.bind_parameters([x])).data for x in xx])
        pauli_rotation_values = np.array([PauliRotation.matrix(pauli, x) for x in xx])

        if not np.allclose(gate_values, pauli_rotation_values):
            return None

        return pauli

    def full_pauli_generator(self, qubit_indices, num_qubits):
        # num_qubits = qargs[0]._register.size
        pauli = self.pauli
        short_generator = pauli.to_label()
        generator = ['I']*num_qubits
        for label, q in zip(short_generator, qubit_indices):
            generator[q] = label

        return Pauli(''.join(generator)[::-1])

    @staticmethod
    def matrix(pauli, x):
        return np.cos(x/2)*np.eye(2**pauli.num_qubits)-1j*pauli.to_matrix()*np.sin(x/2)

    def fix_parameter(self, parameter_value):
        # num_qubits = gate.num_qubits
        if np.allclose(parameter_value, 0):
            gate = QuantumCircuit(self.gate.num_qubits).to_gate(label='I')
        elif np.allclose(parameter_value, np.pi/2):
            # gate_pauli = PauliRotation.pauli_generator_from_gate(gate)
            # gate = PauliRotation.pauli_root(Pauli(gate_pauli))
            gate = self.pauli_root
        else:
            raise ValueError(f'Can only fix parameter to 0 or pi/2, got {parameter_value}.')
        return gate

    @staticmethod
    def pauli_root_from_pauli(pauli):
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

    # def pauli_operator(self):
    #     labels = ['I']*self.num_qubits
    #     for n, index in enumerate(self.qubit_indices):
    #         labels[index] = PauliRotation.pauli_gates[self.gate.name][1][n]
    #     labels = labels[::-1]  # Qiskits little-endian convention
    #     return Pauli(''.join(labels))


class Loss:
    """Loss function is always assumed to be of the form L=<0|U* H U|0>.

    However, for the unitary compilation problem U is a non-trivial function of the circuit."""
    def __init__(self, construct_circuit, hamiltonian):
        self.hamiltonian = hamiltonian
        self.construct_circuit = construct_circuit

    def evaluate_at(self, qc):
        full_qc = self.construct_circuit(qc)
        return Statevector(full_qc).expectation_value(self.hamiltonian.to_operator())

    @staticmethod
    def from_hamiltonian(H):
        assert np.allclose(H.conj().T, H), f'Hamiltonian is not hermitian. H =\n {H}'
        sop = SparsePauliOp.from_operator(H)
        loss_circuit = lambda qc: qc
        return Loss(loss_circuit, sop)

    @staticmethod
    def from_state(s):
        rho = DensityMatrix(s)
        return Loss.from_hamiltonian(rho.data)

    @staticmethod
    def from_unitary(u):

        num_qubits = u.num_qubits

        def loss_circuit(qc):
            assert qc.num_qubits == u.num_qubits, f'Number of qubits in the circuit {qc.num_qubits} does not match that in unitary {num_qubits}'
            qc_extended = Loss.hilbert_schmidt_circuit(num_qubits)
            return qc_extended.compose(qc, range(num_qubits))

        u_circuit = Loss.hilbert_schmidt_circuit(num_qubits).compose(u.to_instruction(), range(num_qubits))
        u_state = Statevector(u_circuit)
        u_rho = DensityMatrix(u_state)
        u_hamiltonian = SparsePauliOp.from_operator(u_rho)

        return Loss(loss_circuit, u_hamiltonian)

    @staticmethod
    def hilbert_schmidt_circuit(num_qubits):
        qc = CliffordPhi(2*num_qubits)
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.cx(i, i+num_qubits)
        return qc


def add_row_to_echelon(w, m):
    for row in m:
        if np.all(row == 0):
            continue
        i, = row.nonzero()
        i = i[0]
        if w[i]:
            w += row
    w %= 2

    if not np.all(w == 0):
        i, = w.nonzero()[0]
        for n, row in enumerate(m):
            if row[i]:
                m[n] = (row + w) % 2

    return np.concatenate([m, np.array([w])])


def is_linearly_dependent(pauli_x, pauli_echelon):
    # Pauli can be expanded into existing ones if it becomes zero in the echelon form.
    m = add_row_to_echelon(pauli_x, pauli_echelon)
    return np.all(m[-1] == 0)


class PauliSpan:
    def __init__(self, paulis):
        self.num_qubits = len(paulis[0])
        self.paulis = paulis
        self.matrix_form = np.array([self.projection_x(pauli) for pauli in self.paulis])
        self.normal_form = None
        self.starting_indices = None

    @staticmethod
    def projection_x(pauli):
        return pauli.x

    @staticmethod
    def add_row_to_normal_form(normal_form, starting_indices, new_row):

        # remove Nones
        rows = [row for row, si in zip(normal_form, starting_indices) if si is not None]
        indices = [si for si in starting_indices if si is not None]

        # sort
        rows = [row for _, row in sorted(zip(indices, rows))]
        indices = sorted(indices)

        # incomplete gaussian elimination
        for i, row in zip(indices, rows):
            if new_row[i]:
                new_row ^= row

        if not np.all(new_row == 0):
            i_last, = new_row.nonzero()
            i_last = i_last[0]
        else:
            i_last = None

        return np.concatenate([normal_form, np.array([new_row])]), starting_indices+[i_last]

    def compute_normal_form(self):
        matrix_form = self.matrix_form.copy()
        m0 = matrix_form[0]
        normal_form = np.array([m0])

        if not np.all(m0 == 0):
            i0, = m0.nonzero()
            i0 = i0[0]
        else:
            i0 = None

        starting_indices = [i0]

        for row in matrix_form[1:]:
            normal_form, starting_indices = self.add_row_to_normal_form(normal_form, starting_indices, row)

        self.normal_form = normal_form
        self.starting_indices = starting_indices

        return normal_form

    def is_dependent(self, observable, num_palis, num_paulis_previous=None):
        # If the pauli span of the last check is provided and the new rank is nor smaller, it's still dependent.
        if num_paulis_previous:
            if self.rank(num_paulis_previous) == self.rank(num_palis):
                return True
