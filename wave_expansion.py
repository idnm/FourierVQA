import copy

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Clifford, StabilizerState, Pauli, Statevector, Operator, \
    pauli_basis, random_pauli
from tqdm import tqdm

from experiments_utils import random_node_distribution, random_norm_distribution


# TODO
# - Estimate functional norm as well as block norm
# - Estimate number of terms from monte-carlo
# - Profile runtime. How many nodes are processed?


class PauliCircuit:
    def __init__(self, paulis, final_clifford=None, parameters=None):
        self.paulis = paulis
        self.final_clifford = final_clifford
        self.parameters = parameters

    @property
    def num_qubits(self):
        return self.paulis[0].num_qubits

    @property
    def num_paulis(self):
        return len(self.paulis)

    def default_parameters(self):
        return [Parameter(f'x{i}') for i in range(len(self.paulis))]

    def to_parameterized_circuit(self, default_parameters=False):
        if self.parameters is not None and default_parameters is False:
            parameters = self.parameters
        else:
            parameters = self.default_parameters()

        qc = QuantumCircuit(self.num_qubits)
        for pauli, parameter in zip(self.paulis, parameters):
            gate = PauliEvolutionGate(pauli, 1/2*parameter)
            qc.append(gate, range(qc.num_qubits))
        if self.final_clifford:
            qc.append(self.final_clifford.to_instruction(), range(qc.num_qubits))

        return qc

    @staticmethod
    def default_parameters_to_values_dict(parameters, values):
        parameter_indices = [int(p.name[1:]) for p in parameters]
        return {parameter: values[i] for parameter, i in zip(parameters, parameter_indices)}

    def expectation_value(self, observable, parameter_values):
        qc = self.to_parameterized_circuit(default_parameters=True)
        parameters_dict = self.default_parameters_to_values_dict(qc.parameters, parameter_values)

        state = Statevector(qc.bind_parameters(parameters_dict))
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
        pauli_rotation_values = np.array([PauliCircuit.pauli_rotation_matrix(pauli, x) for x in xx])

        if not np.allclose(gate_values, pauli_rotation_values):
            raise ValueError(f'Gate {gate.name} is not a Pauli rotation.')

        return pauli

    @staticmethod
    def pauli_rotation_matrix(pauli, x):
        return np.cos(x/2)*np.eye(2**pauli.num_qubits)-1j*pauli.to_matrix()*np.sin(x/2)

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
        if decomposition is None:
            return None
        return [self.independent_paulis[n] for n, coefficient in enumerate(decomposition) if coefficient]

    def decomposition_contains_pauli(self, decomposition, num_pauli):
        return num_pauli in self.list_decomposition(decomposition)

    def is_independent(self, num_pauli):
        return num_pauli in self.independent_paulis

    def decomposition_requires_paulis(self, decomposition):
        # if decomposition:
        if np.all(decomposition == 0):
            return 0
        return max(self.list_decomposition(decomposition))

    def compute_decomposition(self, observable, num_paulis):
        normal_form = self.normal_form[:num_paulis]
        decomposition_matrix = self.decomposition_matrix[:num_paulis]
        normal_form, decomposition_matrix = self.extend_normal_form(normal_form, decomposition_matrix, observable.x)

        # Decomposition successful.
        if np.all(normal_form[-1] == 0):
            return decomposition_matrix[-1]
        else:
            return None

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

        if num_pauli in self.dependent_paulis:
            # For dependent paulis decompositions are stored in the decomposition matrix
            update = self.decomposition_matrix[num_pauli]
        else:
            # For independent paulis,
            # decomposition vector is one-hot vector with True at the number of pauli in the basis.
            update = np.eye(self.dim, dtype=bool)[self.independent_paulis.index(num_pauli)]

        return decomposition ^ update


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

    @staticmethod
    def random(num_qubits, num_paulis, seed=0):
        np.random.seed(seed)
        seeds = np.random.randint(0, 10**6, 2)
        pauli_circuit = PauliCircuit.random(num_qubits, num_paulis, seed=seeds[0])
        observable = random_pauli(num_qubits, seed=seeds[1])
        return FourierComputation(pauli_circuit, observable)

    def run(self, check_admissible=True, max_order=None, verbose=True):

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
        progress_bar = tqdm(range(num_iterations), disable=not verbose)
        for _ in progress_bar:
            self.num_iterations += 1
            self.incomplete_nodes, self.complete_nodes = self.iteration(
                self.incomplete_nodes, self.complete_nodes, check_admissible)

            relative, absolute, remaining = self.status()
            progress_bar.set_postfix_str(f'(relative: {relative:.2%}, absolute: {absolute:.2g}, remaining: {remaining:.2g})')
            if len(self.incomplete_nodes) == 0:
                break

    def sample(self, num_samples=1000, seed=0):
        node_stats = np.zeros(self.pauli_space.num_paulis+1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32, num_samples)
        for seed in seeds:
            sample_level = self.single_sample(seed)
            node_stats[sample_level] += 1
        return node_stats

    def single_sample(self, seed):
        np.random.seed(seed)
        node = FourierComputationNode(self.pauli_circuit.num_paulis, self.observable, ())
        node.remove_commuting_paulis(self.pauli_space)
        branch_length = 0
        while node.num_paulis > 0:
            random_bool = np.random.randint(2)
            if random_bool:
                node.observable = node.observable.compose(self.pauli_space.paulis[node.num_paulis-1])
            node.num_paulis -= 1
            node.remove_commuting_paulis(self.pauli_space)
            branch_length += 1
        return branch_length

    def estimate_node_count(self, num_samples=1000, seed=0):
        norm_stats = self.sample(num_samples=num_samples, seed=seed)
        # Turn into a probability distribution
        norm_stats /= norm_stats.sum()
        norm_stats = np.trim_zeros(norm_stats, trim='b')

        node_stats = norm_stats * 2**np.arange(len(norm_stats))
        return node_stats.sum()

    def iteration(self, incomplete_nodes, complete_nodes, check_admissibility):
        if not incomplete_nodes:
            return [], complete_nodes

        new_incomplete_nodes = []
        for node in incomplete_nodes:
            incomplete_nodes, completed_nodes = node.branch_and_refine(self.pauli_space, check_admissibility)

            new_incomplete_nodes.extend(incomplete_nodes)
            complete_nodes.extend(completed_nodes)

        return new_incomplete_nodes, complete_nodes

    def status(self):
        norm_found = sum(self.norm_stats(only_nonzero=True))
        norm_remaining_bound = self.bound_remaining_norm()
        try:
            relative_norm_found = norm_found/(norm_found+norm_remaining_bound)
        except ZeroDivisionError:
            relative_norm_found = 1

        return relative_norm_found, norm_found, norm_remaining_bound

    def bound_remaining_norm(self):
        M = self.pauli_space.num_paulis
        orders = [node.order for node in self.incomplete_nodes]
        order_statistic = [orders.count(m) for m in range(M + 1)]
        return sum([n / (2 ** m) for m, n in enumerate(order_statistic)])

    def evaluate_loss_at(self, parameters):
        state0 = StabilizerState(Pauli('I'*self.pauli_circuit.num_qubits))
        results = [state0.expectation_value(node.observable)*node.monomial(parameters) for node in self.complete_nodes]

        return sum(results)

    def bound_remaining_loss(self, parameters):
        # Not trivial since |cos(x)|+|sin(x)|< sqrt(2) and grows.
        # Better estimates might be available.

        return None
        # return sum([np.abs(node.monomial(parameters)) for node in self.incomplete_nodes])

    def node_stats(self, only_nonzero=False):
        M = self.pauli_space.num_paulis
        if only_nonzero:
            orders = [node.order for node in self.complete_nodes if not np.allclose(node.expectation_value, 0)]
        else:
            orders = [node.order for node in self.complete_nodes]
        return [orders.count(m) for m in range(M + 1)]

    def norm_stats(self, only_nonzero=False):
        return [n / (2 ** m) for m, n in enumerate(self.node_stats(only_nonzero))]

    def visualize(self):
        M = self.pauli_space.num_paulis

        plt.scatter(range(M + 1), np.array(self.node_stats()) / (3 / 2) ** M)
        plt.scatter(range(M + 1), self.norm_stats())

        plt.plot(random_node_distribution(M), linestyle='--')
        plt.plot(random_norm_distribution(M), linestyle='--')


class FourierComputationNode:
    def __init__(self, num_paulis, observable, branch_history):
        self.num_paulis = num_paulis
        self.observable = observable
        self.branch_history = branch_history
        self.is_admissible = None
        self.observable_decomposition = None
        self.expectation_value = None

    @property
    def order(self):
        return len(self.branch_history)

    @property
    def is_complete(self):
        # The node is complete is no Paulis remain.
        node_is_complete = self.num_paulis == 0
        if node_is_complete:
            self.compute_expectation_value()
        return node_is_complete

    def __repr__(self):
        return f'{self.expectation_value} {self.num_paulis} {self.observable} {self.branch_history}'

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
                self.expectation_value = 0
                complete_nodes = [self]
                incomplete_nodes = []
                return incomplete_nodes, complete_nodes

            # If the branching Pauli is dependent, both branches are admissible.
            # If it is independent a single branch is admissible.
            idx_branching_pauli = self.num_paulis-1
            if pauli_space.is_independent(idx_branching_pauli):
                # If observable requires the branching Pauli in its decomposition the cos branch is not admissible.
                if pauli_space.decomposition_contains_pauli(self.observable_decomposition, idx_branching_pauli):
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

    def compute_expectation_value(self): 
        state = StabilizerState(Pauli('I'*self.observable.num_qubits))
        expectation_value = state.expectation_value(self.observable)
        self.expectation_value = expectation_value
        return expectation_value
